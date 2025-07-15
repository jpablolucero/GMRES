#pragma once
#include <vector>
#include <concepts>
#include <type_traits>
#include <algorithm>
#include <numeric>

template <typename T>
concept HasUnsignedSizeType = requires
  {
    typename T::size_type;
    requires std::is_unsigned_v<typename T::size_type>;
  };

template <typename T>
concept HasSizeFunction = requires(T t)
  {
    { t.size() } -> std::same_as<typename T::size_type>;
  };

template <typename T>
concept ConstructibleFromInteger = requires(T, typename T::size_type x)
  {
    requires std::is_integral_v<typename T::size_type>;
    { T(x) };
  };

template <typename T>
concept HasFloatingPointValueType = requires
  {
    typename T::value_type;
    requires std::is_floating_point_v<typename T::value_type>;
  };

template <typename T>
concept HasIndexOperator = requires(T t, typename T::size_type i)
  {
    { t[i] } -> std::convertible_to<typename T::value_type>;
  };

template <typename T>
concept ValidVectorType
= HasUnsignedSizeType<T> &&
  HasSizeFunction<T> &&
  ConstructibleFromInteger<T> &&
  HasFloatingPointValueType<T> &&
  HasIndexOperator<T>;

template <typename InnerProduct, typename V>
concept ValidInnerProduct =
ValidVectorType<V> &&
requires(InnerProduct inner_product, const V& v1, const V& v2)
  {
    { inner_product(v1, v2) } -> std::convertible_to<typename V::value_type>;
  };

template <class M>
class IdentityPreconditioner
{
public:
  IdentityPreconditioner() {}
  IdentityPreconditioner(const M&) {}
  
  template<class V>
  const V& operator()(const V& ve) const
  {
    return ve;
  }
};

struct DefaultInnerProduct
{
  template <ValidVectorType V>
  typename V::value_type operator()(const V & v1, const V & v2) const
  {
    typename V::value_type result = 0.0;
    for (typename V::size_type i = 0; i < v1.size(); ++i)
      result += v1[i] * v2[i];
    return result;
  }
};

template <class M,
	  class InnerProduct = DefaultInnerProduct,
	  class Preconditioner = IdentityPreconditioner<M>>
class GMRES
{
public:

  struct Parameters
  {
    std::size_t max_iter = 30;
    std::size_t restart_iter = 30;
    double tol = 1e-6;
  };

  GMRES(const M& A)
    : A_(A), innerProduct_(getDefaultInnerProduct()), preconditioner_(A) {}

  GMRES(const M& A, const InnerProduct & innerProduct)
    : A_(A), innerProduct_(innerProduct), preconditioner_(getDefaultpreconditioner()) {}
  
  GMRES(const M& A, const InnerProduct & innerProduct, const Preconditioner & preconditioner)
    : A_(A), innerProduct_(innerProduct), preconditioner_(preconditioner) {}

  GMRES(const M& A, const Parameters& p)
    : A_(A), innerProduct_(getDefaultInnerProduct()), preconditioner_(A), parameters_(p) {}

  GMRES(const M& A, const InnerProduct & innerProduct, const Parameters& p)
    : A_(A), innerProduct_(innerProduct), preconditioner_(getDefaultpreconditioner()), parameters_(p) {}

  GMRES(const M& A, const InnerProduct & innerProduct, const Preconditioner & preconditioner, const Parameters& p)
    : A_(A), innerProduct_(innerProduct), preconditioner_(preconditioner), parameters_(p) {}

  GMRES(const GMRES&) = delete;
  GMRES(GMRES&&) = delete;
  GMRES& operator=(const GMRES&) = delete;
  GMRES& operator=(GMRES&&) = delete;

  Parameters getParameters() const
  {
    return parameters_;
  }

  void setParameters(const Parameters& p)
  {
    parameters_ = p;
  }

  template<class V>
  std::pair<std::size_t, double> operator()(const V& b, V& x) const
  {
    static_assert(ValidVectorType<V>,"Not a valid vector class");
    static_assert(ValidInnerProduct<InnerProduct, V>,
                  "InnerProduct is not valid for this vector type");
    return GMRESImplementation<M,InnerProduct,Preconditioner,V>
      (A_, innerProduct_, preconditioner_, b, x,
       parameters_.max_iter,
       parameters_.restart_iter,
       parameters_.tol);
  }

private:
  const M& A_; 
  const InnerProduct& innerProduct_;
  const Preconditioner& preconditioner_;
  Parameters parameters_;

  static const InnerProduct& getDefaultInnerProduct()
  {
    static const DefaultInnerProduct ip{};
    return ip;
  }

  static const Preconditioner& getDefaultpreconditioner()
  {
    static const IdentityPreconditioner<M> ip{};
    return ip;
  }

};

template<typename T>
std::pair<T, T> givens_rotation(T x1, T x2)
{
  T d = std::sqrt(x1 * x1 + x2 * x2);
  if (d == T(0)) return std::make_pair(T(1), T(0));
  return std::make_pair(x1 / d, -x2 / d);
}

template<class T>
void applyGivensRotation(auto& h_col, std::vector<T>& cs, std::vector<T>& sn,
                           typename std::vector<T>::size_type j) {
  using size_type = typename std::vector<T>::size_type;
  using value_type = T;

  // Not parallelizable because of data dependency
  for (size_type i = 0; i < j; ++i)
    {
      value_type tmp = cs[i] * h_col[i] - sn[i] * h_col[i + 1];
      h_col[i + 1] = sn[i] * h_col[i] + cs[i] * h_col[i + 1];
      h_col[i] = tmp;
    }
  if (j < h_col.size() - 1)
    {
      std::tie(cs[j], sn[j]) = givens_rotation(h_col[j], h_col[j + 1]);
      h_col[j] = cs[j] * h_col[j] - sn[j] * h_col[j + 1];
      h_col[j + 1] = value_type(0);
    }
  else
    {
      cs[j] = value_type(1.0);
      sn[j] = value_type(0);
    }

}

auto solveUpperTriangular(const auto & U,
			  auto b)
{

  using size_t = typename decltype(b)::size_type ;
  using value_t = typename decltype(b)::value_type ;

  size_t n = b.size();

  std::vector<double> x(n,0.0);

  for (int i = n - 1; i >= 0; --i)
    {
      value_t sum = 0.0; 
      for (size_t j = i + 1; j < n; ++j)
	sum += U[j][i] * x[j];
      x[i] = (b[i] - sum) / U[i][i];
    }
  
  return std::move(x);
}

template<class Op, class InnerProduct, class Preconditioner, class V>
auto
GMRESImplementation(const Op& A, const InnerProduct& innerProduct, const Preconditioner& preconditioner, const V& b, V& x,
		    typename V::size_type max_iter_, typename V::size_type restart_iter_,
		    typename V::value_type tol)
{
  typedef typename V::value_type real;
  typedef typename V::size_type natural;


  natural n_local = x.size();
  std::vector<natural> idx(n_local); std::iota(idx.begin(),idx.end(),0);

  V ones = x;
  std::for_each(std::begin(idx), std::end(idx),
		[&ones](std::size_t i)
		{
		  ones[i] = 1.0;
		});

  natural n_global = static_cast<natural>(innerProduct(ones,ones)+0.5);
  natural max_iter = max_iter_ ;
  natural restart_iter = std::min(n_global, static_cast<natural>(restart_iter_));

  auto norm_2 = [&](const auto & vin)
  {
    return std::sqrt(innerProduct(vin,vin));
  };

  V r = A(x);
  std::for_each(std::begin(idx), std::end(idx),
		[&r, &b](std::size_t i) {
		  r[i] = b[i] - r[i];
		});
 
  r = preconditioner(r);
  real r_norm = norm_2(r);
  real b_norm = norm_2(preconditioner(b));

  if (std::abs(b_norm) < 1.E-14)
    {
      std::fill(x.begin(),x.end(),0.0);
      return std::make_pair(natural(0), 0.);
    }

  real error = norm_2(r) / b_norm;
  if (error < tol) return std::make_pair(natural(0), error);

  std::vector<real> sn(restart_iter, 0);
  std::vector<real> cs(restart_iter, 0);

  std::vector<V> Q;
  std::vector<std::vector<real>> H;

  natural num_iter = 1;
  while (num_iter <= max_iter)
    {

      if(Q.size() == 0) Q.push_back(V(n_local));
      
      std::for_each(idx.begin(), idx.end(),
		    [&](natural i) {
		      Q[0][i] = r[i] / r_norm;
		    });

      std::vector<real> beta(restart_iter + 1,0.);
      beta[0] = r_norm;

      natural j = 0;
      std::vector<natural> idx_j;

      // This is a naturally sequential loop, can't use std algorithms
      for (; j < restart_iter && num_iter <= max_iter; ++j, ++num_iter)
	{

	  if (H.size() < restart_iter) H.push_back(std::vector<real>(H.size()+2,0.0));
	  else std::fill(H[j].begin(),H[j].end(),0.0);
	  if (Q.size() < restart_iter + 1) Q.push_back(V(n_local,0.0));

	  Q[j+1] = preconditioner(A(Q[j]));
	  idx_j.push_back(j);

	  // Naturally sequential
	  for (natural k = 0; k <= j; ++k)
	    {
	      H[j][k] = innerProduct(Q[k], Q[j+1]);
	      std::for_each(idx.begin(), idx.end(),
			    [&Q, &H, &j, &k](natural i)
			    {
			      Q[j + 1][i] -= Q[k][i] * H[j][k];
			    });
	    }

	  H[j][j+1] = std::sqrt(innerProduct(Q[j+1], Q[j+1]));

	  std::for_each(idx.begin(), idx.end(),
			[&Q, &H, &j](natural i)
			{
			  Q[j + 1][i] /= H[j][j + 1];
			});

	  applyGivensRotation(H[j], cs, sn, j);

	  beta[j+1] = sn[j]*beta[j];
	  beta[j]   = cs[j]*beta[j];
	  
	  error = std::abs(beta[j+1]) / b_norm;

	  if (error <= tol)
	    {
	      std::vector<real> beta_proj(beta.begin(), beta.begin()+j+1);
	      auto sol = solveUpperTriangular(H, beta_proj);
	      std::for_each(idx.begin(), idx.end(),
			    [&x, &Q, &sol, &idx_j](natural i)
			    {
			      x[i] += std::transform_reduce(idx_j.begin(), idx_j.end(),
							    0.0,
							    std::plus<>(),
							    [&](natural k) {
							      return Q[k][i] * sol[k];
							    });
			    });
	      return std::make_pair(num_iter, error);
	    }
 	}
      
      std::vector<real> beta_proj(beta.begin(), beta.begin()+j);
      auto sol = solveUpperTriangular(H, beta_proj);
      std::for_each(idx.begin(), idx.end(),
		    [&x, &Q, &sol, &idx_j](natural i)
		    {
		      x[i] += std::transform_reduce(idx_j.begin(), idx_j.end(),
						    0.0,
						    std::plus<>(),
						    [&](natural k) {
						      return Q[k][i] * sol[k];
						    });
		    });
      r = A(x);
      std::for_each(std::begin(idx), std::end(idx),
		    [&r, &b](std::size_t i) {
		      r[i] = b[i] - r[i];
		    });
      r = preconditioner(r);
      r_norm = norm_2(r);
      error  = norm_2(r) / b_norm;

      if (error < tol)
	return std::make_pair(num_iter, error);
      
    }

  return std::make_pair(max_iter, error);
  
}
