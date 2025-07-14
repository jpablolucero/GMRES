#pragma once
#include <vector>
#include <concepts>
#include <type_traits>
#include <algorithm>
#include <numeric>
#include "solver_tools.hpp"

template <class M,
	  class InnerProduct = default_inner_product,
	  class Preconditioner = identity_precond<M>>
class gmres {
  public:

  struct param {
    int max_iter = 30;
    int restart_iter = 30;
    double tol = 1e-6;
  };

  typedef std::pair<int, double> return_type;

  gmres(const M& A)
    : A_(A), inner_product_(getDefaultInnerProduct()), PInv_(A) {}

  gmres(const M& A, const InnerProduct & inner_product)
    : A_(A), inner_product_(inner_product), PInv_(getDefaultPInv()) {}
  
  gmres(const M& A, const InnerProduct & inner_product, const Preconditioner & PInv)
    : A_(A), inner_product_(inner_product), PInv_(PInv) {}

  gmres(const M& A, const InnerProduct & inner_product, const Preconditioner && PInv)
    : A_(A), inner_product_(inner_product), PInv_(std::move(PInv)) {}

  gmres(const M& A, const param& p)
    : A_(A), inner_product_(getDefaultInnerProduct()), PInv_(A), param_(p) {}

  gmres(const M& A, const InnerProduct & inner_product, const param& p)
    : A_(A), inner_product_(inner_product), PInv_(getDefaultPInv()), param_(p) {}

  gmres(const M& A, const InnerProduct & inner_product, const Preconditioner & PInv, const param& p)
    : A_(A), inner_product_(inner_product), PInv_(PInv), param_(p) {}

  gmres(const M& A, const InnerProduct & inner_product, const Preconditioner && PInv, const param& p)
    : A_(A), inner_product_(inner_product), PInv_(std::move(PInv)), param_(p) {}

  gmres(const gmres&) = delete;
  gmres(gmres&&) = delete;
  gmres& operator=(const gmres&) = delete;
  gmres& operator=(gmres&&) = delete;

  param get_param() const {
    return param_;
  }

  void set_param(const param& p) {
    param_ = p;
  }

  template<class V>
  return_type operator()(const V& b, V& x) const
  {
    static_assert(ValidVectorType<V>,"Not a valid vector class");
    static_assert(ValidInnerProduct<InnerProduct, V>,
                  "InnerProduct is not valid for this vector type");
    return gmres_impl<M,InnerProduct,Preconditioner,V>
      (A_, inner_product_, PInv_, b, x,
       param_.max_iter,
       param_.restart_iter,
       param_.tol);
  }

private:
  const M& A_; 
  const InnerProduct& inner_product_;
  const Preconditioner& PInv_;
  param param_;

  static const InnerProduct& getDefaultInnerProduct()
  {
    static const InnerProduct ip{};
    return ip;
  }

  static const Preconditioner& getDefaultPInv()
  {
    static const identity_precond<M> ip{};
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
void apply_givens_rotation(auto& h_col, std::vector<T>& cs, std::vector<T>& sn,
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

template<class Op, class InnerProduct, class PrecOp, class V>
auto
gmres_impl(const Op& A, const InnerProduct& inner_product, const PrecOp& PInv, const V& b, V& x,
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

  natural n_global = static_cast<natural>(inner_product(ones,ones)+0.5);
  natural max_iter = max_iter_ ;
  natural restart_iter = std::min(n_global, static_cast<natural>(restart_iter_));

  auto norm_2 = [&](const auto & vin)
  {
    return std::sqrt(inner_product(vin,vin));
  };

  V r = A(x);
  std::for_each(std::begin(idx), std::end(idx),
		[&r, &b](std::size_t i) {
		  r[i] = b[i] - r[i];
		});
 
  r = PInv(r);
  real r_norm = norm_2(r);
  real b_norm = norm_2(PInv(b));

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

	  Q[j+1] = PInv(A(Q[j]));
	  idx_j.push_back(j);

	  // Naturally sequential
	  for (natural k = 0; k <= j; ++k)
	    {
	      H[j][k] = inner_product(Q[k], Q[j+1]);
	      std::for_each(idx.begin(), idx.end(),
			    [&Q, &H, &j, &k](natural i)
			    {
			      Q[j + 1][i] -= Q[k][i] * H[j][k];
			    });
	    }

	  H[j][j+1] = std::sqrt(inner_product(Q[j+1], Q[j+1]));

	  std::for_each(idx.begin(), idx.end(),
			[&Q, &H, &j](natural i)
			{
			  Q[j + 1][i] /= H[j][j + 1];
			});

	  apply_givens_rotation(H[j], cs, sn, j);

	  beta[j+1] = sn[j]*beta[j];
	  beta[j]   = cs[j]*beta[j];
	  
	  error = std::abs(beta[j+1]) / b_norm;

	  std::cout << "It: " << num_iter << ". Err: " << std::scientific << error << std::endl ;

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
      r = PInv(r);
      r_norm = norm_2(r);
      error  = norm_2(r) / b_norm;

      if (error < tol)
	return std::make_pair(num_iter, error);
      
    }

  return std::make_pair(max_iter, error);
  
}
