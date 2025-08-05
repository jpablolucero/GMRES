#pragma once
#include <vector>
#include <concepts>
#include <type_traits>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <solver_tools.hpp>

template <typename M,
	  typename InnerProduct = DefaultInnerProduct,
	  typename Preconditioner = IdentityPreconditioner<M>>
class GMRES
{
public:

  struct Parameters
  {
    std::size_t max_iter = 100;
    std::size_t restart_iter = 30;
    double tol = 1e-6;
  };

  GMRES(const M& A)
    : A_(A), innerProduct_(getDefaultInnerProduct()), preconditioner_(getDefaultpreconditioner()) {}

  GMRES(const M& A, const InnerProduct & innerProduct)
    : A_(A), innerProduct_(innerProduct), preconditioner_(getDefaultpreconditioner()) {}
  
  GMRES(const M& A, const InnerProduct & innerProduct, const Preconditioner & preconditioner)
    : A_(A), innerProduct_(innerProduct), preconditioner_(preconditioner) {}

  GMRES(const M& A, const Parameters& p)
    : A_(A), innerProduct_(getDefaultInnerProduct()), preconditioner_(getDefaultpreconditioner()), parameters_(p) {}

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

  template<typename V>
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

template<typename real>
void applyGivensRotation(auto& h_col, std::vector<real>& cs, std::vector<real>& sn,
			 typename std::vector<real>::size_type j) {

  auto givens_rotation = [&](real x1, real x2) -> std::pair<real, real>
  {
    real d = std::sqrt(x1 * x1 + x2 * x2);
    if (d < 10*std::numeric_limits<real>::epsilon()) return std::make_pair(real(1), real(0));
    return std::make_pair(x1 / d, -x2 / d);
  };

  // Not parallelizable because of data dependency
  for (std::size_t i = 0; i < j; ++i)
    {
      real tmp = cs[i] * h_col[i] - sn[i] * h_col[i + 1];
      h_col[i + 1] = sn[i] * h_col[i] + cs[i] * h_col[i + 1];
      h_col[i] = tmp;
    }
  if (j < h_col.size() - 1)
    {
      std::tie(cs[j], sn[j]) = givens_rotation(h_col[j], h_col[j + 1]);
      h_col[j] = cs[j] * h_col[j] - sn[j] * h_col[j + 1];
      h_col[j + 1] = real(0);
    }
  else
    {
      cs[j] = real(1.0);
      sn[j] = real(0);
    }

}

template <typename real>
auto solveUpperTriangular(const std::vector<std::vector<real>> & U,
			  auto b)
{

  std::size_t n = b.size();

  std::vector<real> x(n,0.0);

  for (int i = n - 1; i >= 0; --i)
    {
      real sum = 0.0; 
      for (std::size_t j = i + 1; j < n; ++j)
	sum += U[j][i] * x[j];
      x[i] = (b[i] - sum) / U[i][i];
    }
  
  return std::move(x);
}

template<typename Op, typename InnerProduct, typename Preconditioner, typename V>
std::pair<typename V::size_type,typename V::value_type>
GMRESImplementation(const Op& A, const InnerProduct& innerProduct, const Preconditioner& preconditioner, const V& b, V& x,
		    typename V::size_type max_iter_, typename V::size_type restart_iter_,
		    typename V::value_type tol)
{
  typedef typename V::value_type real;
  typedef typename V::size_type natural;

  const natural n_local = x.size();
  std::vector<natural> idx(n_local); std::iota(idx.begin(),idx.end(),0);

  V ones = x;
  std::for_each(std::begin(idx), std::end(idx),
		[&ones](std::size_t i)
		{
		  ones[i] = 1.0;
		});

  const natural n_global = static_cast<natural>(innerProduct(ones,ones)+0.5);
  const natural max_iter = max_iter_ ;
  const natural restart_iter = std::min(n_global, static_cast<natural>(restart_iter_));

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
	  if (Q.size() < restart_iter + 1) Q.push_back(V(n_local));

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

	  std::vector<real> tmp(j+1, 0.0);

	  for (natural k = 0; k <= j; ++k) {
	    tmp[k] = innerProduct(Q[k], Q[j+1]);
	    std::for_each(idx.begin(), idx.end(),
			  [&Q, &j, &k, &tmp](natural i) {
			    Q[j+1][i] -= Q[k][i] * tmp[k];
			  });
	    H[j][k] += tmp[k];
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
