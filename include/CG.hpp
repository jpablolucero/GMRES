#pragma once
#include <vector>
#include <concepts>
#include <type_traits>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <solver_tools.hpp>

template <class M,
	  class InnerProduct = DefaultInnerProduct,
	  class Preconditioner = IdentityPreconditioner<M>>
class CG
{
  public:

  struct Parameters
  {
    std::size_t max_iter = 100;
    std::size_t restart_iter = 30;
    double tol = 1e-6;
  };

  CG(const M& A)
    : A_(A), innerProduct_(getDefaultInnerProduct()), preconditioner_(getDefaultpreconditioner()) {}

  CG(const M& A, const InnerProduct & innerProduct)
    : A_(A), innerProduct_(innerProduct), preconditioner_(getDefaultpreconditioner()) {}
  
  CG(const M& A, const InnerProduct & innerProduct, const Preconditioner & preconditioner)
    : A_(A), innerProduct_(innerProduct), preconditioner_(preconditioner) {}

  CG(const M& A, const Parameters& p)
    : A_(A), innerProduct_(getDefaultInnerProduct()), preconditioner_(getDefaultpreconditioner()), parameters_(p) {}

  CG(const M& A, const InnerProduct & innerProduct, const Parameters& p)
    : A_(A), innerProduct_(innerProduct), preconditioner_(getDefaultpreconditioner()), parameters_(p) {}

  CG(const M& A, const InnerProduct & innerProduct, const Preconditioner & preconditioner, const Parameters& p)
    : A_(A), innerProduct_(innerProduct), preconditioner_(preconditioner), parameters_(p) {}

  CG(const CG&) = delete;
  CG(CG&&) = delete;
  CG& operator=(const CG&) = delete;
  CG& operator=(CG&&) = delete;

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
    return CGImplementation<M,InnerProduct,Preconditioner,V>
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

template<class Op, class InnerProduct, class Preconditioner, class V>
auto
CGImplementation(const Op& A, const InnerProduct& innerProduct, const Preconditioner& preconditioner, const V& b, V& x,
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
 
  real b_norm = norm_2(b);
  if (std::abs(b_norm) < 1.E-14)
    {
      std::fill(x.begin(),x.end(),0.0);
      return std::make_pair(natural(0), 0.);
    }

  V z = preconditioner(r);
  real error = norm_2(r) / b_norm;
  if (error < tol) return std::make_pair(natural(0), error);

  V p = z;
  real r_sq_old = innerProduct(r, z);

  natural num_iter = 1;
  for (; num_iter <= max_iter; ++num_iter)
    {
      V Ap = A(p);
      real alpha = r_sq_old / innerProduct(p, Ap);
      std::for_each(idx.begin(),idx.end(),
		    [&x,&alpha,&p](std::size_t i)
		    {
		      x[i] += alpha * p[i];
		    });
      std::for_each(idx.begin(),idx.end(),
		    [&r,&alpha,&Ap](std::size_t i)
		    {
		      r[i] -= alpha * Ap[i];
		    });
      z = preconditioner(r);
      real r_sq_new = innerProduct(r, z);
      error = norm_2(r) / b_norm;
      if (error < tol)
	return std::make_pair(num_iter, error);
      std::for_each(idx.begin(),idx.end(),
		    [&p,r_quotient=r_sq_new / r_sq_old,&z](std::size_t i)
		    {
		      p[i] = z[i] + r_quotient * p[i];
		    });
      r_sq_old = r_sq_new;
    }

  return std::make_pair(n_global,norm_2(r)/b_norm);
}
