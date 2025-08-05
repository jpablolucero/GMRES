#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <GMRES.hpp>

template <typename T>
bool JacobiPrecond()
{
  std::cout << std::endl ;
  std::cout << "JacobiPrecond:JacobiPrecond ------------- BEGIN ---------------" << std::endl ;
  std::cout << std::endl ;

  std::vector<std::vector<std::pair<std::size_t, T>>> A(6);

  A[0] = {{0,  4.0}, {1, -1.0}, {3, -1.0}};
  A[1] = {{0, -1.0}, {1,  4.0}, {2, -1.0}, {4, -1.0}};
  A[2] = {{1, -1.0}, {2,  4.0}, {5, -1.0}};
  A[3] = {{0, -1.0}, {3,  4.0}, {4, -1.0}};
  A[4] = {{1, -1.0}, {3, -1.0}, {4,  4.0}, {5, -1.0}};
  A[5] = {{2, -1.0}, {4, -1.0}, {5,  4.0}};

  std::vector<T> refSol(6, 0.0);
  refSol[0] = 1.0;
  refSol[1] = 2.0;
  refSol[2] = 3.0;
  refSol[3] = 4.0;
  refSol[4] = 5.0;
  refSol[5] = 6.0;

  std::vector<T> rhs(6, 0.0);
  for (std::size_t i = 0; i < A.size(); ++i)
    for (const auto& [j, val] : A[i])
      rhs[i] += val * refSol[j];

  std::vector<T> sol(6, 0.0);

  auto op = [&](const std::vector<T>& in)
  {
    std::vector<T> out(in.size(), 0.0);
    for (std::size_t i = 0; i < A.size(); ++i)
      for (const auto& [j, val] : A[i])
        out[i] += val * in[j];
    return out;
  };

  std::vector<T> diag(6);
  for (std::size_t i = 0; i < A.size(); ++i)
    for (const auto& [j, val] : A[i])
      if (j == i) diag[i] = val;

  auto inner_product = [](const auto& a,const auto& b)
  {
    return std::inner_product(a.begin(),a.end(),b.begin(),0.0);
  };
  
  auto preconditioner = [diag](const std::vector<T>& v)
  {
    std::vector<T> out(v.size(), 0.0);
    for (std::size_t i = 0; i < v.size(); ++i)
      out[i] = v[i] / diag[i];
    return out;
  };
  
  GMRES solver(op, inner_product, preconditioner);
  auto parameters = solver.getParameters();
  parameters.max_iter = 1E9;
  parameters.restart_iter = 30;
  parameters.tol = 1.E-6;
  solver.setParameters(parameters);

  auto res = solver(rhs, sol);

  std::cout << "Iterations: " << res.first << "\t" << "Residual Reduction: " << std::scientific << res.second << std::endl ;

  if ((res.first != 5) || (res.second > 100.*std::numeric_limits<T>::epsilon()))
    {
      std::cout << std::endl ;
      std::cout << "JacobiPrecond:JacobiPrecond ------------- FAIL ---------------" << std::endl ;
      std::cout << std::endl ;
      return false;
    }
  std::cout << std::endl ;
  std::cout << "JacobiPrecond:JacobiPrecond ------------- SUCCESS ---------------" << std::endl ;
  std::cout << std::endl ;
  return true;

}

int main(int argc, char **argv)
{
  if (JacobiPrecond<float>()&&JacobiPrecond<double>()&&JacobiPrecond<long double>())
    {
      std::cout << std::endl ;
      std::cout << "JacobiPrecond ------------- SUCCESS ---------------" << std::endl ;
      std::cout << std::endl ;
      return 0;
    }
  else
    {
      std::cout << std::endl ;
      std::cout << "JacobiPrecond ------------- FAIL ---------------" << std::endl ;
      std::cout << std::endl ;
      return -1;
    }
  
}
