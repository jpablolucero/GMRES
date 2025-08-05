#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <GMRES.hpp>

template <typename T>
bool Functor()
{
  std::cout << std::endl ;
  std::cout << "Functor:Functor ------------- BEGIN ---------------" << std::endl ;
  std::cout << std::endl ;

  std::vector<T> rhs(3, 0.0);
  rhs[0] = 14.0;
  rhs[1] = 32.0;
  rhs[2] = 50.0;

  std::vector<std::vector<T>> A(3, std::vector<T>(3, 0.0));
  A[0][0] = 1.0; A[0][1] = 2.0; A[0][2] = 3.0;
  A[1][0] = 4.0; A[1][1] = 5.0; A[1][2] = 6.0;
  A[2][0] = 7.0; A[2][1] = 8.0; A[2][2] = 9.0;

  std::vector<T> sol(3, 0.0);

  std::vector<T> refSol(3, 0.0);
  refSol[0] = 1.0;
  refSol[1] = 2.0;
  refSol[2] = 3.0;

  auto op = [&](const std::vector<T>& in)
  {
    std::vector<T> out(in.size(), 0.0);
    for (std::size_t i = 0 ; i < A.size() ; i++)
      for (std::size_t j = 0 ; j < A[0].size() ; j++)
        out[i] += A[i][j] * in[j];
    return out;
  };

  GMRES solver(op);  // using default inner product

  auto parameters = solver.getParameters();
  parameters.max_iter = 1E9;
  parameters.restart_iter = 30;
  parameters.tol = 1.E-6;
  solver.setParameters(parameters);

  auto res = solver(rhs, sol);

  std::cout << "Iterations: " << res.first << "\t" << "Residual Reduction: " << std::scientific << res.second << std::endl ;

  if ((res.first != 2) || (res.second > 100.*std::numeric_limits<T>::epsilon()))
    {
      std::cout << std::endl ;
      std::cout << "Functor:Functor ------------- FAIL ---------------" << std::endl ;
      std::cout << std::endl ;
      return false;
    }

  std::cout << std::endl ;
  std::cout << "Functor:Functor ------------- SUCCESS ---------------" << std::endl ;
  std::cout << std::endl ;
  return true;
}

int main()
{
  if (Functor<float>()&&Functor<double>()&&Functor<long double>())
    {
      std::cout << std::endl ;
      std::cout << "Functor ------------- SUCCESS ---------------" << std::endl ;
      std::cout << std::endl ;
      return 0;
    }
  else
    {
      std::cout << std::endl ;
      std::cout << "Functor ------------- FAIL ---------------" << std::endl ;
      std::cout << std::endl ;
      return -1;
    }
}

