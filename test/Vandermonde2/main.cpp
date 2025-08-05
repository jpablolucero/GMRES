#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <GMRES.hpp>

template <typename T>
bool Vandermonde2()
{
  std::cout << std::endl ;
  std::cout << "Vandermonde2:Vandermonde2 ------------- BEGIN ---------------" << std::endl ;
  std::cout << std::endl ;

  std::vector<T> rhs(4, 0.0);
  rhs[0] =  -6.0;
  rhs[1] =   2.0;
  rhs[2] =  12.0;
  rhs[3] = -10.0;

  std::vector<std::vector<T>> A(4, std::vector<T>(4, 0.0));
  A[0][0] =  1.0; A[0][1] = 1.0; A[0][2] = 1.0; A[0][3] = 1.0;
  A[1][0] =  8.0; A[1][1] = 4.0; A[1][2] = 2.0; A[1][3] = 1.0;
  A[2][0] = 64.0; A[2][1] =16.0; A[2][2] = 4.0; A[2][3] = 1.0;
  A[3][0] = 27.0; A[3][1] = 9.0; A[3][2] = 3.0; A[3][3] = 1.0;

  std::vector<T> sol(4, 0.0);

  std::vector<T> refSol(4, 0.0);
  refSol[0] =   9.0;
  refSol[1] = -64.0;
  refSol[2] = 137.0;
   refSol[3] = -88.0;

  auto op = [&](const std::vector<T>& in)
  {
    std::vector<T> out(in.size(), 0.0);
    for (std::size_t i = 0 ; i < A.size() ; i++)
      for (std::size_t j = 0 ; j < A[0].size() ; j++)
        out[i] += A[i][j] * in[j];
    return out;
  };

  GMRES solver(op);

  auto parameters = solver.getParameters();
  parameters.max_iter = 1E9;
  parameters.restart_iter = 30;
  parameters.tol = 1.E-8;
  solver.setParameters(parameters);

  auto res = solver(rhs, sol);

  std::cout << "Iterations: " << res.first << "\t" << "Residual Reduction: " << std::scientific << res.second << std::endl;

  if ((res.first != 4) || res.second > 100.*std::numeric_limits<T>::epsilon())
    {
      std::cout << std::endl ;
      std::cout << "Vandermonde2:Vandermonde2 ------------- FAIL ---------------" << std::endl ;
      std::cout << std::endl ;
      return false;
    }
  std::cout << std::endl ;
  std::cout << "Vandermonde2:Vandermonde2 ------------- SUCCESS ---------------" << std::endl ;
  std::cout << std::endl ;
  return true;

}


int main(int argc, char **argv)
{
  if (Vandermonde2<float>()&&Vandermonde2<double>()&&Vandermonde2<long double>())
    {
      std::cout << std::endl ;
      std::cout << "Vandermonde2 ------------- SUCCESS ---------------" << std::endl ;
      std::cout << std::endl ;
      return 0;
    }
  else
    {
      std::cout << std::endl ;
      std::cout << "Vandermonde2 ------------- FAIL ---------------" << std::endl ;
      std::cout << std::endl ;
      return -1;
    }
  
}

