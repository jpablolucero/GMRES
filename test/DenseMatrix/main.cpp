#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <numeric>
#include "gmres.hpp"

int main()
{
  std::vector<double> rhs(3,0.0);
  rhs[0] = 14.0;
  rhs[1] = 32.0;
  rhs[2] = 50.0;

  std::vector<std::vector<double>> A(3,std::vector<double>(3,0.0));
  A[0][0] = 1.0;
  A[0][1] = 2.0;
  A[0][2] = 3.0;
  A[1][0] = 4.0;
  A[1][1] = 5.0;
  A[1][2] = 6.0;
  A[2][0] = 7.0;
  A[2][1] = 8.0;
  A[2][2] = 9.0;

  std::vector<double> sol(3, 0.0);

  std::vector<double> refSol(3, 0.0);
  refSol[0] = 1.0;
  refSol[1] = 2.0;
  refSol[2] = 3.0;

  auto op = [&](const std::vector<double>& in)
  {
    std::vector<double> out(in.size(),0.0);
    for (std::size_t i = 0 ; i < A.size() ; i++)
      for (std::size_t j = 0 ; j < A[0].size() ; j++)
	out[i] += A[i][j] * in[j];
    return out;
  };

  gmres solver(op);

  auto param = solver.get_param();
  param.max_iter = 1E9;
  param.restart_iter = 30;
  param.tol = 1.E-8;
  solver.set_param(param);

  auto res = solver(rhs, sol);

  std::cout << "Iterations: " << res.first << "\t" << "Residual Reduction: " << res.second << std::endl ;

  if ((res.first == 2) and (res.second < 1.E-14))
    return 0;
  else
    return -1;
}

