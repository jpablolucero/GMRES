#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <numeric>
#include "gmres.hpp"

int main()
{
  std::vector<double> rhs(4, 0.0);
  rhs[0] =  -6.0;
  rhs[1] =   2.0;
  rhs[2] =  12.0;
  rhs[3] = -10.0;

  std::vector<std::vector<double>> A(4, std::vector<double>(4, 0.0));
  A[0][0] =  1.0; A[0][1] = 1.0; A[0][2] = 1.0; A[0][3] = 1.0;
  A[1][0] =  8.0; A[1][1] = 4.0; A[1][2] = 2.0; A[1][3] = 1.0;
  A[2][0] = 64.0; A[2][1] =16.0; A[2][2] = 4.0; A[2][3] = 1.0;
  A[3][0] = 27.0; A[3][1] = 9.0; A[3][2] = 3.0; A[3][3] = 1.0;

  std::vector<double> sol(4, 0.0);

  std::vector<double> refSol(4, 0.0);
  refSol[0] =   9.0;
  refSol[1] = -64.0;
  refSol[2] = 137.0;
  refSol[3] = -88.0;

  auto op = [&](const std::vector<double>& in)
  {
    std::vector<double> out(in.size(), 0.0);
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

  if ((res.first == 4) and (res.second < 1.E-10))
    return 0;
  else
    return -1;
}

