#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <numeric>
#include "GMRES.hpp"

int main()
{
  std::vector<double> rhs(3, 0.0);
  rhs[0] = -6.0;
  rhs[1] =  2.0;
  rhs[2] = 12.0;

  std::vector<std::vector<double>> A(3, std::vector<double>(3, 0.0));
  A[0][0] =  1.0; A[0][1] = 1.0; A[0][2] = 1.0;
  A[1][0] =  4.0; A[1][1] = 2.0; A[1][2] = 1.0;
  A[2][0] = 16.0; A[2][1] = 4.0; A[2][2] = 1.0;

  std::vector<double> sol(3, 0.0);

  std::vector<double> refSol(3, 0.0);
  refSol[0] = -1.0;
  refSol[1] = 11.0;
  refSol[2] = -16.0;

  auto op = [&](const std::vector<double>& in)
  {
    std::vector<double> out(in.size(), 0.0);
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

  std::cout << "Iterations: " << res.first << "\t" << "Residual Reduction: " << res.second << std::endl;

  if ((res.first == 3) and (res.second < 1.E-14))
    return 0;
  else
    return -1;
}

