#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <numeric>
#include "gmres.hpp"

int main()
{
  std::vector<std::vector<std::pair<std::size_t, double>>> A(6);

  A[0] = {{0,  4.0}, {1, -1.0}, {3, -1.0}};
  A[1] = {{0, -1.0}, {1,  4.0}, {2, -1.0}, {4, -1.0}};
  A[2] = {{1, -1.0}, {2,  4.0}, {5, -1.0}};
  A[3] = {{0, -1.0}, {3,  4.0}, {4, -1.0}};
  A[4] = {{1, -1.0}, {3, -1.0}, {4,  4.0}, {5, -1.0}};
  A[5] = {{2, -1.0}, {4, -1.0}, {5,  4.0}};

  std::vector<double> refSol(6, 0.0);
  refSol[0] = 1.0;
  refSol[1] = 2.0;
  refSol[2] = 3.0;
  refSol[3] = 4.0;
  refSol[4] = 5.0;
  refSol[5] = 6.0;

  std::vector<double> rhs(6, 0.0);
  for (std::size_t i = 0; i < A.size(); ++i)
    for (const auto& [j, val] : A[i])
      rhs[i] += val * refSol[j];

  std::vector<double> sol(6, 0.0);

  auto op = [&](const std::vector<double>& in)
  {
    std::vector<double> out(in.size(), 0.0);
    for (std::size_t i = 0; i < A.size(); ++i)
      for (const auto& [j, val] : A[i])
        out[i] += val * in[j];
    return out;
  };

  gmres solver(op);

  auto param = solver.get_param();
  param.max_iter = 1E9;
  param.restart_iter = 30;
  param.tol = 1.E-8;
  solver.set_param(param);

  auto res = solver(rhs, sol);

  std::cout << "Iterations: " << res.first << "\t" << "Residual Reduction: " << res.second << std::endl;

  return 0;
}
