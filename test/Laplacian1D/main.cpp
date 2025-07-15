#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <numeric>
#include "GMRES.hpp"

int main()
{
  const std::size_t N = 64;
  const double h = 1./static_cast<double>(N-1);

  auto op = [N, h](const std::vector<double>& vin) {
    typedef typename std::vector<double>::value_type real;
    std::vector<double> vout(N);
    vout[0] = vin[0];
    for (std::size_t i = 1; i + 1 < N; ++i) {
      vout[i] = (-vin[i - 1] + 2. * vin[i] - vin[i + 1]) / h;
    }
    vout[N - 1] = vin[N - 1];
    return vout;
  };
 
  std::vector<double> mesh(N);
  std::iota(mesh.begin(), mesh.end(), 0);
  std::transform(mesh.begin(), mesh.end(), mesh.begin(),
		 [h](std::size_t i) { return i * h; });
 
  std::vector<double> refSol(N,0.),sol(N,0.);
  std::vector<double> rhs(N,1./static_cast<double>(N-1));rhs[0]=0.;rhs[N-1]=0.;

  std::transform(mesh.begin(), mesh.end(), refSol.begin(),
		 [h](const double& x) {
                   return 1. / 8. - 0.5 * (x - 0.5) * (x - 0.5);
		 });
 
  auto norm_2 = [](const auto& v)
  {
    return std::sqrt(std::inner_product(v.begin(),v.end(),v.begin(),0.0));
  };

  auto b = op(refSol);

  GMRES solver(op);

  auto parameters = solver.getParameters();
  parameters.max_iter = N;
  parameters.restart_iter = 30;
  parameters.tol = 1.E-6;
  solver.setParameters(parameters);
  
  auto res = solver(b,sol);

    std::vector<double> error(sol.size(),0.0);
  for (std::size_t i = 0 ; i < sol.size() ; i++)
    error[i] = refSol[i] - sol[i];

  std::cout << "Iterations: " << res.first << "\t" << "Residual Reduction: " << res.second << ". Discrete Error: " << norm_2(error) << std::endl ;

  if ((res.first == 64) and (res.second < 5.E-2))
    return 0;
  else
    return -1;
}

