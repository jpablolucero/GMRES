#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <GMRES.hpp>

// Laplacian1D test can be reproduced in MATLAB to obtain the reference result as follows
/*
N = 128;
h = 1/(N-1);
M = @(v) [ ...
    v(1); ...
    (-v(1:end-2) + 2*v(2:end-1) - v(3:end)) / h; ...
    v(end) ...
];
rhs = ones(N,1)/(N-1);
rhs([1 end]) = 0;
tol     = 1e-8;
maxit   = 3;
restart = 30;
x0      = zeros(N,1);
[sol, flag, relres, iter, resvec] = gmres(M, rhs, restart, tol, maxit, ...
                                                               [], [], x0);
x = linspace(0, 1, N);
refSol = 1/8 - 0.5*(x - 0.5).^2;   
relError = norm(sol - refSol') / norm(refSol');
fprintf(['Iterations %d. ' ...
    'Relative Residual = %.8f. ' ...
    'Relative Error = %.8f\n.'],...
    length(resvec), resvec(end)/norm(rhs),relError);
*/
template <typename T>
bool Laplacian1D()
{
  std::cout << std::endl ;
  std::cout << "Laplacian1D:Laplacian1D ------------- BEGIN ---------------" << std::endl ;
  std::cout << std::endl ;

  const std::size_t N = 128;
  const T h = 1./static_cast<T>(N-1);

  auto M = [N,h]<ValidVectorType V>(const V & vin)
    {
      typedef typename V::value_type real;
      V vout(N);
      vout[0] = vin[0];
      for (std::size_t i = 1; i + 1 < N; ++i) 
	vout[i] = (-vin[i - 1] + 2. * vin[i] - vin[i + 1]) / h;
      vout[N-1] = vin[N-1];
      return vout;
    };

  std::vector<T> mesh(N);
  std::iota(mesh.begin(), mesh.end(), 0);
  std::transform(mesh.begin(), mesh.end(), mesh.begin(),
		 [h](std::size_t i) { return i * h; });
  
  std::vector<T> refSol(N,0.),sol(N,0.);
  std::vector<T> rhs(N,1./static_cast<T>(N-1));rhs[0]=0.;rhs[N-1]=0.;
  
  std::transform(mesh.begin(), mesh.end(), refSol.begin(),
		 [h](const T& x) {
                   return 1. / 8. - 0.5 * (x - 0.5) * (x - 0.5);
		 });

  
  auto norm_2 = [](const auto& v)
  {
    return std::sqrt(std::inner_product(v.begin(),v.end(),v.begin(),0.0));
  };
  
  auto b = M(refSol);

  GMRES solver(M);

  auto param = solver.getParameters();
  param.max_iter = 90;
  param.restart_iter = 30;
  param.tol = 1.E-6;
  solver.setParameters(param);
  
  auto res = solver(b,sol);

  std::vector<T> error(sol.size(),0.0);
  for (std::size_t i = 0 ; i < sol.size() ; i++)
    error[i] = refSol[i] - sol[i];

  auto relativeError = norm_2(error)/norm_2(refSol);
  
  std::cout << "Iterations: " << res.first << "\t" << "Residual Reduction: " << res.second << ". Discrete Error: " << norm_2(error) << std::endl ;
  
  if (std::abs(0.40082103-res.second) > 1.5E-5)
    {
      std::cout << "Last residual does not coincide with the reference calculated from MATLAB: "
		<< std::fixed << std::setprecision(8) << res.second << " vs. " << 0.40082103 << std::endl ;
      std::cout << "Laplacian1D:Laplacian1D ------------- FAIL ---------------" << std::endl ;
      return false;
    }
  if (std::abs(0.40324826-relativeError) > 1.5E-5)
    {
      std::cout << "Last relative error does not coincide with the reference calculated from MATLAB: "
		<< std::fixed << std::setprecision(8) << relativeError << " vs. " << 0.40324826 << std::endl ;
      std::cout << "Laplacian1D:Laplacian1D ------------- FAIL ---------------" << std::endl ;
      return false;
    }

  std::cout << std::endl ;
  std::cout << "Laplacian1D:Laplacian1D ------------- SUCCESS ---------------" << std::endl ;
  std::cout << std::endl ;

  return true;
  
}

int main(int argc, char **argv)
{
  if (Laplacian1D<float>()&&Laplacian1D<double>()&&Laplacian1D<long double>())
    {
      std::cout << std::endl ;
      std::cout << "Laplacian1D ------------- SUCCESS ---------------" << std::endl ;
      std::cout << std::endl ;
      return 0;
    }
  else
    {
      std::cout << std::endl ;
      std::cout << "Laplacian1D ------------- FAIL ---------------" << std::endl ;
      std::cout << std::endl ;
      return -1;
    }
  
}
