#include <mpi.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <GMRES.hpp>

template <typename T>
bool Laplacian1DParallel(int argc, char** argv)
{

  MPI_Datatype MPI_T =
    std::is_same_v<T,float>       ? MPI_FLOAT
    : std::is_same_v<T,long double>      ? MPI_LONG_DOUBLE
    : /*else*/                         MPI_DOUBLE;
 
  std::cout << std::endl ;
  std::cout << "Laplacian1DParallel:Laplacian1DParallel ------------- BEGIN ---------------" << std::endl ;
  std::cout << std::endl ;

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const std::size_t N = 64;
  const T h = 1. / static_cast<T>(N - 1);

  if (N % size != 0) {
    if (rank == 0) std::cerr << "N must be divisible by number of processes.\n";
    MPI_Finalize();
    return -1;
  }

  const std::size_t local_N = N / size;
  const std::size_t start = rank * local_N;

  auto op = [=](const std::vector<T>& vin_local) {
    std::vector<T> vout(local_N);
    T left_ghost = 0.0, right_ghost = 0.0;

    if (rank > 0)
      MPI_Sendrecv(&vin_local[0], 1, MPI_T, rank - 1, 0,
                   &left_ghost, 1, MPI_T, rank - 1, 1,
                   MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    if (rank < size - 1)
      MPI_Sendrecv(&vin_local[local_N - 1], 1, MPI_T, rank + 1, 1,
                   &right_ghost, 1, MPI_T, rank + 1, 0,
                   MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    for (std::size_t i = 0; i < local_N; ++i) {
      std::size_t gi = start + i;
      if (gi == 0 || gi == N - 1)
	{
	  vout[i] = vin_local[i];
	}
      else
	{
	  T left  = (i == 0) ? left_ghost : vin_local[i - 1];
	  T right = (i == local_N - 1) ? right_ghost : vin_local[i + 1];
	  vout[i] = (-left + 2. * vin_local[i] - right) / h;
	}
    }

    return vout;
  };

  // Global mesh, local chunk
  std::vector<T> mesh(local_N);
  std::iota(mesh.begin(), mesh.end(), start);
  std::transform(mesh.begin(), mesh.end(), mesh.begin(), [h](std::size_t i) { return i * h; });

  std::vector<T> refSol(local_N), sol(local_N, 0.0), rhs(local_N, 1. / (N - 1));
  if (start == 0) rhs[0] = 0.;
  if (start + local_N == N) rhs[local_N - 1] = 0.;

  std::transform(mesh.begin(), mesh.end(), refSol.begin(),
                 [h](T x) { return 1. / 8. - 0.5 * (x - 0.5) * (x - 0.5); });
  
  auto b = op(refSol);

  auto inner_product = [&](const std::vector<T>& v1, const std::vector<T>& v2) {
    T local = std::inner_product(v1.begin(), v1.end(), v2.begin(), 0.0);
    T global = 0.0;
    MPI_Allreduce(&local, &global, 1, MPI_T, MPI_SUM, MPI_COMM_WORLD);
    return global;
  };

  GMRES solver(op,inner_product);
  auto parameters = solver.getParameters();
  parameters.max_iter = N;
  parameters.restart_iter = 30;
  parameters.tol = 1.E-6;
  solver.setParameters(parameters);

  auto res = solver(b, sol);

  auto norm_2 = [&](const std::vector<T>& v) {
    return std::sqrt(inner_product(v,v));
  };

  std::vector<T> error(sol.size());
  for (std::size_t i = 0; i < sol.size(); ++i)
    error[i] = refSol[i] - sol[i];

  auto l2error = norm_2(error);
  
  std::cout << "Iterations: " << res.first
	    << "\tResidual Reduction: " << std::scientific << res.second
	    << ". Discrete Error: " << l2error << std::endl;
  if ((res.first != 64) || (res.second > 5.E-2))
    {
      std::cout << std::endl ;
      std::cout << "Laplacian1DParallel:Laplacian1DParallel ------------- FAIL ---------------" << std::endl ;
      std::cout << std::endl ;
      return false;
    }

  std::cout << std::endl ;
  std::cout << "Laplacian1DParallel:Laplacian1DParallel ------------- SUCCESS ---------------" << std::endl ;
  std::cout << std::endl ;

  return true;

}

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  if (Laplacian1DParallel<float>(argc,argv)&&
      Laplacian1DParallel<double>(argc,argv)&&
      Laplacian1DParallel<long double>(argc,argv))
    {
      std::cout << std::endl ;
      std::cout << "Laplacian1DParallel ------------- SUCCESS ---------------" << std::endl ;
      std::cout << std::endl ;
      MPI_Finalize();
      return 0;
    }
  else
    {
      std::cout << std::endl ;
      std::cout << "Laplacian1DParallel ------------- FAIL ---------------" << std::endl ;
      std::cout << std::endl ;
      MPI_Finalize();
      return -1;
    }
  
}
