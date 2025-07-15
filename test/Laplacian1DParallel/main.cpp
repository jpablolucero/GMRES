#include <mpi.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <numeric>
#include "GMRES.hpp"

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const std::size_t N = 64;
  const double h = 1. / static_cast<double>(N - 1);

  if (N % size != 0) {
    if (rank == 0) std::cerr << "N must be divisible by number of processes.\n";
    MPI_Finalize();
    return -1;
  }

  const std::size_t local_N = N / size;
  const std::size_t start = rank * local_N;

  auto op = [=](const std::vector<double>& vin_local) {
    std::vector<double> vout(local_N);
    double left_ghost = 0.0, right_ghost = 0.0;

    if (rank > 0)
      MPI_Sendrecv(&vin_local[0], 1, MPI_DOUBLE, rank - 1, 0,
                   &left_ghost, 1, MPI_DOUBLE, rank - 1, 1,
                   MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    if (rank < size - 1)
      MPI_Sendrecv(&vin_local[local_N - 1], 1, MPI_DOUBLE, rank + 1, 1,
                   &right_ghost, 1, MPI_DOUBLE, rank + 1, 0,
                   MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    for (std::size_t i = 0; i < local_N; ++i) {
      std::size_t gi = start + i;
      if (gi == 0 || gi == N - 1)
	{
	  vout[i] = vin_local[i];
	}
      else
	{
	  double left  = (i == 0) ? left_ghost : vin_local[i - 1];
	  double right = (i == local_N - 1) ? right_ghost : vin_local[i + 1];
	  vout[i] = (-left + 2. * vin_local[i] - right) / h;
	}
    }

    return vout;
  };

  // Global mesh, local chunk
  std::vector<double> mesh(local_N);
  std::iota(mesh.begin(), mesh.end(), start);
  std::transform(mesh.begin(), mesh.end(), mesh.begin(), [h](std::size_t i) { return i * h; });

  std::vector<double> refSol(local_N), sol(local_N, 0.0), rhs(local_N, 1. / (N - 1));
  if (start == 0) rhs[0] = 0.;
  if (start + local_N == N) rhs[local_N - 1] = 0.;

  std::transform(mesh.begin(), mesh.end(), refSol.begin(),
                 [h](double x) { return 1. / 8. - 0.5 * (x - 0.5) * (x - 0.5); });
  
  auto b = op(refSol);

  auto inner_product = [](const std::vector<double>& v1, const std::vector<double>& v2) {
    double local = std::inner_product(v1.begin(), v1.end(), v2.begin(), 0.0);
    double global = 0.0;
    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return global;
  };

  GMRES solver(op,inner_product);
  auto parameters = solver.get_parameters();
  parameters.max_iter = N;
  parameters.restart_iter = 30;
  parameters.tol = 1.E-6;
  solver.set_parameters(parameters);

  auto res = solver(b, sol);

  auto norm_2 = [&](const std::vector<double>& v) {
    return std::sqrt(inner_product(v,v));
  };

  std::vector<double> error(sol.size());
  for (std::size_t i = 0; i < sol.size(); ++i)
    error[i] = refSol[i] - sol[i];

  auto l2error = norm_2(error);
  
  if (rank == 0)
    {
      std::cout << "Iterations: " << res.first
		<< "\tResidual Reduction: " << res.second
		<< ". Discrete Error: " << l2error << std::endl;
      if ((res.first == 64) && (res.second < 5.E-2))
        MPI_Finalize(), exit(0);
      else
        MPI_Finalize(), exit(-1);
    }

  MPI_Finalize();
  return 0;
}
