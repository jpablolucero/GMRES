#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <GMRES.hpp>

// upperTriangular test Reference right hand side can be obtained in MATLAB with the script
/*
digits(60)
U = vpa([
 1    0.5   -1.2   3.3   2.2   -0.7   4.4   5.5;
 0    2.1   -0.7   6.6  -1.1    2.4  -3.3   4.8;
 0    0     3.14   1.59  2.65 -3.58   9.79 -1.32;
 0    0     0      4.25 -6.12  7.13  -8.14  9.15;
 0    0     0      0     5.5   -2.5    3.3  -4.4;
 0    0     0      0     0     6.6   -7.7   8.8;
 0    0     0      0     0     0      7.77 -1.11;
 0    0     0      0     0     0      0     8.88
], 34);
b     = U * ones(8,1);
x_ref = ones(8,1);
for k = 1:length(b)                                                                                                                                                                                                                                                                                                                 
     fprintf('%s\n', char(vpa(b(k), 21)));                                                                                                                                                                                                                                                                                          
end
*/
template <typename T>
bool upperTriangular()
{
  std::cout << std::endl ;
  std::cout << "UpperTriangular:upperTriangular ------------- BEGIN ---------------" << std::endl ;
  std::cout << std::endl ;

  std::vector<std::vector<T>> U = {
    {T(1L),    T(0L),     T(0L),     T(0L),     T(0L),     T(0L),      T(0L),      T(0L)},
    {T(0.5L),  T(2.1L),   T(0L),     T(0L),     T(0L),     T(0L),      T(0L),      T(0L)},
    {T(-1.2L), T(-0.7L),  T(3.14L),  T(0L),     T(0L),     T(0L),      T(0L),      T(0L)},
    {T(3.3L),  T(6.6L),   T(1.59L),  T(4.25L),  T(0L),     T(0L),      T(0L),      T(0L)},
    {T(2.2L),  T(-1.1L),  T(2.65L),  T(-6.12L), T(5.5L),   T(0L),      T(0L),      T(0L)},
    {T(-0.7L), T(2.4L),   T(-3.58L), T(7.13L),  T(-2.5L),  T(6.6L),    T(0L),      T(0L)},
    {T(4.4L),  T(-3.3L),  T(9.79L),  T(-8.14L), T(3.3L),   T(-7.7L),   T(7.77L),   T(0L)},
    {T(5.5L),  T(4.8L),   T(-1.32L), T(9.15L),  T(-4.4L),  T(8.8L),    T(-1.11L),  T(8.88L)}
  };
  
  std::vector<T> b     = {
    T(15L),
    T(10.8L),
    T(12.27L),
    T(6.27L),
    T(1.9L),
    T(7.7L),
    T(6.66L),
    T(8.88L)
  };
  
  std::vector<T> x_ref = {
    T(1L),T(1L),T(1L),T(1L),T(1L),T(1L),T(1L),T(1L)
  };

  std::vector<T> x = solveUpperTriangular(U, b);

  T res = std::sqrt(std::inner_product(x.begin(), x.end(), x_ref.begin(), T(0L),
				       std::plus<>(),
				       [](T x, T y) { T d = x - y; return d * d; }));

  std::cout << "Difference to reference = " << std::scientific << res << std::endl ;
  
  if (res >= 100.*std::numeric_limits<T>::epsilon())
    {
      std::cout << std::endl ;
      std::cout << "UpperTriangular:upperTriangular ------------- FAIL ---------------" << std::endl ;
      std::cout << std::endl ;
      return false;
    }
  std::cout << std::endl ;
  std::cout << "UpperTriangular:upperTriangular ------------- SUCCESS ---------------" << std::endl ;
  std::cout << std::endl ;
  return true;
  
}

int main(int argc, char **argv)
{
  if (upperTriangular<float>()&&upperTriangular<double>()&&upperTriangular<long double>())
    {
      std::cout << std::endl ;
      std::cout << "UpperTriangular ------------- SUCCESS ---------------" << std::endl ;
      std::cout << std::endl ;
      return 0;
    }
  else
    {
      std::cout << std::endl ;
      std::cout << "UpperTriangular ------------- FAIL ---------------" << std::endl ;
      std::cout << std::endl ;
      return -1;
    }
  
}
