#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <GMRES.hpp>

// Givens test Reference result can be obtained in MATLAB with the script
/*
digits(60)
h_col = vpa((1:13)',60);
cs = v pa([3/5; 5/13; 8/17; 7/25; 9/41;11/61;12/37;20/29;28/53;33/65;16/65;48/73],60);
sn = -vpa([4/5;12/13;15/17;24/25;40/41;60/61;35/37;21/29;45/53;56/65;63/65;55/73],60);
j = length(cs);
for i = 1:(j-1)
    h_col(i:i+1) = [cs(i), -sn(i);
                    sn(i),  cs(i)] * h_col(i:i+1);
end
if j < numel(h_col)
    cs(j) =  h_col(j)   / sqrt(h_col(j)^2 + h_col(j+1)^2);
    sn(j) = -h_col(j+1) / sqrt(h_col(j)^2 + h_col(j+1)^2);
    h_col(j)   = sqrt(h_col(j)^2 + h_col(j+1)^2);
    h_col(j+1) = 0;
end
for k = 1:length(h_col)
     fprintf('%s\n', char(vpa(h_col(k), 32)));
end
*/
template <typename T>
bool Givens()
{
  std::cout << std::endl ;
  std::cout << "Givens:Givens ------------- BEGIN ---------------" << std::endl ;
  std::cout << std::endl ;

  std::vector<T> cs = {
    T(3L)/T(5L),    // (3,4,5)
    T(5L)/T(13L),   // (5,12,13)
    T(8L)/T(17L),   // (8,15,17)
    T(7L)/T(25L),   // (7,24,25)
    T(9L)/T(41L),   // (9,40,41)
    T(11L)/T(61L),  // (11,60,61)
    T(12L)/T(37L),  // (12,35,37)
    T(20L)/T(29L),  // (20,21,29)
    T(28L)/T(53L),  // (28,45,53)
    T(33L)/T(65L),  // (33,56,65)
    T(16L)/T(65L),  // (16,63,65)
    T(48L)/T(73L)   // (48,55,73)
  };
  std::vector<T> sn = {
    -T(4L)/T(5L),
    -T(12L)/T(13L),
    -T(15L)/T(17L),
    -T(24L)/T(25L),
    -T(40L)/T(41L),
    -T(60L)/T(61L),
    -T(35L)/T(37L),
    -T(21L)/T(29L),
    -T(45L)/T(53L),
    -T(56L)/T(65L),
    -T(63L)/T(65L),
    -T(55L)/T(73L)
  };
  
  std::vector<T> h_col = {T(1L),T(2L),T(3L),T(4L),T(5L),T(6L),T(7L),T(8L),T(9L),T(10L),T(11L),T(12L),T(13L)};

  applyGivensRotation(h_col,cs,sn,11);

  std::vector<T> h_col_ref = {
    T(2.2L),
    T(2.9230769230769230769230769230769L),
    T(3.898642533936651583710407239819L),
    T(5.133212669683257918552036199095L),
    T(5.9101953426774086745392340801236L),
    T(7.0774390696210203701324899904292L),
    T(7.636962825240455364876184783516L),
    T(8.1670287998036817373180191136173L),
    T(10.854516099731542583450004491692L),
    T(10.230243731812450144822386444966L),
    T(12.690769984287200312465800347464L),
    T(13.057111956046140511463152520221L),
    T(0.0L)} ;
    
    T res = std::sqrt(std::inner_product(h_col.begin(), h_col.end(), h_col_ref.begin(), T(0L),
				       std::plus<>(),
				       [](T x, T y) { T d = x - y; return d * d; }));

  std::cout << "Difference to reference = " << std::scientific << res << std::endl ;
  
  if (res >= 100.*std::numeric_limits<T>::epsilon())
    {
      std::cout << std::endl ;
      std::cout << "Givens:Givens ------------- FAIL ---------------" << std::endl ;
      std::cout << std::endl ;
      return false;
    }
  std::cout << std::endl ;
  std::cout << "Givens:Givens ------------- SUCCESS ---------------" << std::endl ;
  std::cout << std::endl ;
  return true;
  
}

int main(int argc, char **argv)
{
  if (Givens<float>()&&Givens<double>()&&Givens<long double>())
    {
      std::cout << std::endl ;
      std::cout << "Givens ------------- SUCCESS ---------------" << std::endl ;
      std::cout << std::endl ;
      return 0;
    }
  else
    {
      std::cout << std::endl ;
      std::cout << "Givens ------------- FAIL ---------------" << std::endl ;
      std::cout << std::endl ;
      return -1;
    }
  
}
