#include <eigen3/Eigen/Core>
#include "LinAlg/Matrix.hpp"
#include "LinAlg/SpdMatrix.hpp"
#include "LinAlg/Cholesky.hpp"
#include "LinAlg/EigenMap.hpp"

using namespace BOOM;
using std::cout;
using std::endl;

int main() {
  Matrix A(3, 3);
  A.randomize();
  Matrix B(3, 3);
  B.randomize();

  Matrix C(3, 3);
  EigenMap(C) = EigenMap(A) + EigenMap(B);
  cout << "C from eigen = " << endl << C
            << "Direct calculation = " << endl
            << A + B << endl;

  SpdMatrix V(3);
  V.randomize();

  EigenMap(C) = EigenMap(V).selfadjointView<Eigen::Upper>() * EigenMap(B);
  cout << "C from eigen = " << endl << C
       << "Direct calculation = " << endl
       << V * B << endl;

  B.randomize();
  EigenMap(C) = EigenMap(A) *
      EigenMap(V).selfadjointView<Eigen::Upper>() *
      EigenMap(A).transpose();
  cout << "C from eigen = " << endl << C
       << "Direct calculation = " << endl
       << sandwich(A, V) << endl;
}
  
