#include <LinAlg/SVD.hpp>
#include <stdexcept>
#include <sstream>
#include <cpputil/report_error.hpp>

extern "C"{
 void dgesvd_(const char *,  // JOBU
              const char *,  // JOBVT
              int *,         // nrow(m)
              int *,         // ncol(n)
              double *,      // A
              int *,         // LDA
              double *,      // S
              double *,      // U
              int *,         // LDU
              double *,      // VT
              int *,         // LDVT
              double *,      // WORK
              int *,         // LWORK
              int *);         // INFO
}

namespace BOOM{
SVD::SVD(const Matrix &m)
    : left_(m.nrow(), m.nrow()),
      right_(m.ncol(), m.ncol()),
      values_(std::min<uint>(m.nrow(), m.ncol())),
      work_(1)
{

  Matrix A(m);
  int mm = m.nrow();
  int n = m.ncol();
  int lwork = -1;
  int info;
  // query optimal workspace
  dgesvd_("A",             // JOBU
          "A",             // JOBVT
          &mm,             // nrow(m)
          &n,              // ncol(n)
          A.data(),
          &mm,             // LDA
          values_.data(),  // S
          left_.data(),    // U
          &mm,             // LDU
          right_.data(),   // VT
          &n,              // LDVT
          work_.data(),    // WORK
          &lwork,          // LWORK
          &info            // INFO
          );

  lwork = lround(work_[0]);
  work_.resize(lwork);

  dgesvd_("A",            // JOBU
          "A",            // JOBVT
          &mm,            // nrow(m)
          &n,             // ncol(n)
          A.data(),       // A
          &mm,            // LDA
          values_.data(), // S
          left_.data(),   // U
          &mm,            // LDU
          right_.data(),  // VT
          &n,             // LDVT
          work_.data(),   // WORK
          &lwork,         // LWORK
          &info           // INFO
          );
}

const Vector & SVD::values()const{return values_;}
const Matrix & SVD::left()const{return left_;}
const Matrix & SVD::right()const{return right_;}
Matrix SVD::original_matrix()const{
  Matrix Sigma(left_.ncol(), right_.nrow(), 0.0);
  Sigma.set_diag(values_);
  Matrix ans = left_ * Sigma * right_;
  return ans;
}

Matrix SVD::solve(const Matrix &rhs, double tol)const{
  Matrix ans = left_.Tmult(rhs);
  for(uint i = 0; i<ans.nrow(); ++i){
    double scale = values_[i]/values_[0];
    ans.row(i) *= scale < tol ? 0 : 1.0/values_[i];
  }
  ans = right_.Tmult(ans);
  return ans;
}


Vector SVD::solve(const Vector &rhs, double tol)const{
  Vector ans = left_.Tmult(rhs);
   for(uint i = 0; i<ans.size(); ++i){
     double scale = values_[i]/values_[0];
     ans(i) *= scale < tol ? 0 : 1.0/values_[i];
   }
   ans = right_.Tmult(ans);
  return ans;
}

Matrix SVD::inv()const{
  bool invertible = left_.is_square()
      && right_.is_square()
      && left_.nrow() == right_.nrow();
  if(!invertible){
    std::ostringstream err;
    err << "error in SVD::inv(), only square matrices can be inverted"
        << std::endl
        << "original matrix = " << std::endl << original_matrix()
        << std::endl;
    report_error(err.str());
  }
  return solve(left_.Id());
}

}
