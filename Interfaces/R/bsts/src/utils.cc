#include "utils.h"
#include "R_ext/Arith.h"  // for R_IsNA

namespace BOOM {
namespace bsts {

//======================================================================
// Initialize the model to be empty, except for variables that are
// known to be present with probability 1.
void DropUnforcedCoefficients(const Ptr<GlmModel> &glm,
                              const BOOM::Vector &prior_inclusion_probs) {
  glm->coef().drop_all();
  for (int i = 0; i < prior_inclusion_probs.size(); ++i) {
    if (prior_inclusion_probs[i] >= 1.0) {
      glm->coef().add(i);
    }
  }
}

Matrix ExtractPredictors(SEXP r_object,
                         const std::string &name,
                         int default_length) {
  SEXP r_predictors = getListElement(r_object, name);
  if (Rf_isNull(r_predictors)) {
    return Matrix(default_length, 1, 1.0);
  } else {
    Matrix ans = ToBoomMatrix(r_predictors);
    if (ans.nrow() != default_length) {
      report_error("Matrix of predictors had an unexpected number of rows.");
    }
    return ans;
  }
}

std::vector<bool> IsObserved(SEXP r_vector) {
  if (!Rf_isNumeric(r_vector)) {
    report_error("Input vector is non-numeric.");
  }
  size_t n = Rf_length(r_vector);
  std::vector<bool> ans(n);
  double *data = REAL(Rf_coerceVector(r_vector, REALSXP));
  for (int i = 0; i < n; ++i) {
    ans[i] = !R_IsNA(data[i]);
  }
  return ans;
}

}  // namespace bsts
}  // namespace BOOM
