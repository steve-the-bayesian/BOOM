/*
  Copyright (C) 2007 Steven L. Scott

  This library is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License as published by the Free Software Foundation; either
  version 2.1 of the License, or (at your option) any later version.

  This library is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA
*/
#ifndef BOOM_GLM_COEFS_HPP
#define BOOM_GLM_COEFS_HPP

#include <Models/ParamTypes.hpp>
#include <LinAlg/Selector.hpp>

namespace BOOM{
  class GlmCoefs
    : public VectorParams
  {
  public:
    explicit GlmCoefs(uint p, bool all=true);  // beta is 0..p

    // If infer_model_selection is true then zero-valued coefficients
    // will be marked as excluded from the model.
    explicit GlmCoefs(const Vector &b, bool infer_model_selection=false);
    GlmCoefs(const Vector &b, const Selector &Inc);
    GlmCoefs(const GlmCoefs &rhs);
    GlmCoefs * clone() const override;

    //---     model selection  -----------
    const Selector &inc()const;
    bool inc(uint p)const;
    void set_inc(const Selector &);
    void add(uint i);
    void drop(uint i);
    void flip(uint i);
    void drop_all();
    void add_all();

    //---- size querries...
    uint size(bool minimal=true)const override; // number included/possible covariates
    uint nvars()const;
    uint nvars_possible()const;
    uint nvars_excluded()const;

    //--- the main job of glm's...
    double predict(const Vector &x)const;
    double predict(const VectorView &x)const;
    double predict(const ConstVectorView &x)const;

    Vector predict(const Matrix &design_matrix)const;
    void predict(const Matrix &design_matrix, Vector &result)const;
    void predict(const Matrix &design_matrix, VectorView result)const;

    //------ operations for only included variables --------
    Vector included_coefficients()const;
    void set_included_coefficients(const Vector &b);
    // Args:
    //   b:  The nonzero elements of beta.
    //   inc: Indicates the positions of the nonzero elements.  Must
    //     satisfy inc.nvars() == beta.size().
    void set_included_coefficients(const Vector &b,
                                   const Selector &inc);

    //----- operations for both included and excluded variables ----
    const Vector & Beta()const;    // reports 0 for excluded positions
    // Consider whether to call infer_sparsity after calling set_Beta.
    void set_Beta(const Vector &);
    double &  Beta(uint I);        // I indexes possible covariates
    double Beta(uint I)const;      // I indexes possible covariates

    // Drop all coefficients with value 0.  Add all others.
    void infer_sparsity();

    Vector vectorize(bool minimal=true)const override;
    Vector::const_iterator unvectorize(
        Vector::const_iterator &v, bool minimal=true) override;
    Vector::const_iterator unvectorize(const Vector &v, bool minimal=true) override;

  private:
    Selector inc_;
    mutable Vector included_coefficients_;
    mutable bool included_coefficients_current_;

    void inc_from_beta(const Vector &v);
    uint indx(uint i)const{return inc_.indx(i);}
    void wrong_size_beta(const Vector &b)const;
    void fill_beta()const;
    void setup_obs();

    double & operator[](uint i);
    double operator[](uint i)const;
  };

}  // namespace BOOM
#endif  // BOOM_GLM_COEFS_HPP
