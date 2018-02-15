/*
  Copyright (C) 2005-2009 Steven L. Scott

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
#ifndef BOOM_PRODUCT_DIRICHLET_MODEL_HPP
#define BOOM_PRODUCT_DIRICHLET_MODEL_HPP
#include "Models/ModelTypes.hpp"
#include "Models/ParamTypes.hpp"
#include "Models/Sufstat.hpp"
#include "Models/Policies/SufstatDataPolicy.hpp"
#include "Models/Policies/PriorPolicy.hpp"
#include "Models/Policies/ParamPolicy_1.hpp"

namespace BOOM{

  class ProductDirichletSuf
    : public SufstatDetails<MatrixData>
  {
  public:
    ProductDirichletSuf(uint p);
    ProductDirichletSuf(const ProductDirichletSuf &rhs);
    ProductDirichletSuf * clone()const override;
    const Matrix & sumlog()const;
    double n()const;
    void Update(const MatrixData &) override;
    void clear() override;
    void combine(const Ptr<ProductDirichletSuf> &);
    void combine(const ProductDirichletSuf &);
    ProductDirichletSuf * abstract_combine(Sufstat *s) override;

    Vector vectorize(bool minimal=true)const override;
    Vector::const_iterator unvectorize(Vector::const_iterator &v,
                                            bool minimal=true) override;
    Vector::const_iterator unvectorize(const Vector &v,
                                            bool minimal=true) override;
    ostream &print(ostream &out)const override;
  private:
    Matrix sumlog_;
    double n_;
  };


// class ProductDirichletModelBase
//     : public SufstatDataPolicy<MatrixData, ProductDirichletSuf>



  class ProductDirichletModel
    : public ParamPolicy_1<MatrixParams>,
      public SufstatDataPolicy<MatrixData, ProductDirichletSuf>,
      public PriorPolicy,
      public dLoglikeModel
  {
  public:
    ProductDirichletModel(uint p);  // default is uniform:  Nu = 1
    ProductDirichletModel(const Matrix &NU);
    ProductDirichletModel(const Vector &wgt, const Matrix &Pi);
    ProductDirichletModel(const ProductDirichletModel &);

    ProductDirichletModel * clone()const override;
    uint dim()const;
    Ptr<MatrixParams> Nu_prm();
    const Ptr<MatrixParams> Nu_prm()const;
    const Matrix & Nu()const;

    void set_Nu(const Matrix &Nu);

    double pdf(const Ptr<Data> &dp, bool logscale)const;
    double pdf(const Matrix &Pi, bool logscale)const;
    //    double Logp(const Vector &, Vector &, Matrix &, uint nd)const;

    // The argument is a vector created by stacking the columns of the
    // parameter Nu.
    double loglike(const Vector &Nu_columns)const override;
    double dloglike(const Vector &Nu_columns, Vector &g)const override;

    Matrix sim(RNG &rng = GlobalRng::rng)const;
  };
}

#endif // BOOM_PRODUCT_DIRICHLET_MODEL_HPP
