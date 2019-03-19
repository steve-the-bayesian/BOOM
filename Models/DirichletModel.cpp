// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005 Steven L. Scott

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

#include "Models/DirichletModel.hpp"
#include "cpputil/math_utils.hpp"

#include <cmath>
#include <sstream>
#include <stdexcept>
#include "Models/PosteriorSamplers/PosteriorSampler.hpp"
#include "Models/SufstatAbstractCombineImpl.hpp"
#include "distributions.hpp"  // for rgamma, lgamma, digamma, etc.

namespace BOOM {

  //======================================================================
  using DS = BOOM::DirichletSuf;

  DS::DirichletSuf(uint S) : sumlog_(S, 0.0), n_(0){};

  DS::DirichletSuf(const DirichletSuf &rhs)
      : Sufstat(rhs),
        SufstatDetails<VectorData>(rhs),
        sumlog_(rhs.sumlog_),
        n_(rhs.n_) {}

  DS *DS::clone() const { return new DS(*this); }

  void DS::clear() {
    sumlog_ = 0.0;
    n_ = 0.0;
  }

  void DS::Update(const VectorData &x) {
    n_ += 1.0;
    sumlog_ += log(x.value());
  }

  void DS::add_mixture_data(const Vector &x, double prob) {
    n_ += prob;
    sumlog_.axpy(log(x), prob);
  }

  const Vector &DS::sumlog() const { return sumlog_; }
  double DS::n() const { return n_; }

  void DS::combine(const Ptr<DS> &s) {
    sumlog_ += s->sumlog_;
    n_ += s->n_;
  }

  void DS::combine(const DS &s) {
    sumlog_ += s.sumlog_;
    n_ += s.n_;
  }

  DirichletSuf *DS::abstract_combine(Sufstat *s) {
    return abstract_combine_impl(this, s);
  }

  Vector DS::vectorize(bool) const {
    Vector ans = sumlog_;
    ans.push_back(n_);
    return ans;
  }

  Vector::const_iterator DS::unvectorize(Vector::const_iterator &v, bool) {
    uint dim = sumlog_.size();

    Vector tmp(v, v + dim);
    v += dim;
    sumlog_ = tmp;
    n_ = *v;
    return ++v;
  }

  Vector::const_iterator DS::unvectorize(const Vector &v, bool minimal) {
    Vector::const_iterator it = v.begin();
    return unvectorize(it, minimal);
  }

  std::ostream &DS::print(std::ostream &out) const { return out << n_ << " " << sumlog_; }
  //======================================================================
  using DM = BOOM::DirichletModel;

  DM::DirichletModel(uint S, double Nu)
      : ParamPolicy(new VectorParams(S, Nu)), DataPolicy(new DS(S)) {}

  DM::DirichletModel(const Vector &Nu)
      : ParamPolicy(new VectorParams(Nu)), DataPolicy(new DS(Nu.size())) {}

  DM::DirichletModel(const DirichletModel &rhs)
      : Model(rhs),
        VectorModel(rhs),
        ParamPolicy(rhs),
        DataPolicy(rhs),
        PriorPolicy(rhs),
        DiffVectorModel(rhs),
        NumOptModel(rhs),
        MixtureComponent(rhs) {}

  DM *DM::clone() const { return new DirichletModel(*this); }

  Ptr<VectorParams> DM::Nu() { return ParamPolicy::prm(); }
  const Ptr<VectorParams> DM::Nu() const { return ParamPolicy::prm(); }

  uint DM::dim() const { return nu().size(); }
  const Vector &DM::nu() const { return Nu()->value(); }
  const double &DM::nu(uint i) const { return nu()[i]; }
  void DM::set_nu(const Vector &newnu) { Nu()->set(newnu); }

  Vector DM::pi() const {
    Vector ans(nu());
    double nc = ans.sum();
    return ans / nc;
  }
  double DM::pi(uint i) const { return nu(i) / nu().sum(); }

  double DM::pdf(const Ptr<Data> &dp, bool logscale) const {
    return pdf(DAT(dp)->value(), logscale);
  }

  double DM::pdf(const Data *dp, bool logscale) const {
    return pdf(DAT(dp)->value(), logscale);
  }

  double DM::pdf(const Vector &pi, bool logscale) const {
    return ddirichlet(pi, nu(), logscale);
  }

  // Args:
  //   probs: A vector of probabilities to be evaluated.  If no
  //     derivatives are desired then probs can either match the
  //     dimension of nu(), in which case it must sum to 1, or its
  //     dimension must be one less, in which case its sum must be
  //     non-negative but can't exceed 1.  Element 0 is assumed to be
  //     the function of the other elements, so that probs0 = 1 -
  //     sum(probs).  If derivatives are desired then probs.size()
  //     must be nu().size() - 1.
  //   gradient: The derivative of the unconstrained elements of
  //     probs.  This is only accessed if nderivs > 0.  It will be
  //     resized if needed, so that its dimension is one less than the
  //     dimension of nu().
  //   Hessian: Second derivative of logp with respect to the free
  //     elements of probs (i.e. not the first one).  This matrix is
  //     only accessed if nderiv > 1, in which case it will be resized.
  //   nderiv:  The number of derivatives desired.
  double DirichletModel::Logp(const Vector &probs, Vector &gradient,
                              Matrix &Hessian, uint nderiv) const {
    // Because sum(p)=1, there are only p.size()-1 free elements in p.
    // The constraint is enforced by expressing the first element of p
    // as a function of the other variables.  The corresponding elements
    // in g and h are zeroed.
    if (probs.size() == nu().size() && nderiv == 0) {
      return ddirichlet(probs, nu(), true);
    } else if (probs.size() + 1 != nu().size()) {
      report_error(
          "probs is the wrong size in DirichletModel::Logp.  "
          "Its dimension should be one less than nu().size()");
    }
    const Vector &n(nu());
    double p0 = 1 - sum(probs);
    Vector full_probs(probs.size() + 1);
    full_probs[0] = p0;
    VectorView(full_probs, 1) = probs;
    double ans = ddirichlet(full_probs, n, true);
    if (nderiv > 0) {
      gradient.resize(probs.size());
      for (int i = 0; i < probs.size(); ++i) {
        gradient[i] = (n[i + 1] - 1) / probs[i] - (n[0] - 1) / p0;
        if (nderiv > 1) {
          Hessian.resize(probs.size(), probs.size());
          for (int j = 0; j < probs.size(); ++j) {
            Hessian(i, j) =
                -(n[0] - 1) / square(p0) -
                (i == j ? (1.0 - n[i + 1]) / square(probs[i]) : 0.0);
          }
        }
      }
    }
    return ans;
  }

  //======================================================================
  double DirichletModel::Loglike(const Vector &nu, Vector &g, Matrix &h,
                                 uint nd) const {
    /* returns log likelihood for the parameters of a Dirichlet
       distribution with sufficient statistic sumlogpi(lo..hi).  If
       pi(1)(lo..hi)..pi(nobs)(lo..hi) are probability vectors, then
       sumlogpi(j) = sum_i log(pi(i,j))

       if(nd>0) then the g(lo..hi) is filled with the gradient (with
       respect to nu).  If nd>1 then hess(lo..hi)(lo..hi) is filled
       with the hessian (wrt nu).  Otherwise the algorithm can be called
       with either g or hess = 0.

    */

    const Vector &sumlogpi(suf()->sumlog());
    double nobs = suf()->n();
    Vector *G(nd > 0 ? &g : nullptr);
    Matrix *H(nd > 1 ? &h : nullptr);
    return dirichlet_loglike(nu, G, H, sumlogpi, nobs);
  }

  Vector DirichletModel::sim(RNG &rng) const {
    return rdirichlet_mt(rng, nu());
  }

  void DirichletModel::add_mixture_data(const Ptr<Data> &dp, double prob) {
    suf()->add_mixture_data(DAT(dp)->value(), prob);
  }
}  // namespace BOOM
