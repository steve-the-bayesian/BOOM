// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2010 Steven L. Scott

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

#ifndef BOOM_TIM_HPP
#define BOOM_TIM_HPP
#include <functional>
#include "Samplers/MetropolisHastings.hpp"
#include "Samplers/Sampler.hpp"
#include "numopt.hpp"

namespace BOOM {

  // TIM stands for Tailored Independence Metropolis.  Use it when the
  // target function is approximately the log of a multivariate normal
  // distribution.
  //
  // TIM works by approximating the target distribution by a
  // multivariate normal or multivariate T distribution.  It locates
  // the mode of the target distribution using a call to max_nd(),
  // then approximates the curvature of the target by evaluating the
  // Hessian at the mode.  The implied normal or T distribution is
  // used as a proposal for an independence Metropolis sampler.
  //
  // TIM is also powerful, like Tim the enchanter
  // http://www.youtube.com/watch?v=JTbrIo1p-So
  class TIM : public MetropolisHastings {
   public:
    // Args:
    //   logf: A function or functor taking arguments
    //     theta:  A vector at which to evaluate logf
    //     gradient:  The gradient of logf at theta.
    //     hessian:  The hessian of logf at theta.
    //     nd: The number of derivatives to take.  If nd < 2 then
    //       hessian is not referenced.  If nd < 1 then gradient is
    //       not referenced.
    //     logf returns the log of the un-normalized target distribution
    //     at theta.
    //   nu:  The degrees of freedom parameter to use for the
    explicit TIM(const std::function<double(const Vector &, Vector &, Matrix &, int)>
            &logf,
        double nu = 3, RNG *rng = 0);

    TIM(const BOOM::Target &logf, const BOOM::dTarget &dlogf,
        const BOOM::d2Target &d2logf, double nu = 3, RNG *rng = 0);

    Vector draw(const Vector &old) override;

    // In the typical use case the mode is located each iteration.  If
    // you want to avoid locating the mode use fix_mode(true).  To
    // turn mode location back off again use fix_mode(false).
    void fix_mode(bool yn = true);

    // Locates the mode of the target distribution, with 'old' as a
    // starting value.  Returns 'true' if the mode was located
    // successfully, 'false' otherwise.
    //
    // Users will normally not have to call locate_mode directly, but
    // you can if you want.
    bool locate_mode(const Vector &old);

    // In some rare cases (e.g. spike and slab models with varying
    // dimensions) you may want to store the mode in some other
    // location.  In those cases you can use set_mode to store a
    // previously found mode.  Once set_mode() has been called the
    // supplied modal approximation will be used until set_mode is
    // called again.
    void set_mode(const Vector &location, const Matrix &hessian);

    // Once locate_mode has been called, the following can be called
    // to get the location of the mode and the inverse of the variance
    // of the approximating normal at the mode (i.e. the negative
    // Hessian).
    const Vector &mode() const;
    const SpdMatrix &ivar() const;

   private:
    void report_failure(const Vector &old);
    Ptr<MvtIndepProposal> create_proposal(int dim, double nu);
    void check_proposal(int dim);

    Ptr<MvtIndepProposal> prop_;
    double nu_;
    BOOM::Target f_;
    BOOM::dTarget df_;
    BOOM::d2Target d2f_;
    Vector cand_;
    Vector dummy_gradient_;
    Matrix dummy_Hessian_;
    bool mode_is_fixed_;
    bool mode_has_been_found_;
  };
}  // namespace BOOM
#endif
