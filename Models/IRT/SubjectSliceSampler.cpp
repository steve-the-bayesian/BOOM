/*
  Copyright (C) 2006 Steven L. Scott

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
#include "Models/IRT/SubjectSliceSampler.hpp"
#include "Models/IRT/Subject.hpp"
#include "Models/IRT/SubjectPrior.hpp"

#include "Samplers/SliceSampler.hpp"

#include "cpputil/ParamHolder.hpp"
#include "cpputil/math_utils.hpp"

namespace BOOM {
  namespace IRT {

    typedef SubjectTF TF;
    SubjectTF::SubjectTF(const Ptr<Subject> &subject,
                         const Ptr<SubjectPrior> &prior)
        : sub(subject), pri(prior), prms(sub->Theta_prm()), wsp(sub->Theta()) {}

    double SubjectTF::operator()(const Vector &v) const {
      ParamHolder ph(v, prms, wsp);
      double ans = pri->pdf(sub, true);
      if (ans == BOOM::negative_infinity()) return ans;
      ans += sub->loglike();
      return ans;
    }

    //======================================================================

    typedef SubjectSliceSampler SSS;
    SSS::SubjectSliceSampler(const Ptr<Subject> &s, const Ptr<SubjectPrior> &p,
                             RNG &seeding_rng)
        : PosteriorSampler(seeding_rng),
          sub(s),
          pri(p),
          target(sub, pri),
          sam(new SliceSampler(target)) {}

    SSS *SSS::clone() const { return new SSS(*this); }

    void SSS::draw() {
      Theta = sam->draw(sub->Theta());
      sub->set_Theta(Theta);
    }

    double SSS::logpri() const { return pri->pdf(sub, true); }

  }  // namespace IRT
}  // namespace BOOM
