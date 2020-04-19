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
#include "Models/IRT/DafePcrRwm.hpp"
#include "cpputil/ParamHolder.hpp"
#include "cpputil/math_utils.hpp"

#include "Models/IRT/PartialCreditModel.hpp"
#include "Models/IRT/SubjectPrior.hpp"
#include "Models/MvnModel.hpp"
#include "Models/MvtModel.hpp"

#include "Samplers/MetropolisHastings.hpp"
#include "TargetFun/TargetFun.hpp"

namespace BOOM {
  namespace IRT {

    typedef PartialCreditModel PCR;

    class SubjectObsTF : public TargetFun {
     public:
      SubjectObsTF(const Ptr<Subject> &s, const Ptr<SubjectPrior> &p);
      SubjectObsTF *clone() const;
      double operator()(const Vector &theta) const;

     private:
      Ptr<Subject> sub;
      Ptr<SubjectPrior> pri;
      Ptr<VectorParams> Theta;
      mutable Vector wsp;
    };
    typedef SubjectObsTF TF;
    //------------------------------------------------------------

    TF::SubjectObsTF(const Ptr<Subject> &s, const Ptr<SubjectPrior> &p)
        : sub(s), pri(p), Theta(s->Theta_prm()), wsp(Theta->value()) {}

    TF *TF::clone() const { return new TF(*this); }

    double TF::operator()(const Vector &v) const {
      ParamHolder ph(v, Theta, wsp);
      double ans = pri->pdf(sub, true);
      if (ans == BOOM::negative_infinity()) return ans;
      ans += sub->loglike();
      return ans;
    }

    //======================================================================
    typedef DafePcrRwmSubjectSampler SS;

    SS::DafePcrRwmSubjectSampler(const Ptr<Subject> &s,
                                 const Ptr<SubjectPrior> &p, double Tdf,
                                 RNG &seeding_rng)
        : PosteriorSampler(seeding_rng),
          sub(s),
          prior(p),
          sigsq(1.644934066848226),
          ivar(s->Nscales()),
          Theta(s->Nscales()) {
      uint Ndim = s->Nscales();
      TF target(sub, prior);
      SpdMatrix Siginv(Ndim);
      Siginv.set_diag(1.0);
      prop = new MvtRwmProposal(Siginv, Tdf);
      sampler = new MetropolisHastings(target, prop);
    }

    //------------------------------------------------------------

    void SS::draw() {
      get_moments();
      prop->set_ivar(ivar);
      Theta = sampler->draw(sub->Theta());
      sub->set_Theta(Theta);
    }

    double SS::logpri() const { return prior->pdf(sub, true); }

    void SS::get_moments() {
      ivar = prior->siginv();
      for (auto &item_response : sub->item_responses()) {
        accumulate_moments(item_response);
      }
    }

    void SS::accumulate_moments(std::pair<Ptr<Item>, Response> ir) {
      Ptr<Item> it = ir.first;
      Ptr<PCR> pcr = it.dcast<PCR>();
      double a = pcr->a();
      uint M = it->maxscore();
      uint which = pcr->which_subscale();
      for (uint m = 1; m <= M; ++m) {
        double ma = m * a;
        double w = ma * ma;
        ivar(which, which) += w / sigsq;
      }
    }

  }  // namespace IRT
}  // namespace BOOM
