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
#include "Models/IRT/DafePcr.hpp"
#include "Models/IRT/Subject.hpp"
#include "Models/MvnModel.hpp"
#include "Models/MvtModel.hpp"

#include "Models/IRT/PartialCreditModel.hpp"
#include "Models/IRT/SubjectPrior.hpp"

#include "Samplers/MetropolisHastings.hpp"
#include "TargetFun/TargetFun.hpp"
#include "cpputil/ParamHolder.hpp"
#include "distributions.hpp"

namespace BOOM {
  namespace IRT {

    typedef DafePcrSubject DAFE;
    typedef DafePcrDataImputer IMP;
    typedef PartialCreditModel PCR;

    namespace {
      class SubjectTF : public TargetFun {
       public:
        SubjectTF(const Ptr<Subject> &s, const Ptr<SubjectPrior> &pri,
                  const Ptr<IMP> &Imp);
        double operator()(const Vector &) const;
        SubjectTF *clone() const { return new SubjectTF(*this); }

       private:
        Ptr<Subject> subject;
        Ptr<SubjectPrior> prior;
        Ptr<IMP> imp;
        mutable Vector wsp;
        mutable double ans;
        void loglike_contrib(std::pair<Ptr<Item>, Response>) const;
      };

      SubjectTF::SubjectTF(const Ptr<Subject> &s, const Ptr<SubjectPrior> &pri,
                           const Ptr<IMP> &Imp)
          : subject(s), prior(pri), imp(Imp), wsp(0), ans(0) {
        //    pri->add_data(s);  // THIS LOOKS DANGEROUS
      }

      double SubjectTF::operator()(const Vector &theta) const {
        ParamHolder ph(theta, subject->Theta_prm(), wsp);
        ans = prior->pdf(subject, true);
        for (auto &item_response : subject->item_responses()) {
          loglike_contrib(item_response);
        }
        return ans;
      }
      void SubjectTF::loglike_contrib(std::pair<Ptr<Item>, Response> ir) const {
        Ptr<Item> it = ir.first;
        Ptr<PCR> pcr = it.dcast<PCR>();
        Response r = ir.second;
        const Vector &u(imp->get_u(r));
        const Vector &eta(pcr->fill_eta(subject->Theta()));
        for (uint m = 0; m <= it->maxscore(); ++m) {
          ans += dexv(u[m], eta[m], 1, true);
        }
      }
    }  // namespace
    //======================================================================

    DAFE::DafePcrSubject(const Ptr<Subject> &Sub, const Ptr<SubjectPrior> &Pri,
                         const Ptr<IMP> &Imp, double Tdf, RNG &seeding_rng)
        : PosteriorSampler(seeding_rng),
          subject(Sub),
          pri(Pri),
          imp(Imp),
          sigsq(1.644934066848226),  // pi^2/6
          mean(Sub->Nscales()),
          Ivar(Sub->Nscales()) {
      SubjectTF target(subject, pri, imp);
      uint dim = subject->Nscales();
      SpdMatrix Ominv(dim);
      Ominv.set_diag(1.0);
      prop = new MvtIndepProposal(Vector(dim), Ominv, Tdf);
      sampler = new MetropolisHastings(target, prop);
    }
    //------------------------------------------------------------
    double DAFE::logpri() const { return pri->pdf(subject, true); }
    //------------------------------------------------------------
    void DAFE::draw() {
      set_moments();
      mean = sampler->draw(subject->Theta());
      subject->set_Theta(mean);
    }
    //------------------------------------------------------------
    void DAFE::set_moments() {
      Ivar = pri->siginv();              // correlation matrix
      mean = Ivar * pri->mean(subject);  // zero, typically
      for (auto &item_response : subject->item_responses()) {
        accumulate_moments(item_response);
      }
      mean = Ivar.solve(mean);
      prop->set_mu(mean);
      prop->set_ivar(Ivar);
    };
    //------------------------------------------------------------
    void DAFE::accumulate_moments(std::pair<Ptr<Item>, Response> ir) {
      Ptr<Item> it = ir.first;
      Ptr<PCR> pcr = it.dcast<PCR>();
      Response r = ir.second;
      const Vector &u(imp->get_u(r));
      const Vector &beta(pcr->beta());  // size == M+1
      double a = pcr->a();
      uint M = it->maxscore();
      uint which = pcr->which_subscale();
      //      bool d0_fixed(pcr->is_d0_fixed());
      //       if(d0_fixed){
      //     for(uint m=1; m<=M; ++m){
      //       double ma = m*a;
      //       double w = ma*ma;
      //       double tmp = (u[m]-beta[m-1])/ma;
      //       mean[which] += w*tmp/sigsq;
      //       Ivar(which,which) += w/sigsq;
      //     }
      //       }else{
      for (uint m = 0; m <= M; ++m) {
        double ma = (m + 1) * a;
        double w = ma * ma;
        double tmp = (u[m] - beta[m]) / ma;
        mean[which] += w * tmp / sigsq;
        Ivar(which, which) += w / sigsq;
      }
      //      }
    }
  }  // namespace IRT
}  // namespace BOOM
