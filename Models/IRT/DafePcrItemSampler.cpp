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
#include "Models/IRT/PartialCreditModel.hpp"
#include "Models/IRT/Subject.hpp"
#include "Models/MvnModel.hpp"
#include "Models/MvtModel.hpp"
#include "Samplers/MetropolisHastings.hpp"
#include "TargetFun/TargetFun.hpp"
#include "cpputil/ParamHolder.hpp"
#include "cpputil/math_utils.hpp"
#include "distributions.hpp"

#include <functional>
#include <iomanip>

namespace BOOM {
  namespace IRT {
    typedef DafePcrItemSampler ISAM;
    typedef PartialCreditModel PCR;
    typedef DafePcrDataImputer IMP;
    //======================================================================
    class PcrBetaHolder {
     public:
      PcrBetaHolder(const Vector &newb, const Ptr<PartialCreditModel> &pcr,
                    Vector &V)
          : v(V), mod(pcr) {
        v = mod->beta();
        mod->set_beta(newb);
      }
      ~PcrBetaHolder() { mod->set_beta(v); }

     private:
      Vector &v;
      Ptr<PartialCreditModel> mod;
    };
    //======================================================================
    class ItemDafeTF : public TargetFun {
      // evaluates posterior probability of a vector b
     public:
      ItemDafeTF(const Ptr<PCR> &it, const Ptr<MvnModel> &pri,
                 const Ptr<IMP> &Imp)
          : mod(it), prior(pri), imp(Imp), ans(0), t(it->parameter_vector()) {}
      double operator()(const Vector &b) const;
      ItemDafeTF *clone() const { return new ItemDafeTF(*this); }

     private:
      Ptr<PCR> mod;
      Ptr<MvnModel> prior;
      Ptr<IMP> imp;  // must be assigned to  'item'.
      // contains imputed latent data
      mutable Vector tmpbeta;
      mutable double ans;
      std::vector<Ptr<Params>> t;
      void logp_sub(const Ptr<Subject> &s) const;
    };
    void ItemDafeTF::logp_sub(const Ptr<Subject> &s) const {
      Response r = s->response(mod);
      const Vector &u(imp->get_u(r, true));
      const Vector &Theta(s->Theta());
      const Vector &eta(mod->fill_eta(Theta));
      assert(u.size() == eta.size());
      for (uint i = 0; i < u.size(); ++i) ans += dexv(u[i], eta[i], 1.0, true);
    }
    double ItemDafeTF::operator()(const Vector &b) const {
      PcrBetaHolder ph(b, mod, tmpbeta);
      if (mod->a() <= 0) return BOOM::negative_infinity();
      ans = 0.0;
      for (auto &subject : mod->subjects()) {
        logp_sub(subject);
      }
      return ans;
    }
    //======================================================================
    ISAM::DafePcrItemSampler(const Ptr<PCR> &Mod,
                             const Ptr<DafePcrDataImputer> &Imp,
                             const Ptr<MvnModel> &Prior, double Tdf,
                             RNG &seeding_rng)
        : PosteriorSampler(seeding_rng),
          mod(Mod),
          prior(Prior),
          imp(Imp),
          sigsq(1.644934066848226)  // pi^2/6
    {
      Matrix X(Mod->X(1.0));
      xtx = SpdMatrix(X.ncol());
      xtu = Vector(X.ncol());

      ItemDafeTF target(mod, prior, imp);
      uint dim = mod->beta().size();
      SpdMatrix Ominv(dim);
      Ominv.set_diag(1.0);
      prop = new MvtIndepProposal(Vector(dim), Ominv, Tdf);
      sampler = new MetropolisHastings(target, prop);
    }
    //------------------------------------------------------------
    double ISAM::logpri() const { return prior->logp(mod->beta()); }
    //------------------------------------------------------------
    void ISAM::draw() {
      get_moments();  // fills xtx and xtu
      prop->set_mu(mean);
      prop->set_ivar(ivar);
      Vector b = mod->beta();
      b = sampler->draw(b);
      mod->set_beta(b);
      mod->sync_params();
    }
    //----------------------------------------------------------------------
    void ISAM::get_moments() {
      xtx = 0.0;
      xtu = 0.0;
      for (auto &subject : mod->subjects()) {
        accumulate_moments(subject);
      }
      ivar = as_symmetric(xtx) / sigsq + prior->siginv();
      mean = ivar.solve(prior->siginv() * prior->mu() + xtu / sigsq);
    }
    //----------------------------------------------------------------------
    void ISAM::accumulate_moments(const Ptr<Subject> &s) {
      const Matrix &X(mod->X(s->Theta()));
      xtx.add_inner(X);
      Response r = s->response(mod);
      const Vector &u(imp->get_u(r, true));
      xtu.add_Xty(X, u);
    }
  }  // namespace IRT
}  // namespace BOOM
