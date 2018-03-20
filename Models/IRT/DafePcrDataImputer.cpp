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

#include <functional>

#include "Models/IRT/DafePcr.hpp"
#include "Models/IRT/Item.hpp"
#include "Models/IRT/PartialCreditModel.hpp"
#include "Models/IRT/Subject.hpp"

#include "cpputil/lse.hpp"  // for lse and lse2
#include "cpputil/report_error.hpp"

#include "distributions.hpp"

namespace BOOM {
  namespace IRT {

    typedef PartialCreditModel PCR;
    typedef DafePcrDataImputer IMP;

    IMP::DafePcrDataImputer(RNG &seeding_rng)
        : PosteriorSampler(seeding_rng), Eta(0), mu(-0.577215664901533) {}

    void IMP::add_item(const Ptr<PCR> &mod) {
      items.insert(mod);
      setup_latent_data(mod);
    }
    //------------------------------------------------------------
    void IMP::setup_latent_data(const Ptr<PCR> &mod) {
      for (const auto &subject : mod->subjects()) {
        setup_data_1(mod, subject);
      }
    }
    //------------------------------------------------------------
    inline void mod_not_found(const Ptr<PCR> &mod, const Ptr<Subject> &s) {
      ostringstream msg;
      msg << "item " << mod->id() << " not found  in subject " << s->id()
          << endl;
      report_error(msg.str());
    }
    //------------------------------------------------------------
    void IMP::setup_data_1(const Ptr<PCR> &mod, const Ptr<Subject> &subject) {
      Response r = subject->response(mod);
      if (!r) mod_not_found(mod, subject);
      latent_data[r] = Vector(1 + mod->maxscore());
    }
    //------------------------------------------------------------
    double IMP::logpri() const { return 0.0; }
    //------------------------------------------------------------
    Vector IMP::get_u(const Response &r, bool nag) const {
      std::map<Response, Vector>::const_iterator it = latent_data.find(r);
      if (it == latent_data.end()) {
        if (nag) {
          ostringstream msg;
          msg << "response not found in DafePcrDataImputer::get_u";
          report_error(msg.str());
        }
        return Vector();
      }
      const Vector &v(it->second);
      return v;
    }
    //---------------- for debugging only ----------------------
    void IMP::set_u(const Response &r, const Vector &u) { latent_data[r] = u; }
    //------------------------------------------------------------
    void IMP::draw() {
      for (auto &item : items) {
        draw_item_u(item);
      }
    }
    //------------------------------------------------------------
    void IMP::draw_item_u(const Ptr<PCR> &mod) {
      const SubjectSet &subjects(mod->subjects());
      for (auto &subject : subjects) {
        draw_one(mod, subject);
      }
    }
    //------------------------------------------------------------
    void IMP::draw_one(const Ptr<PCR> &mod, const Ptr<Subject> &subject) {
      Response r = subject->response(mod);
      if (!r) mod_not_found(mod, subject);
      Vector &u(latent_data[r]);
      Eta.resize(r->nlevels());
      const Vector &Eta(mod->fill_eta(subject->Theta()));
      impute_u(u, Eta, r->value());
    }
    //------------------------------------------------------------
    void IMP::impute_u(Vector &u, const Vector &eta, uint y) {
      double log_nc = lse(eta);
      double logzmin = rlexp(log_nc);
      uint M = u.size();
      for (uint m = 0; m < M; ++m) {
        if (m == y)
          u[m] = mu - logzmin;
        else
          u[m] = mu - lse2(logzmin, rlexp(eta[m]));
      }
    }

  }  // namespace IRT
}  // namespace BOOM
