/*
  Copyright (C) 2005-2022 Steven L. Scott

  This library is free software; you can redistribute it and/or modify it under
  the terms of the GNU Lesser General Public License as published by the Free
  Software Foundation; either version 2.1 of the License, or (at your option)
  any later version.

  This library is distributed in the hope that it will be useful, but WITHOUT
  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
  FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
  details.

  You should have received a copy of the GNU Lesser General Public License along
  with this library; if not, write to the Free Software Foundation, Inc., 51
  Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA
*/

#include "Models/FactorModels/PoissonFactorModel.hpp"

#include <algorithm>

namespace BOOM {

  namespace {
    using PoissonFactor::Site;
    using PoissonFactor::Visitor;

    // A "less than" operator for Ptr<thing> where thing has an id() method.
    template <class OBJ>
    class IdLess {
     public:
      bool operator()(const Ptr<OBJ> &lhs, const Ptr<OBJ> &rhs) const {
        return lhs->id() < rhs->id();
      }

      bool operator()(const Ptr<OBJ> &lhs, int64_t rhs) const {
        return lhs->id() < rhs;
      }

      bool operator()(int64_t lhs, const Ptr<OBJ> &rhs) const {
        return lhs < rhs->id();
      }
    };

  }

  namespace PoissonFactor {
    void intrusive_ptr_add_ref(Site *site) {site->up_count();}

    void intrusive_ptr_release(Site *site) {
      site->down_count();
      if (site->ref_count() == 0) {
        delete site;
      }
    }

    void intrusive_ptr_add_ref(Visitor *visitor) {visitor->up_count();}

    void intrusive_ptr_release(Visitor *visitor) {
      visitor->down_count();
      if (visitor->ref_count() == 0) {
        delete visitor;
      }
    }

    void Visitor::visit(const Ptr<Site> &site, int ntimes) {
      sites_visited_[site] += ntimes;
    }

    void Site::observe_visitor(const Ptr<Visitor> &visitor, int ntimes) {
      observed_visitors_[visitor] += ntimes;
    }

    void Site::set_lambda(const Vector &lambda) {
      visitation_rates_->set(lambda);
      log_lambda_ = log(lambda);
    }

  }  // namespace PoissonFactor



  PoissonFactorModel::PoissonFactorModel(int num_classes)
      : sum_of_lambdas_(num_classes, 0.0)
  {}

  void PoissonFactorModel::add_data(const Ptr<PoissonFactorData> &data_point) {
    record_visit(data_point->visitor_id(),
                 data_point->site_id(),
                 data_point->nvisits());
  }

  void PoissonFactorModel::record_visit(
      int64_t visitor_id, int64_t site_id, int nvisits) {

    // Get the Visitor pointer, or make a new one.
    auto visitor_it = std::lower_bound(
        visitors_.begin(), visitors_.end(), visitor_id, IdLess<Visitor>());
    Ptr<Visitor> visitor;
    if (visitor_it == visitors_.end() || (*visitor_it)->id() != visitor_id) {
      visitor.reset(new Visitor(visitor_id, number_of_classes()));
      visitors_.insert(visitor_it, visitor);
    } else {
      visitor = *visitor_it;
    }

    auto site_it = std::lower_bound(
        sites_.begin(), sites_.end(), site_id, IdLess<Site>());
    Ptr<Site> site;
    if (site_it == sites_.end() || (*site_it)->id() != site_id) {
    } else {
      site = *site_it;
    }
    visitor->visit(site, nvisits);
    site->observe_visitor(visitor, nvisits);
  }


}  // namespace BOOM
