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
#include <iomanip>

#include "cpputil/report_error.hpp"

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

    template <class OBJ>
    Ptr<OBJ> get_by_id(int64_t id, const std::vector<Ptr<OBJ>> &things) {
      auto it = std::lower_bound(
          things.begin(), things.end(), id, IdLess<OBJ>());
      if (it == things.end() || (*it)->id() != id) {
        return nullptr;
      } else {
        return *it;
      }
    }

  }

  PoissonFactorData * PoissonFactorData::clone() const {
    return new PoissonFactorData(*this);
  }

  std::ostream &PoissonFactorData::display(std::ostream &out) const {
    out << std::setw(12) << visitor_id_
        << std::setw(12) << site_id_
        << nvisits_;
    return out;
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

    Site::Site(int64_t id, int num_classes)
        : id_(id),
          visitation_rates_(new VectorParams(num_classes, 1.0)),
          log_lambda_(log(visitation_rates_->value())),
          prior_a_(new VectorParams(num_classes, 1.0)),
          prior_b_(new VectorParams(num_classes, 1.0))
    {}

    void Site::observe_visitor(const Ptr<Visitor> &visitor, int ntimes) {
      observed_visitors_[visitor] += ntimes;
    }

    void Site::set_lambda(const Vector &lambda) {
      visitation_rates_->set(lambda);
      log_lambda_ = log(lambda);
    }

    void Site::set_prior(const Vector &prior_a, const Vector &prior_b) {
      prior_a_.reset(new VectorParams(prior_a));
      prior_b_.reset(new VectorParams(prior_b));
    }

  }  // namespace PoissonFactor



  PoissonFactorModel::PoissonFactorModel(int num_classes)
      : sum_of_lambdas_(num_classes, negative_infinity())
  {}

  PoissonFactorModel::PoissonFactorModel(const PoissonFactorModel &rhs) {
    *this = rhs;
  }

  PoissonFactorModel & PoissonFactorModel::operator=(const PoissonFactorModel &rhs) {
    if (&rhs != this) {
      for (const auto &visitor : rhs.visitors_) {
        for (const auto &it : visitor->sites_visited()) {
          int64_t site_id = it.first->id();
          int ntimes = it.second;
          record_visit(visitor->id(), site_id, ntimes);
        }
      }

      for (auto &visitor : visitors_) {
        Ptr<Visitor> parent = rhs.get_visitor(visitor->id());
        visitor->set_class_probabilities(parent->class_probabilities());
        visitor->set_class_member_indicator(parent->imputed_class_membership());
      }

      for (auto &site : sites_) {
        Ptr<Site> parent = rhs.get_site(site->id());
        site->set_prior(parent->prior_a(), parent->prior_b());
      }

      sum_of_lambdas_ = rhs.sum_of_lambdas_;
    }
    return *this;
  }

  PoissonFactorModel * PoissonFactorModel::clone() const {
    return new PoissonFactorModel(*this);
  }

  PoissonFactorModel::PoissonFactorModel(PoissonFactorModel &&rhs)
      : DataPolicy(std::move(rhs)),
        visitors_(std::move(rhs.visitors_)),
        sites_(std::move(rhs.sites_)),
        sum_of_lambdas_(rhs.sum_of_lambdas_)
  {}

  void PoissonFactorModel::add_data(const Ptr<PoissonFactorData> &data_point) {
    record_visit(data_point->visitor_id(),
                 data_point->site_id(),
                 data_point->nvisits());
  }

  void PoissonFactorModel::add_data(const Ptr<Data> &data_point) {
    Ptr<PoissonFactorData> native_data_point = data_point.dcast<PoissonFactorData>();
    if (!native_data_point) {
      report_error("Data point could not be cast to the native data type.");
    }
    add_data(native_data_point);
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
      site.reset(new Site(site_id, number_of_classes()));
      sites_.insert(site_it, site);
    } else {
      site = *site_it;
    }
    visitor->visit(site, nvisits);
    site->observe_visitor(visitor, nvisits);
  }

  Ptr<Site> PoissonFactorModel::get_site(int64_t id) const {
    return get_by_id(id, sites_);
  }

  Ptr<Visitor> PoissonFactorModel::get_visitor(int64_t id) const {
    return get_by_id(id, visitors_);
  }

  const Vector & PoissonFactorModel::sum_of_lambdas() const {
    if (!std::isfinite(sum_of_lambdas_[0])) {
      sum_of_lambdas_ = 0.0;
      for (const auto &site : sites_) {
        sum_of_lambdas_ += site->lambda();
      }
    }
    return sum_of_lambdas_;
  }


}  // namespace BOOM
