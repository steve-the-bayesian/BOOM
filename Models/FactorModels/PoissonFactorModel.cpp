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
    using Site = FactorModels::PoissonSite;
    using Visitor = FactorModels::PoissonVisitor;
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

  namespace FactorModels {
    void Visitor::visit(const Ptr<Site> &site, int ntimes) {
      sites_visited_[site] += ntimes;
    }

    Int Visitor::number_of_visits() const {
      Int ans = 0;
      for (const auto &it : sites_visited_) {
        ans += it.second;
      }
      return ans;
    }

    Site::PoissonSite(const std::string &id, int num_classes)
        : SiteBase(id),
          visitation_rates_(new VectorParams(num_classes, 1.0)),
          log_lambda_(log(visitation_rates_->value()))
    {}

    void Site::observe_visitor(const Ptr<Visitor> &visitor, int ntimes) {
      observed_visitors_[visitor] += ntimes;
    }

    void Site::set_lambda(const Vector &lambda) {
      visitation_rates_->set(lambda);
      log_lambda_ = log(lambda);
    }

    Int Site::number_of_visits() const {
      Int ans = 0;
      for (const auto &it : observed_visitors_) {
        ans += it.second;
      }
      return ans;
    }
    
    Matrix Site::visitor_counts() const {
      Matrix ans(number_of_classes(), 2, 0.0);
      for (const auto &it : observed_visitors_) {
        int visit_count = it.second;
        const Ptr<Visitor> visitor = it.first;
        int level = visitor->imputed_class_membership();
        ans(level, 0) += visit_count;
        ans(level, 1) += 1;
      }
      return ans;
    }

  }  // namespace FactorModels


  PoissonFactorModel::PoissonFactorModel(int num_classes)
      : num_classes_(num_classes),
        sum_of_lambdas_(num_classes, 0.0),
        sum_of_lambdas_current_(false)
  {}

  PoissonFactorModel::PoissonFactorModel(const PoissonFactorModel &rhs) {
    *this = rhs;
  }

  PoissonFactorModel & PoissonFactorModel::operator=(const PoissonFactorModel &rhs) {
    if (&rhs != this) {
      clear_data();
      for (const auto &visitor_it : rhs.visitors_) {
        const Ptr<Visitor> &visitor(visitor_it.second);
        for (const auto &it : visitor->sites_visited()) {
          const std::string &site_id = it.first->id();
          int ntimes = it.second;
          record_visit(visitor->id(), site_id, ntimes);
        }
      }

      for (auto &visitor_it : visitors_) {
        Ptr<Visitor> &visitor(visitor_it.second);
        Ptr<Visitor> parent = rhs.visitor(visitor->id());
        visitor->set_class_probabilities(parent->class_probabilities());
        visitor->set_class_member_indicator(parent->imputed_class_membership());
      }

      for (auto &site_it : sites_) {
        Ptr<Site> &site(site_it.second);
        Ptr<Site> parent = rhs.site(site->id());
        site->set_lambda(parent->lambda());
      }
    }
    return *this;
  }

  PoissonFactorModel * PoissonFactorModel::clone() const {
    return new PoissonFactorModel(*this);
  }

  PoissonFactorModel::PoissonFactorModel(PoissonFactorModel &&rhs)
      : visitors_(std::move(rhs.visitors_)),
        sites_(std::move(rhs.sites_)),
        sum_of_lambdas_current_(false)
  {}

  void PoissonFactorModel::add_data(const Ptr<Data> &data_point) {
    Ptr<PoissonFactorData> native_data_point = data_point.dcast<PoissonFactorData>();
    if (!native_data_point) {
      report_error("Data point could not be cast to the native data type.");
    }
    record_visit(native_data_point->visitor_id(),
                 native_data_point->site_id(),
                 native_data_point->nvisits());
  }

  void PoissonFactorModel::clear_data() {
    for (auto &site_it : sites_) {
      site_it.second->clear();
    }
    sites_.clear();

    for (auto &visitor_it : visitors_) {
      visitor_it.second->clear();
    }
    visitors_.clear();
  }

  void PoissonFactorModel::combine_data(
      const Model &, bool) {
    report_error("combine_data is not implemented for PoissonFactorModel.");
  }

  void PoissonFactorModel::record_visit(
      const std::string & visitor_id,
      const std::string & site_id,
      int nvisits) {
    // Get the Visitor pointer, or make a new one.
    auto visitor_it = visitors_.find(visitor_id);
    Ptr<Visitor> visitor;
    if (visitor_it == visitors_.end() || visitor_it->second->id() != visitor_id) {
      visitor.reset(new Visitor(visitor_id, number_of_classes()));
      visitors_[visitor_id] = visitor;
    } else {
      visitor = visitor_it->second;
    }

    auto site_it = sites_.find(site_id);
    Ptr<Site> site;
    if (site_it == sites_.end() || site_it->second->id() != site_id) {
      site.reset(new Site(site_id, number_of_classes()));
      sites_[site_id] = site;
      site->set_observer(this, [this]() {this->sum_of_lambdas_current_ = false;});
    } else {
      site = site_it->second;
    }
    visitor->visit(site, nvisits);
    site->observe_visitor(visitor, nvisits);
  }

  Ptr<Site> PoissonFactorModel::site(const std::string &id) const {
    auto it = sites_.find(id);
    if (it == sites_.end()) {
      return nullptr;
    } else {
      return it->second;
    }
  }

  Ptr<Visitor> PoissonFactorModel::visitor(const std::string &id) const {
    auto it = visitors_.find(id);
    if (it == visitors_.end()) {
      return nullptr;
    } else {
      return it->second;
    }
  }

  const Vector &PoissonFactorModel::sum_of_lambdas() const {
    if (!sum_of_lambdas_current_) {
      if (sum_of_lambdas_.size() != number_of_classes()) {
        sum_of_lambdas_.resize(number_of_classes());
      }
      sum_of_lambdas_ *= 0;
      for (const auto &it : sites_) {
        sum_of_lambdas_ += it.second->lambda();
      }
      sum_of_lambdas_current_ = true;
    }
    return sum_of_lambdas_;
  }

}  // namespace BOOM
