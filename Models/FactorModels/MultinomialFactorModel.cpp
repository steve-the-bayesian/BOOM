/*
  Copyright (C) 2005-2024 Steven L. Scott

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

#include "Models/FactorModels/MultinomialFactorModel.hpp"
#include "cpputil/report_error.hpp"

namespace BOOM {

  namespace {
    using Site = FactorModels::MultinomialSite;
    using Visitor = FactorModels::MultinomialVisitor;
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

    Site::MultinomialSite(const std::string &id, int num_classes)
        : SiteBase(id),
          visit_probs_(new VectorParams(Vector(num_classes, .5)))
    {
      refresh_probs();
    }

    void Site::observe_visitor(const Ptr<Visitor> &visitor, int ntimes) {
      observed_visitors_[visitor] += ntimes;
    }

    void Site::set_probs(const Vector &probs) {
      visit_probs_->set(probs);
      refresh_probs();
    }

    Int Site::number_of_visits() const {
      Int ans = 0;
      for (const auto &it : observed_visitors_) {
        ans += it.second;
      }
      return ans;
    }

    void Site::refresh_probs() {
      logprob_ = log(visit_probs_->value());
      logprob_complement_ = log(1.0 - visit_probs_->value());
    }

  }  // namespace FactorModels

  MultinomialFactorModel::MultinomialFactorModel(int num_classes)
      : num_classes_(num_classes)
  {}

  MultinomialFactorModel::MultinomialFactorModel(const MultinomialFactorModel &rhs) {
    operator=(rhs);
  }

  MultinomialFactorModel & MultinomialFactorModel::operator=(
      const MultinomialFactorModel &rhs) {
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
        site->set_probs(parent->visit_probs());
      }
    }
    return *this;
  }

  MultinomialFactorModel * MultinomialFactorModel::clone() const {
    return new MultinomialFactorModel(*this);
  }

  void MultinomialFactorModel::add_data(const Ptr<Data> &data_point) {
    Ptr<MultinomialFactorData> native_data_point = data_point.dcast<MultinomialFactorData>();
    if (!native_data_point) {
      report_error("Data point could not be caset to MultinomialFactorData.");
    }
    record_visit(native_data_point->visitor_id(),
                 native_data_point->site_id(),
                 native_data_point->nvisits());
  }

  void MultinomialFactorModel::combine_data(const Model &rhs, bool) {
    try {
      const MultinomialFactorModel & rhs_model(dynamic_cast<const MultinomialFactorModel &>(rhs));
      for (const auto &visitor_it : rhs_model.visitors()) {
        const Ptr<Visitor> &visitor(visitor_it.second);
        for (const auto &site_it : visitor->sites_visited()) {
          const Ptr<Site> &site(site_it.first);
          int ntimes = site_it.second;
          record_visit(visitor->id(), site->id(), ntimes);
        }
      }
    } catch (const std::bad_cast &ex) {
      report_error("Could not convert model to MultinomialFactorModel");
    } catch (const std::exception &ex) {
      report_error("Unknown exception occurred in combine_data.");
    }
  }

  void MultinomialFactorModel::clear_data() {
    for (auto &site_it : sites_) {
      site_it.second->clear();
    }
    sites_.clear();

    for (auto &visitor_it : visitors_) {
      visitor_it.second->clear();
    }
    visitors_.clear();
  }

  void MultinomialFactorModel::record_visit(
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
      // site->set_observer(this, [this]() {this->sum_of_lambdas_current_ = false;});
    } else {
      site = site_it->second;
    }
    visitor->visit(site, nvisits);
    site->observe_visitor(visitor, nvisits);
  }

  Int MultinomialFactorModel::get_site_index(const std::string &id) const {
    auto it = sites_.find(id);
    if (it == sites_.end()) {
      return -1;
    } else {
      return std::distance(sites_.begin(), it);
    }
  }

  std::map<std::string, Int> MultinomialFactorModel::site_index_map() const {
    Int index = 0;
    std::map<std::string, Int> ans;
    for (const auto &it : sites_) {
      ans[it.first] = index++;
    }
    return ans;
  }

  Ptr<Site> MultinomialFactorModel::site(const std::string &id) const {
    auto it = sites_.find(id);
    if (it == sites_.end()) {
      return nullptr;
    } else {
      return it->second;
    }
  }

  Ptr<Visitor> MultinomialFactorModel::visitor(const std::string &id) const {
    auto it = visitors_.find(id);
    if (it == visitors_.end()) {
      return nullptr;
    } else {
      return it->second;
    }
  }

}  // namespace BOOM
