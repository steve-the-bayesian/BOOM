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

    void Visitor::merge(const Visitor &rhs) {
      for (const auto &sites_el : rhs.sites_visited_) {
        int count = sites_el.second;
        const Ptr<Site> &site(sites_el.first);
        auto it = sites_visited_.find(site);
        if (it == sites_visited_.end()) {
          sites_visited_[site] = count;
        } else {
          it->second += count;
        }
      }
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

    void Site::merge(const Site &rhs) {
      if (rhs.id() != id()) {
        report_error("Attempt to merge sites with different ID's.");
      }
      for (const auto &visitor_el : rhs.observed_visitors_) {
        int count = visitor_el.second;
        const Ptr<Visitor> &visitor(visitor_el.first);
        auto it = observed_visitors_.find(visitor);
        if (it == observed_visitors_.end()) {
          observed_visitors_[visitor] = count;
        } else {
          it->second += count;
        }
      }
    }

  }  // namespace FactorModels

  MultinomialFactorModel::MultinomialFactorModel(int num_classes,
                                                 const std::string &default_site_name)
      : num_classes_(num_classes),
        default_site_name_(default_site_name)
  {}

  MultinomialFactorModel::MultinomialFactorModel(const MultinomialFactorModel &rhs) {
    operator=(rhs);
  }

  MultinomialFactorModel & MultinomialFactorModel::operator=(
      const MultinomialFactorModel &rhs) {
    if (&rhs != this) {
      default_site_name_ = rhs.default_site_name_;
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

  void MultinomialFactorModel::add_site(const Ptr<Site> &site) {
    sites_[site->id()] = site;
  }

  void MultinomialFactorModel::add_data(const Ptr<Data> &data_point) {
    Ptr<MultinomialFactorData> native_data_point =
        data_point.dcast<MultinomialFactorData>();
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
      combine_data_mt(rhs_model);
    } catch (const std::bad_cast &ex) {
      report_error("Could not convert model to MultinomialFactorModel");
    } catch (const std::exception &ex) {
      report_error("Unknown exception occurred in combine_data.");
    }
  }

  void MultinomialFactorModel::combine_data_mt(const MultinomialFactorModel &rhs) {
    // Step 1 Merge each site from rhs into this->sites_.
    for (const auto &rhs_site : rhs.sites_) {
      const std::string &site_id(rhs_site.first);
      const Ptr<Site> &site(rhs_site.second);
      auto it = sites_.find(site_id);
      if (it == sites_.end()) {
        // If rhs has a site we don't we can just steal its pointer.
        sites_[site_id] = site;
      } else {
        // If we both have a site then merge the visits.
        sites_[site_id]->merge(*site);
      }
    }

    // Step 2: Merge each visitor.
    for (const auto &rhs_visitor : rhs.visitors_) {
      const std::string &visitor_id(rhs_visitor.first);
      const Ptr<Visitor> &visitor(rhs_visitor.second);
      auto it = visitors_.find(visitor_id);
      if (it == visitors_.end()) {
        visitors_[visitor_id] = visitor;
      } else {
        it->second->merge(*visitor);
      }
    }
  }

  void MultinomialFactorModel::clear_data() {
    for (auto &site_el : sites_) {
      site_el.second->clear();
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

  void MultinomialFactorModel::extract_data(
      std::vector<std::string> &user_ids_output,
      std::vector<std::string> &site_ids_output,
      std::vector<int> &count_output) const {
    user_ids_output.clear();
    site_ids_output.clear();
    count_output.clear();

    for (const auto &visitor_el : visitors_) {
      const Ptr<Visitor> &visitor(visitor_el.second);
      const std::string &visitor_id = visitor->id();
      for (const auto &site_el : visitor->sites_visited()) {
        const std::string &site_id(site_el.first->id());
        int num_visits = site_el.second;
        user_ids_output.push_back(visitor_id);
        site_ids_output.push_back(site_id);
        count_output.push_back(num_visits);
      }
    }
  }

}  // namespace BOOM
