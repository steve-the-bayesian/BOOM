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

namespace BOOM {

  namespace {
    using Site = FactorModels::MultinomialSite;
    using Visitor = FactorModels::MultinomialVisitor;
  }

  
  namespace FactorModels {

    void Visitor::visit(const Ptr<Site> &site, int ntimes) {
      sites_visited_[site] += ntimes;
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

    void Site::refresh_probs() {
      logprob_ = log(visit_probs_->value());
      logprob_complement_ = log(1.0 - visit_probs_->value());
    }
    
  }  // namespace FactorModels

  MultinomialFactorModel::MultinomialFactorModel(int num_classes)
      : num_classes_(num_classes)
  {}

  ////////////////
  ////////////////  Add all the copy constructors.
  ////////////////
  ////////////////
  ////////////////
  
  MultinomialFactorModel * MultinomialFactorModel::clone() const {
    return new MultinomialFactorModel(*this);
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
