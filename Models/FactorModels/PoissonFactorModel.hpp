#ifndef BOOM_MODELS_FACTOR_MODELS_POISSON_FACTOR_MODEL_HPP_
#define BOOM_MODELS_FACTOR_MODELS_POISSON_FACTOR_MODEL_HPP_

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

#include "Models/Policies/ManyParamPolicy.hpp"
#include "Models/Policies/IID_DataPolicy.hpp"
#include "Models/Policies/PriorPolicy.hpp"

namespace BOOM {

  class PoissonFactorData
      : public Data {
   public:
   private:
    int64_t user_id_;
    int64_t site_id_;
    int visit_count;
  };

  namespace {
    class Site;

    //----------------------------------------------------------------------
    // A visitor belongs to one of K classes.
    class Visitor {
     public:
      Visitor(int64_t id)
          : id_(id)
      {}

      void visit(Site &site, int ntimes);
      void set_class_probabilities(const Vector &probs);
      const Vector &class_probabilities() const {
        return class_probabilities->value();}

     private:
      int64_t id_;
      Ptr<VectorParams> class_probabilities_;
      int imputed_class_membership_;

      std::map<int64_t, int> sites_visited_;
    };

    //----------------------------------------------------------------------
    class Site {
     public:
      Site(int64_t id)
          : id_(id)
      {}

      // Record one or more visits by
      void observe_visitor(const Visitor &visitor, int ntimes);
      void observe_visitor(int64_t visitor, int ntimes);

      int64_t id() const {return id_;}

      bool operator<(const Site &rhs) const {
        return id_ < rhs.id();
      }

     private:
      int64_t id_;
      std::vector<std::pair<int64_t, int>> visitors_;

      // Element k is the Poisson rate at which a visitor of class k visits the site.
      Ptr<VectorParams> visitation_rates_;
    };
  }

  class PoissonFactorModel
      : public ManyParamPolicy,
        public IID_DataPolicy,
        public PriorPolicy
  {
   public:

    void record_visit(int64_t visitor_id, int64_t site_id, int ntimes = 1);

   private:
    std::map<int64_t, Visitor> visitors_;
    std::map<int64_t, Site> sites_;
  };



}  // namespace BOOM

#endif //  BOOM_MODELS_FACTOR_MODELS_POISSON_FACTOR_MODEL_HPP_
