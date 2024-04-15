#ifndef BOOM_FACTOR_MODELS_VISITOR_HPP_
#define BOOM_FACTOR_MODELS_VISITOR_HPP_

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

#include "cpputil/RefCounted.hpp"
#include "Models/ParamTypes.hpp"
#include <string>

namespace BOOM {
  namespace FactorModels {

    // A "less than" operator for Ptr<Thing> where Thing has an id() method.
    // This allows us to use Ptr<Thing> as the index of a std::map.
    template <class OBJ>
    class IdLess {
     public:
      bool operator()(const Ptr<OBJ> &lhs, const Ptr<OBJ> &rhs) const {
        return lhs->id() < rhs->id();
      }

      bool operator()(const Ptr<OBJ> &lhs, const std::string &rhs) const {
        return lhs->id() < rhs;
      }

      bool operator()(const std::string &lhs, const Ptr<OBJ> &rhs) const {
        return lhs < rhs->id();
      }
    };

    // A base class containing the common aspects of Visitors to specific types
    // of Sites.
    class VisitorBase : private RefCounted {
     public:
      // Args:
      //   id:  The unique ID for this visitor.
      //   num_classes: The number of latent classes to which this visitor might
      //     belong.
      VisitorBase(const std::string &id, int num_classes)
          : id_(id),
            class_probabilities_(new VectorParams(
                Vector(num_classes, 1.0 / num_classes))),
            imputed_class_membership_(-1)            
      {}

      virtual ~VisitorBase() {}

      const std::string & id() const {return id_;}

      // Set the class membership probabilities for this class.  These are the
      // conditional probabilities of visitor class membership given all other
      // information.
      void set_class_probabilities(const Vector &probs) {
        if (!class_probabilities_) {
          class_probabilities_.reset(new VectorParams(probs));
        } else {
          class_probabilities_->set(probs);
        }
      }

      // Accessor for the class membership probabilities.
      const Vector &class_probabilities() const {
        return class_probabilities_->value();}

      // Return the imputed class membership indicator (presumably drawn from
      // class_membership_probabilities).
      int imputed_class_membership() const {
        return imputed_class_membership_;
      }

      // Set the imputed class membership value.
      void set_class_member_indicator(int which_class) {
        imputed_class_membership_ = which_class;
      }

      // The number of possible classes the visitor could be in.
      int number_of_classes() const {
        return class_probabilities_->size();
      }

      virtual Int number_of_sites_visited() const = 0;
      virtual Int number_of_visits() const = 0;

     private:
      friend void intrusive_ptr_add_ref(VisitorBase *);
      friend void intrusive_ptr_release(VisitorBase *);
      
      std::string id_;

      // The Visitor belongs to one of K unknown classes. The class
      // probabilities and imputed class membership are set by posterior
      // sampling algorithms as part of an MCMC run.
      Ptr<VectorParams> class_probabilities_;
      int imputed_class_membership_;      
    };
    
    inline void intrusive_ptr_add_ref(FactorModels::VisitorBase *visitor) {
      visitor->up_count();
    }
  
    inline void intrusive_ptr_release(FactorModels::VisitorBase *visitor) {
      visitor->down_count();
      if (visitor->ref_count() == 0) {
        delete visitor;
      }
    }
  }  // namespace FactorModels

}  // namespace BOOM

#endif  // BOOM_FACTOR_MODELS_VISITOR_HPP_
