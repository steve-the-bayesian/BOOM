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

#include <vector>
#include <map>
#include <set>

/*
  The PoissonFactorModel describes a matrix n_ij, containing the number of
  visits by visitor i to site j.  There are K categories of visitors, with
  visitor category being a latent variable.

  Each site j has a site-specific set of parameters lambda_jk.  Given that user
  i is in latent class k, the n_ij visits on site j follow n_ij ~
  Poisson(lambda_jk E_i), where E_i is the exposure time for user i.

 */


namespace BOOM {

  // Records a number of visits to a site.
  class PoissonFactorData
      : public Data {
   public:
    PoissonFactorData(int64_t visitor_id,
                      int64_t site_id,
                      int nvisits)
        : visitor_id_(visitor_id),
          site_id_(site_id),
          nvisits_(nvisits)
    {}

    PoissonFactorData * clone() const override;
    std::ostream & display(std::ostream &out) const override;

    int64_t site_id() const {return site_id_;}
    int64_t visitor_id() const {return visitor_id_;}
    int nvisits() const {return nvisits_;}

   private:
    int64_t visitor_id_;
    int64_t site_id_;
    int nvisits_;
  };

  namespace PoissonFactor {
    class Site;
    class Visitor;
    void intrusive_ptr_add_ref(Site *site);
    void intrusive_ptr_add_ref(Visitor *visitor);
    void intrusive_ptr_release(Site *site);
    void intrusive_ptr_release(Visitor *visitor);

    //----------------------------------------------------------------------
    // A visitor belongs to one of K classes.
    class Visitor
        : public RefCounted
    {
     public:
      Visitor(int64_t id, int num_classes)
          : id_(id),
            class_probabilities_(new VectorParams(
                Vector(num_classes, 1.0 / num_classes))),
            imputed_class_membership_(-1)
      {}

      int64_t id() const {return id_;}

      void visit(const Ptr<Site> &site, int ntimes);

      void set_class_probabilities(const Vector &probs) {
        if (!class_probabilities_) {
          class_probabilities_.reset(new VectorParams(probs));
        } else {
          class_probabilities_->set(probs);
        }
      }

      const Vector &class_probabilities() const {
        return class_probabilities_->value();}

      int imputed_class_membership() const {
        return imputed_class_membership_;
      }

      void set_class_member_indicator(int which_class) {
        imputed_class_membership_ = which_class;
      }

      const std::map<Ptr<Site>, int> & sites_visited() const {
        return sites_visited_;
      }

     private:
      int64_t id_;

      // The Visitor belongs to one of K unknown classes. The class
      // probabilities and imputed class membership are set by posterior
      // sampling algorithms as part of an MCMC run.
      Ptr<VectorParams> class_probabilities_;
      int imputed_class_membership_;

      // The map is keyed by a raw pointer to the visited site.  The value is a
      // count of the number of times the site was visited by this Visitor.
      std::map<Ptr<Site>, int> sites_visited_;
    };

    //----------------------------------------------------------------------
    class Site : public RefCounted {
     public:
      Site(int64_t id, int num_classes);

      // Record one or more visits by
      void observe_visitor(const Ptr<Visitor> &visitor, int ntimes);

      int64_t id() const {return id_;}

      // The vector of
      const Vector &lambda() const {return visitation_rates_->value();}
      Vector log_lambda() const {return log_lambda_;}
      void set_lambda(const Vector &lambda);

      const Vector &prior_a() const {return prior_a_->value();}
      const Vector &prior_b() const {return prior_b_->value();}
      void set_prior(const Vector &prior_a, const Vector &prior_b);

      const std::map<Ptr<Visitor>, int> observed_visitors() const {
        return observed_visitors_;
      }

     private:
      int64_t id_;

      // Element k is the Poisson rate at which a visitor of class k visits the site.
      Ptr<VectorParams> visitation_rates_;
      Vector log_lambda_;

      // The number of times each visitor was observed.
      std::map<Ptr<Visitor>, int> observed_visitors_;

      Ptr<VectorParams> prior_a_;
      Ptr<VectorParams> prior_b_;
    };

  } // namespace PoissonFactor

  //===========================================================================
  class PoissonFactorModel
      : public ManyParamPolicy,
        public IID_DataPolicy<PoissonFactorData>,
        public PriorPolicy
  {
   public:
    using Site = PoissonFactor::Site;
    using Visitor = PoissonFactor::Visitor;

    PoissonFactorModel(int num_classes);

    PoissonFactorModel(const PoissonFactorModel &rhs);
    PoissonFactorModel(PoissonFactorModel &&rhs);
    PoissonFactorModel & operator=(const PoissonFactorModel &rhs);
    PoissonFactorModel & operator=(PoissonFactorModel &&rhs);

    PoissonFactorModel * clone() const override;

    void record_visit(int64_t visitor_id, int64_t site_id, int ntimes = 1);
    void add_data(const Ptr<PoissonFactorData> &data_point);

    // The number of latent classes being modeled.
    int number_of_classes() const {return sum_of_lambdas_.size();}

    std::vector<Ptr<Site>> & sites() {return sites_;}
    std::vector<Ptr<Visitor>> & visitors() {return visitors_;}

    // Return nullptr if the requested id is not available.
    Ptr<Site> get_site(int64_t id) const;
    Ptr<Visitor> get_visitor(int64_t id) const;

    const Vector &sum_of_lambdas() const;

    void set_sum_of_lambdas(const Vector &sum_of_lambdas) {
      sum_of_lambdas_ = sum_of_lambdas;
    }

   private:
    // Both visitors_ and sites_ are stored in the order of their ID's.
    std::vector<Ptr<Visitor>> visitors_;
    std::vector<Ptr<Site>> sites_;

    mutable Vector sum_of_lambdas_;
  };



}  // namespace BOOM

#endif //  BOOM_MODELS_FACTOR_MODELS_POISSON_FACTOR_MODEL_HPP_
