#ifndef BOOM_MODELS_FACTOR_MODELS_POISSON_FACTOR_MODEL_HPP_
#define BOOM_MODELS_FACTOR_MODELS_POISSON_FACTOR_MODEL_HPP_

/*
  Copyright (C) 2005-2023 Steven L. Scott

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
#include "Models/Policies/PriorPolicy.hpp"

#include <vector>
#include <map>

/*
  The PoissonFactorModel describes a sparse matrix n_ij, containing the number
  of visits by visitor i to site j.  There are K categories of visitors, with
  visitor category being a latent variable.

  User i has a marginal probability theta[k] of belonging to class k.

  Given

  Each site j has a site-specific set of parameters lambda_jk.  Given that user
  i is in latent class k, the n_ij visits on site j follow n_ij ~
  Poisson(lambda_jk E_i), where E_i is the exposure time for user i.

 */


namespace BOOM {

  // Records a number of visits to a site.
  class PoissonFactorData
      : public Data {
   public:
    PoissonFactorData(const std::string &visitor_id,
                      const std::string &site_id,
                      int nvisits)
        : visitor_id_(visitor_id),
          site_id_(site_id),
          nvisits_(nvisits)
    {}

    PoissonFactorData * clone() const override;
    std::ostream & display(std::ostream &out) const override;

    const std::string & site_id() const {return site_id_;}
    const std::string & visitor_id() const {return visitor_id_;}
    int nvisits() const {return nvisits_;}

   private:
    std::string visitor_id_;
    std::string site_id_;
    int nvisits_;
  };

  namespace PoissonFactor {
    class Site;
    class Visitor;
    void intrusive_ptr_add_ref(Site *site);
    void intrusive_ptr_add_ref(Visitor *visitor);
    void intrusive_ptr_release(Site *site);
    void intrusive_ptr_release(Visitor *visitor);

    //-------------------------------------------------------------------------
    // A visitor belongs to one of K classes.
    class Visitor
        : public RefCounted
    {
     public:
      Visitor(const std::string &id, int num_classes)
          : id_(id),
            class_probabilities_(new VectorParams(
                Vector(num_classes, 1.0 / num_classes))),
            imputed_class_membership_(-1)
      {}

      const std::string & id() const {return id_;}

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

      const std::map<Ptr<Site>, int> &sites_visited() const {
        return sites_visited_;
      }

      void clear() {
        sites_visited_.clear();
      }

      int number_of_classes() const {
        return class_probabilities_->size();
      }

     private:
      std::string id_;

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
    // Profile for a visited site (e.g. a URL or a domain).
    class Site : public RefCounted {
     public:

      // Args:
      //   id:  The unique identifier for this site.  The "name".
      //   num_classes:  The number of distinct values in the latent factor.
      Site(const std::string &id, int num_classes);

      int number_of_classes() const {return visitation_rates_->size();}

      // Record one or more visits by the given visitor.
      void observe_visitor(const Ptr<Visitor> &visitor, int ntimes);

      const std::string & id() const {return id_;}

      // The vector of rates
      const Vector &lambda() const {return visitation_rates_->value();}
      Vector log_lambda() const {return log_lambda_;}
      void set_lambda(const Vector &lambda);

      // The prior distribution for the site's rate parameters, broken out by
      // the discrete latent factor.
      //
      // a/b is the prior guess at the poisson rate.  a/b^2 is the variance.
      const Vector &prior_a() const {return prior_a_->value();}
      const Vector &prior_b() const {return prior_b_->value();}
      void set_prior(const Vector &prior_a, const Vector &prior_b);

      const std::map<Ptr<Visitor>, int> &observed_visitors() const {
        return observed_visitors_;
      }

      // Returns a 2-column matrix.  Rows correspond to different levels of the
      // latent category.  The columns are the number of visits and the number
      // of visitor exposures to that category.
      Matrix visitor_counts() const;

      void clear() {
        observed_visitors_.clear();
      }

     private:
      std::string id_;

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
        public PriorPolicy
  {
   public:
    using Site = PoissonFactor::Site;
    using Visitor = PoissonFactor::Visitor;

    explicit PoissonFactorModel(int num_classes);

    PoissonFactorModel(const PoissonFactorModel &rhs);
    PoissonFactorModel(PoissonFactorModel &&rhs);
    PoissonFactorModel & operator=(const PoissonFactorModel &rhs);
    PoissonFactorModel & operator=(PoissonFactorModel &&rhs);

    PoissonFactorModel * clone() const override;

    void record_visit(const std::string &visitor_id,
                      const std::string &site_id,
                      int ntimes = 1);
    void add_data(const Ptr<Data> &data_point) override;
    void clear_data() override;
    void combine_data(const Model &other_model, bool just_suf = true) override;

    // The number of latent classes being modeled.
    int number_of_classes() const {return num_classes_;}
    int number_of_visitors() const {return visitors_.size();}
    int number_of_sites() const {return sites_.size();}

    std::map<std::string, Ptr<Site>> & sites() {return sites_;}
    const std::map<std::string, Ptr<Site>> & sites() const {return sites_;}
    std::map<std::string, Ptr<Visitor>> & visitors() {return visitors_;}
    const std::map<std::string, Ptr<Visitor>> & visitors() const {return visitors_;}

    // Return nullptr if the requested id is not available.
    Ptr<Site> site(const std::string &id) const;
    Ptr<Visitor> visitor(const std::string &id) const;

    const Vector &sum_of_lambdas() const;

   private:
    // Both visitors_ and sites_ are stored in the order of their ID's.
    int num_classes_;
    std::map<std::string, Ptr<Visitor>> visitors_;
    std::map<std::string, Ptr<Site>> sites_;
  };



}  // namespace BOOM

#endif //  BOOM_MODELS_FACTOR_MODELS_POISSON_FACTOR_MODEL_HPP_
