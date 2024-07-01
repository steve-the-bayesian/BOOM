#ifndef BOOM_MULTINOMIAL_FACTOR_MODEL_HPP_
#define BOOM_MULTINOMIAL_FACTOR_MODEL_HPP_
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

#include "Models/Policies/ManyParamPolicy.hpp"
#include "Models/Policies/PriorPolicy.hpp"
#include "Models/FactorModels/VisitorBase.hpp"
#include "Models/FactorModels/SiteBase.hpp"
#include "Models/FactorModels/PoissonFactorModel.hpp"

namespace BOOM {

  using MultinomialFactorData = PoissonFactorData;

  namespace FactorModels {
    class MultinomialSite;

    //==========================================================================
    class MultinomialVisitor : public VisitorBase {
     public:
      // Args:
      //   id:  The unique ID for this visitor.
      //   num_classes: The number of latent classes to which this visitor might
      //     belong.
      MultinomialVisitor(const std::string &id, int num_classes)
          : VisitorBase(id, num_classes)
      {}

      // Record one or more visits by this visitor to a given site.
      // Args:
      //   site:  The site that was visited.
      //   ntimes:  The number of visits that occurred.
      //
      // The total number of visits to a site is the important statistic.  So
      // calling this->visit(B, 2) and then this->visit(B, 3) is equivalent to a
      // single call this->visit(B, 5).
      void visit(const Ptr<MultinomialSite> &site, int ntimes);

      // This visitor's record of the sites that were visited (and the number of
      // times visited).
      const std::map<Ptr<MultinomialSite>, int, IdLess<MultinomialSite>> &
      sites_visited() const {
        return sites_visited_;
      }

      Int number_of_sites_visited() const override {
        return sites_visited_.size();
      }

      Int number_of_visits() const override;

      // Clear the record of the number of sites visited.
      void clear() {
        sites_visited_.clear();
      }

      // Absorb the data from rhs into *this.
      void merge(const MultinomialVisitor &rhs);

     private:
      // The map is keyed by a raw pointer to the visited site.  The value is a
      // count of the number of times the site was visited by this Visitor.
      std::map<Ptr<MultinomialSite>, int, IdLess<MultinomialSite>> sites_visited_;
    };

    //==========================================================================
    class MultinomialSite : public SiteBase {
     public:

      MultinomialSite(const std::string &id, int num_classes);

      // The number of latent categories.
      int number_of_classes() const {return logprob_.size();}

      // Record one or more visits by the given visitor.
      void observe_visitor(const Ptr<MultinomialVisitor> &visitor, int ntimes);

      // The probability that this site would be visited by a random visitor in
      // each class with one 'visit' to spend.
      const Vector visit_probs() const {return visit_probs_->value();}
      const Vector &logprob() const {return logprob_;}
      const Vector &logprob_complement() const {return logprob_complement_;}
      void set_probs(const Vector &probs);

      Int number_of_visitors() const override {
        return observed_visitors_.size();
      }

      Int number_of_visits() const override;

      const std::map<Ptr<MultinomialVisitor>, int, IdLess<MultinomialVisitor>> &
      observed_visitors() const {return observed_visitors_;}

      void clear() {
        observed_visitors_.clear();
      }

      // Absorb the data from rhs into *this.
      void merge(const MultinomialSite &rhs);

     private:
      // visit_probs_[k] is the probability that a user in class k, when
      // choosing a site to visit, would visit this site.  For a fixed value of
      // k visit_probs_[k] sums to 1 across all sites in the system.
      Ptr<VectorParams> visit_probs_;

      // the log of probs.
      Vector logprob_;

      // The log of 1 - probs.
      Vector logprob_complement_;

      // The number of visits from each observed visitor.
      std::map<Ptr<MultinomialVisitor>,
               int,
               IdLess<MultinomialVisitor>> observed_visitors_;

      // Fill the vectors logprob_ and logprob_complement_ from values of
      // visit_probs_.
      void refresh_probs();
    };
  }  // namespace FactorModels

  //===========================================================================
  class MultinomialFactorModel
      : public ManyParamPolicy,
        public PriorPolicy
  {
   public:
    using Site = FactorModels::MultinomialSite;
    using Visitor = FactorModels::MultinomialVisitor;

    // Args:
    //   num_classes:  The number of latent categories a user can be in.
    //   default_site_name: The name to use when an unfamilar site is
    //     encountered.
    explicit MultinomialFactorModel(
        int num_classes,
        const std::string &default_site_name = "Other");
    MultinomialFactorModel(const MultinomialFactorModel &rhs);
    MultinomialFactorModel(MultinomialFactorModel &&rhs) = default;
    MultinomialFactorModel &operator=(const MultinomialFactorModel &rhs);
    MultinomialFactorModel &operator=(MultinomialFactorModel &&rhs) = default;

    MultinomialFactorModel *clone() const override;

    void set_default_site_name(const std::string &default_site_name) {
      default_site_name_ = default_site_name;
    }
    const std::string &default_site_name() const {
      return default_site_name_;
    }

    // Make the model aware of a specific site, without necessarily requiring
    // visits to the site.
    void add_site(const Ptr<Site> &site);

    // Record one or more visits by a visitor to a single site.  If the model
    // already manages of visitor (or site) with the given id's then those
    // objects are adjusted by recording the visit.  If either visitor_id or
    // site_id is previously unseen, then a new Visitor or Site object is
    // created and added to the model's data.
    void record_visit(const std::string &visitor_id,
                      const std::string &site_id,
                      int ntimes = 1);

    // Generic interface for adding data to the model.  This method is here to
    // satisfy an expected BOOM idiom.  While it may prove useful,
    // 'record_visit' is the clearer path to adding data.
    void add_data(const Ptr<Data> &data_point) override;

    // Remove all visitors and sites (and visits to those sites) managed by the model.
    void clear_data() override;

    // Absorb the data from a second PoissonFactorModel into this model.
    void combine_data(const Model &other_model, bool just_suf = true) override;
    void combine_data_mt(const MultinomialFactorModel &rhs);

    // The number of latent classes being modeled.
    int number_of_classes() const {return num_classes_;}

    // The number of visitors this model has seen.
    int number_of_visitors() const {return visitors_.size();}

    // The number of sites this model has seen.
    int number_of_sites() const {return sites_.size();}

    // A directory of Sites, indexed by site id.
    std::map<std::string, Ptr<Site>> & sites() {return sites_;}
    const std::map<std::string, Ptr<Site>> & sites() const {return sites_;}

    // Args:
    //   site_id: The name of a site.
    //
    // Returns:
    //   The index of the site, defined as the "for loop step" in which the site
    //   is encountered when looping over sites.
    Int get_site_index(const std::string &site_id) const;

    // A mapping from site-id's to the "for loop position" of each site.
    std::map<std::string, Int> site_index_map() const;

    // A directory of Visitors, indexed by visitor id.
    std::map<std::string, Ptr<Visitor>> & visitors() {return visitors_;}
    const std::map<std::string, Ptr<Visitor>> & visitors() const {return visitors_;}

    // If the supplied id is recognized, return the Site with that ID.
    // Otherwise return nullptr.
    Ptr<Site> site(const std::string &id) const;

    // If the supplied id is recognized, return the Visitor with that ID.
    // Otherwise return nullptr.
    Ptr<Visitor> visitor(const std::string &id) const;

    // All three arguments are output arguments.  On output they will all be the
    // same size.
    //
    // Args:
    //   On output:
    //     count_output[i] is the number of times user user_ids_output[i]
    //     visited site site_ids_output[i].
    void extract_data(std::vector<std::string> &user_ids_output,
                      std::vector<std::string> &site_ids_output,
                      std::vector<int> &count_output) const;

   private:
    int num_classes_;

    // Sites and visitors stored in order of their ID's.
    std::map<std::string, Ptr<Visitor>> visitors_;
    std::map<std::string, Ptr<Site>> sites_;

    std::string default_site_name_;
  };
}  // namespace BOOM



#endif  // BOOM_MULTINOMIAL_FACTOR_MODEL_HPP_
