/*
  Copyright (C) 2005 Steven L. Scott

  This library is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License as published by the Free Software Foundation; either
  version 2.1 of the License, or (at your option) any later version.

  This library is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA
*/
#ifndef IRT_ITEM_HPP
#define IRT_ITEM_HPP

#include "Models/IRT/IRT.hpp"
#include "Models/ModelTypes.hpp"
#include "Models/Policies/IID_DataPolicy.hpp"

namespace BOOM {
  class GlmCoefs;
  namespace IRT {

    class Item : public IID_DataPolicy<Subject> {
     public:
      friend class Subject;
      friend class IrtModel;

      typedef std::vector<std::string> StringVector;
      Item(const std::string &Id,
           uint Maxscore,
           uint one_subscale,
           uint nscales,
           const std::string &Name = "");
      Item(const std::string &Id, uint Maxscore,
           const std::vector<bool> &subscales,
           const std::string &Name = "");
      Item(const Item &rhs);

      Item *clone() const override = 0;

      uint nscales_this() const;  // number of subscales assessed by this item
      uint Nscales() const;       // total number of subscales
      const Indicators &subscales() const;
      uint maxscore() const;  // maximum score possible on the item
      uint nlevels() const;   // number of possible repsonses: maxscore+1
      bool assigned_to_subject(const Ptr<Subject> &s) const;

      //--- data managment over-rides for Model base class ---
      virtual void add_subject(const Ptr<Subject> &s);
      virtual void remove_subject(const Ptr<Subject> &s);
      void add_data(const Ptr<Data> &) override;
      void add_data(const Ptr<Subject> &) override;
      void clear_data() override;

      const SubjectSet &subjects() const;
      uint Nsubjects() const;

      const std::string &id() const;
      const std::string &name() const;

      // create/get responses
      Response make_response(const std::string &s) const;
      Response make_response(uint m) const;
      Response response(const Ptr<Subject> &);
      const Response response(const Ptr<Subject> &) const;

      void set_response_names(const StringVector &levels);
      const StringVector &possible_responses() const;

      void report(std::ostream &, uint namewidth = 0) const;
      Vector response_histogram() const;

      std::ostream &display(std::ostream &) const;
      virtual std::ostream &display_item_params(std::ostream &,
                                           bool decorate = true) const = 0;

      Response simulate_response(const Vector &Theta) const;

      virtual const Vector &beta() const = 0;

      virtual double pdf(const Ptr<Data> &, bool logsc) const;
      virtual double pdf(const Ptr<Subject> &, bool logsc) const;

      virtual double response_prob(Response r, const Vector &Theta,
                                   bool logscale) const = 0;
      virtual double response_prob(uint r, const Vector &Theta,
                                   bool logscale) const = 0;

      double loglike() const;

     private:
      Indicators subscales_;            // which subscales does this item assess
      std::string id_;                  // internal id, like "17"
      std::string name_;                // external id, like "Toy Story"
      Ptr<CatKey> possible_responses_;  // "0", "1"... "Poor","Fair","Good"...
      void increment_hist(const Ptr<Subject> &, Vector &) const;
      void increment_loglike(const Ptr<Subject> &) const;
      mutable double loglike_ans;
    };

    //======================================================================
    // A "NullItem" is used by Subjects and IrtModels to help them
    // navigate their ItemSets

    class NullItem : public Item {
     public:
      NullItem() : Item("Null", 1, 0, 1, "Null") {}
      NullItem *clone() const override { return new NullItem(*this); }
      std::ostream &display_item_params(std::ostream &out, bool = true) const override {
        return out;
      }
      const Vector &beta() const override { return b; }
      double response_prob(Response, const Vector &, bool) const override {
        return 0.0;
      }
      double response_prob(uint, const Vector &, bool) const override {
        return 0.0;
      }
      double pdf(const Ptr<Data> &, bool) const override { return 0.0; }
      double pdf(const Ptr<Subject> &, bool) const override { return 0.0; }
      std::vector<Ptr<Params>> parameter_vector() override {
        return std::vector<Ptr<Params>>();
      }
      const std::vector<Ptr<Params>> parameter_vector() const override {
        return std::vector<Ptr<Params>>();
      }
      void initialize_params() {}
      void add_data(const Ptr<Data> &) override {}
      void add_data(const Ptr<Subject> &) override {}
      void clear_data() override {}
      void sample_posterior() override {}
      double logpri() const override { return 0.0; }
      int number_of_sampling_methods() const override { return 0; }

     protected:
      PosteriorSampler *sampler(int i) override { return nullptr; }
      PosteriorSampler const *const sampler(int i) const override {
        return nullptr;
      }

     private:
      Vector b;
    };

  }  // namespace IRT
}  // namespace BOOM
#endif  // IRT_ITEM_HPP
