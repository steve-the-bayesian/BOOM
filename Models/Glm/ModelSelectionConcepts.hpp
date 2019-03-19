// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2007 Steven L. Scott

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
#ifndef BOOM_MODEL_SELECTION_CONCEPTS_HPP
#define BOOM_MODEL_SELECTION_CONCEPTS_HPP

#include "LinAlg/Selector.hpp"
#include "Models/BinomialModel.hpp"

namespace BOOM {
  class GlmCoefs;
  class StructuredVariableSelectionPrior;

  namespace ModelSelection {
    class Variable : private RefCounted {
     public:
      Variable(uint pos, double prob, const std::string &name = "");
      Variable(const Variable &rhs);
      virtual Variable *clone() const = 0;
      ~Variable() override;

      virtual std::ostream &print(std::ostream &) const;
      virtual double logp(const Selector &inc) const;

      // Put inc is in a valid state (where logp > -infinity).
      virtual void make_valid(Selector &inc) const = 0;

      void set_prob(double prob);
      uint pos() const;
      double prob() const;
      Ptr<BinomialModel> model();
      const Ptr<BinomialModel> model() const;
      virtual bool parents_are_present(const Selector &g) const = 0;
      const std::string &name() const;

      virtual void add_to(StructuredVariableSelectionPrior &) const = 0;

      friend void intrusive_ptr_add_ref(Variable *v) { v->up_count(); }
      friend void intrusive_ptr_release(Variable *v) {
        v->down_count();
        if (v->ref_count() == 0) delete v;
      }

     private:
      uint pos_;
      Ptr<BinomialModel> mod_;
      std::string name_;
    };
    std::ostream &operator<<(std::ostream &out, const Variable &v);
    //______________________________________________________________________
    class MainEffect : public Variable {
     public:
      MainEffect(uint position, double prob, const std::string &name = "");
      MainEffect *clone() const override;
      virtual bool observed() const;
      bool parents_are_present(const Selector &g) const override;
      void add_to(StructuredVariableSelectionPrior &) const override;
      void make_valid(Selector &inc) const override;
    };
    //______________________________________________________________________
    class MissingMainEffect : public MainEffect {
     public:
      MissingMainEffect(uint position, double prob, uint obs_ind_pos,
                        const std::string &name);
      MissingMainEffect(const MissingMainEffect &rhs);
      MissingMainEffect *clone() const override;
      double logp(const Selector &inc) const override;
      void make_valid(Selector &inc) const override;
      bool observed() const override;
      bool parents_are_present(const Selector &g) const override;
      void add_to(StructuredVariableSelectionPrior &) const override;

     private:
      uint obs_ind_pos_;
    };
    //______________________________________________________________________
    class Interaction : public Variable {
     public:
      Interaction(uint position, double prob, const std::vector<uint> &parents,
                  const std::string &name = "");
      Interaction(const Interaction &rhs);
      Interaction *clone() const override;
      double logp(const Selector &inc) const override;
      void make_valid(Selector &inc) const override;
      uint nparents() const;
      bool parents_are_present(const Selector &g) const override;
      void add_to(StructuredVariableSelectionPrior &) const override;

     private:
      std::vector<uint> parent_pos_;
    };
  }  // namespace ModelSelection
}  // namespace BOOM

#endif  // BOOM_MODEL_SELECTION_CONCEPTS_HPP
