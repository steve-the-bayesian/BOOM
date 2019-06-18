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

#ifndef IRT_SUBJECT_HPP
#define IRT_SUBJECT_HPP

#include "Models/IRT/IRT.hpp"
#include "Models/Policies/IID_DataPolicy.hpp"
#include "Models/Policies/ParamPolicy_1.hpp"
#include "Models/Policies/PriorPolicy.hpp"

namespace BOOM {
  namespace IRT {
    class Item;

    // 'Subject' means 'observational unit' (e.g. student) not
    // 'subject matter'

    class Subject : virtual public Data  // data is the sequence of responses
    {
     public:
      friend class IrtModel;

      Subject(const std::string &Id, uint nscal);
      Subject(const std::string &Id, const Vector &theta);
      Subject(const std::string &Id, uint nscal, const Vector &background_vars);
      Subject(const Subject &rhs);
      Subject *clone() const override;

      Response add_item(const Ptr<Item> &item, uint response);
      Response add_item(const Ptr<Item> &item, const std::string &response);
      Response add_item(const Ptr<Item> &item, Response r);

      // find this subject's response to an item

      const ItemResponseMap &item_responses() const;
      Response response(const Ptr<Item> &) const;
      Ptr<Item> find_item(const std::string &item_id, bool nag = false) const;

      Ptr<VectorParams> Theta_prm();
      const Ptr<VectorParams> Theta_prm() const;
      const Vector &Theta() const;
      void set_Theta(const Vector &v);

      std::ostream &display(std::ostream &) const override;
      std::ostream &display_responses(std::ostream &) const;

      uint Nitems() const;
      uint Nscales() const;

      virtual double loglike() const;
      const std::string &id() const;
      SpdMatrix xtx() const;
      // returns \sum_i \Beta_i \Beta_i^T for betas

      Response simulate_response(const Ptr<Item> &item);

     private:
      std::string id_;  // subject identifier
      ItemResponseMap responses_;
      Ptr<Item> search_helper;
      Ptr<VectorParams> Theta_;
      Vector x_;  // covariates
      Response prototype;
    };
    //----------------------------------------------------------------------

  }  // namespace IRT
}  // namespace BOOM
#endif  // IRT_SUBJECT_HPP
