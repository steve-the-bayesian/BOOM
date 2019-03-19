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

#include "Models/IRT/Subject.hpp"
#include <stdexcept>
#include "Models/Glm/Glm.hpp"
#include "Models/IRT/Item.hpp"
#include "Models/PosteriorSamplers/PosteriorSampler.hpp"
#include "cpputil/report_error.hpp"

namespace BOOM {
  namespace IRT {

    Subject::Subject(const std::string &Id, uint nsub)
        : id_(Id),
          responses_(),
          search_helper(new NullItem),
          Theta_(new VectorParams(nsub, 0.0)),
          x_(),
          prototype() {}

    Subject::Subject(const std::string &Id, const Vector &theta)
        : id_(Id),
          responses_(),
          search_helper(new NullItem),
          Theta_(new VectorParams(theta)),
          x_(),
          prototype() {}

    Subject::Subject(const std::string &Id, uint nsub, const Vector &bg)
        : id_(Id),
          responses_(),
          search_helper(new NullItem),
          Theta_(new VectorParams(nsub, 0.0)),
          x_(bg),
          prototype() {}

    Subject::Subject(const Subject &rhs)
        : Data(rhs),
          id_(rhs.id_),
          responses_(rhs.responses_),
          search_helper(new NullItem),
          Theta_(rhs.Theta_->clone()),
          x_(rhs.x_),
          prototype(rhs.prototype->clone()) {}

    Subject *Subject::clone() const { return new Subject(*this); }

    uint Subject::Nitems() const { return responses_.size(); }
    uint Subject::Nscales() const { return Theta().size(); }

    Response Subject::add_item(const Ptr<Item> &item, Response r) {
      responses_[item] = r;
      return r;
    }

    Response Subject::add_item(const Ptr<Item> &it, uint resp) {
      Response r = new OrdinalData(resp, it->possible_responses_);
      add_item(it, r);
      return r;
    }

    Response Subject::add_item(const Ptr<Item> &it, const std::string &resp) {
      Response r = new OrdinalData(resp, it->possible_responses_);
      add_item(it, r);
      return r;
    }

    std::ostream &Subject::display(std::ostream &out) const {
      out << id();
      if (!x_.empty()) out << x_;
      out << endl;
      return out;
    }

    std::ostream &Subject::display_responses(std::ostream &out) const {
      // display Subject_id \t Item_id \t response
      for (IrIterC it = responses_.begin(); it != responses_.end(); ++it) {
        Ptr<Item> item = it->first;
        Response r = it->second;
        out << this->id() << "\t" << item->id() << "\t";
        r->display(out) << endl;
      }
      return out;
    }

    Ptr<Item> Subject::find_item(const std::string &item_id, bool nag) const {
      search_helper->id_ = item_id;
      IrIterC it = responses_.lower_bound(search_helper);
      if (it == responses_.end() || it->first->id() != item_id) {
        if (nag) {
          ostringstream msg;
          msg << "item with id " << item_id
              << " not found in Subject::find_item";
          report_error(msg.str());
        }
        return Ptr<Item>();
      } else
        return it->first;
    }

    double Subject::loglike() const {
      double ans = 0;
      for (IrIterC it = responses_.begin(); it != responses_.end(); ++it) {
        Ptr<Item> I = it->first;
        Response resp = it->second;
        ans += I->response_prob(resp, Theta(), true);
      }
      return ans;
    }

    const std::string &Subject::id() const { return id_; }

    Ptr<VectorParams> Subject::Theta_prm() { return Theta_; }
    const Ptr<VectorParams> Subject::Theta_prm() const { return Theta_; }

    const Vector &Subject::Theta() const { return Theta_prm()->value(); }

    void Subject::set_Theta(const Vector &v) { Theta_prm()->set(v); }

    const ItemResponseMap &Subject::item_responses() const {
      return responses_;
    }

    Response Subject::response(const Ptr<Item> &item) const {
      IrIterC it = responses_.find(item);
      if (it == responses_.end())
        return Response();
      else
        return it->second;
    }

    SpdMatrix Subject::xtx() const {
      SpdMatrix ans(Nscales(), 0.0);
      Selector inc(Nscales() + 1, true);
      inc.drop(0);

      for (IrIterC it = responses_.begin(); it != responses_.end(); ++it) {
        Ptr<Item> item(it->first);
        Vector b = inc.select(item->beta());
        ans.add_outer(b);
      }
      return ans;
    }

    Response Subject::simulate_response(const Ptr<Item> &it) {
      return it->simulate_response(Theta());
    }

  }  // namespace IRT
}  // namespace BOOM
