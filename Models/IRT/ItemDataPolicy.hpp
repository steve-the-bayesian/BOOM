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

namespace BOOM {
  namespace IRT {

    template <class D>
    class ItemDataPolicy : public ItemModel {
     public:
      typedef D DataType;
      typedef ItemDataPolicy<D> DataPolicy;
      typedef std::vector<Ptr<DataType> > DatasetType;
      typedef Ptr<DatasetType, false> dsetPtr;

     private:
      mutable dsetPtr dat_;       // model owns pointer to its data.
      void refresh_data() const;  // this function
     public:
      ItemDataPolicy();
      ItemDataPolicy(const Ptr<Item> &it);
      ItemDataPolicy(const ItemDataPolicy &rhs);
      ItemDataPolicy *clone() const = 0;
      ItemDataPolicy &operator=(const ItemDataPolicy &rhs);

      void clear_data() {}                 // no-op
      void add_data(const Ptr<Data> &) {}  // no-op

      DatasetType &dat();
      const DatasetType &dat() const;
    };

    //------------------------------------------------
    template <class D>
    ItemDataPolicy<D>::ItemDataPolicy() {}

    template <class D>
    ItemDataPolicy<D>::ItemDataPolicy(const Ptr<Item> &Parent)
        : ItemModel(Parent) {}

    template <class D>
    ItemDataPolicy<D>::ItemDataPolicy(const ItemDataPolicy &rhs)
        : Model(rhs),
          ItemModel(rhs),
          dat_(new DatasetType(rhs.dat().size())) {
    }  // copy constructor makes way for

    template <class D>
    ItemDataPolicy<D> &ItemDataPolicy<D>::operator=(const ItemDataPolicy &rhs) {
      if (&rhs != this) dat_ = new DatasetType(rhs.dat().size());
      return *this;
    }

    template <class D>
    void ItemDataPolicy<D>::refresh_data() const {
      const SubjectSet &subjects(this->subjects());

      uint n = subjects.size();
      if (!dat_)
        dat_ = new DatasetType(n);
      else if (dat_->size() != n)
        dat_->resize(n);

      DatasetType &d(*dat_);
      uint i = 0;
      for (SubjectSet::iterator it = subjects.begin(); it != subjects.end();
           ++it, ++i) {
        Ptr<Subject> s = *it;
        Vector &Theta(s->Theta());
        Response r = response(s);
        if (!d[i])
          d[i] = new DataType(*r, Theta);
        else {
          d[i]->set_x(Theta);  // adds intercept
          d[i]->set_y(r);
        }
      }
    }

    template <class D>
    typename ItemDataPolicy<D>::DatasetType &ItemDataPolicy<D>::dat() {
      refresh_data();
      return *dat_;
    }

    template <class D>
    const typename ItemDataPolicy<D>::DatasetType &ItemDataPolicy<D>::dat()
        const {
      refresh_data();
      return *dat_;
    }

  }  // namespace IRT
}  // namespace BOOM
