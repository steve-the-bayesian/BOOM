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
#ifndef BOOM_IRT_HDR_HPP
#define BOOM_IRT_HDR_HPP

#include <map>
#include <set>
#include <vector>

#include "uint.hpp"
#include "Models/CategoricalData.hpp"
#include "Models/ModelTypes.hpp"

#include "LinAlg/Selector.hpp"

namespace BOOM {
  namespace IRT {
    typedef Selector Indicators;

    class Subject;
    class SubjectPrior;
    class IrtModel;
    class Item;

    struct SubjectLess {
      bool operator()(const Ptr<Subject> &s1, const Ptr<Subject> &s2) const;
    };

    struct ItemLess {
      bool operator()(const Ptr<Item> &i1, const Ptr<Item> &i2) const;
    };

    typedef Ptr<OrdinalData> Response;
    typedef std::vector<Ptr<Subject> > SubjectSet;
    typedef SubjectSet::iterator SI;
    typedef SubjectSet::const_iterator CSI;  // Miami :)

    void add_subject(SubjectSet &, const Ptr<Subject> &);

    typedef std::set<Ptr<Item>, ItemLess> ItemSet;
    typedef ItemSet::iterator ItemIt;
    typedef ItemSet::const_iterator ItemItC;
    typedef std::map<Ptr<Item>, Response, ItemLess> ItemResponseMap;
    typedef ItemResponseMap::iterator IrIter;
    typedef ItemResponseMap::const_iterator IrIterC;

  }  // namespace IRT
}  // namespace BOOM
#endif  // BOOM_IRT_HDR_HPP
