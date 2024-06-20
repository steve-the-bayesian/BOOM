#ifndef BOOM_CPPUTIL_DEFAULT_MAP_HPP_
#define BOOM_CPPUTIL_DEFAULT_MAP_HPP_

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

#include <map>
#include <memory>
#include "cpputil/report_error.hpp"

namespace BOOM {

  // A wrapper for a const std::map that includes a default value.  When looking
  // up an element in the map, if the key is not found, the default key is used
  // instead.
  template<class KEY, class VALUE>
  class DefaultMap {
   public:
    using MapType = std::map<KEY, VALUE>;
    //     using std::map<KEY, VALUE>::iterator_type = iterator_type;
    //     using std::map<KEY, VALUE>::iterator_type = iterator_type;

    // Args:
    //   base_map:  The underlying map.
    //   default_key: The key to use by default when a requested key is not
    //     present in the base map.  An exception is thrown if the default key
    //     is missing from the base map.
    DefaultMap(MapType *base_map, const KEY &default_key)
        : base_map_(base_map),
          default_key_(default_key)
    {
      const auto it = base_map_->find(default_key_);
      if (it == base_map_->end()) {
        report_error("Error in DefaultMap constructor.  "
                     "Default key is not present in the base map.");
      }
      default_value_ = it->second;
    }

    // The number of elements in the base map.
    size_t size() const {return base_map_->size();}

    // Element access.
    const VALUE & operator[](const KEY &key) const {
      return this->at(key);
    }

    // Element access, alternate form.  If the requested key is not present, the
    // value corresponding to 'default_key' is returned.  This differs from
    // base_map->at(), which would generate an exception in that case.
    const VALUE & at(const KEY &key) const {
      const auto it = base_map_->find(key);
      if (it == base_map_->end()) {
        return default_value_;
      } else {
        return it->second;
      }
    }

   private:
    const std::map<KEY, VALUE> *base_map_;

    KEY default_key_;
    VALUE default_value_;
  };

}  // namespace BOOM


#endif  // BOOM_CPPUTIL_DEFAULT_MAP_HPP_
