#ifndef BOOM_FACTOR_MODELS_SITE_BASE_HPP_
#define BOOM_FACTOR_MODELS_SITE_BASE_HPP_
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

#include <string>
#include "uint.hpp"
#include "cpputil/RefCounted.hpp"

namespace BOOM {
  namespace FactorModels {
    
    // A base class containing the common characteristics of Sites.  
    class SiteBase : private RefCounted {
     public:
      // Args:
      //   id:  The unique ID for this visitor.
      SiteBase(const std::string &id)
          : id_(id)
      {}

      virtual Int number_of_visitors() const = 0;
      virtual Int number_of_visits() const = 0;
      
      virtual ~SiteBase() {}
      
      const std::string & id() const {return id_;}
      
     private:
      friend void intrusive_ptr_add_ref(SiteBase *);
      friend void intrusive_ptr_release(SiteBase *);
      std::string id_;
    };

    inline void intrusive_ptr_add_ref(FactorModels::SiteBase *site) {
      site->up_count();
    }
  
    inline void intrusive_ptr_release(FactorModels::SiteBase *site) {
      site->down_count();
      if (site->ref_count() == 0) {
        delete site;
      }
    }
    
  }  // namespace FactorModels
  
  
}  // namespace BOOM

#endif  // BOOM_FACTOR_MODELS_SITE_BASE_HPP_
