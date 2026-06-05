#ifndef BOOM_BANDITS_GENERIC_BANDIT_BASE_HPP_
#define BOOM_BANDITS_GENERIC_BANDIT_BASE_HPP_
/*
  Copyright (C) 2005-2026 Steven L. Scott

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

#include "Models/ParamTypes.hpp"
#include "Models/DataTypes.hpp"
#include "distributions/rng.hpp"

namespace BOOM {

  class GenericBanditBase : private RefCounted {
   public:
    virtual int number_of_arms() const = 0;

    friend void intrusive_ptr_add_ref(GenericBanditBase *b) {b->up_count();}
    friend void intrusive_ptr_release(GenericBanditBase *b) {
      b->down_count();
      if (b->ref_count() == 0) {
        delete b;
      }
    }

  };
  
}  // namespace BOOM

#endif  //  BOOM_BANDITS_GENERIC_BANDIT_BASE_HPP_
