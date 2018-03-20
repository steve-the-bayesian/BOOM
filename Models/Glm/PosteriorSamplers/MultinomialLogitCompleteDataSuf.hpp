// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2014 Steven L. Scott

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

#ifndef BOOM_MULTINOMIAL_LOGIT_COMPLETE_DATA_SUF_HPP_
#define BOOM_MULTINOMIAL_LOGIT_COMPLETE_DATA_SUF_HPP_

#include "LinAlg/SpdMatrix.hpp"
#include "Models/Glm/ChoiceData.hpp"
#include "cpputil/RefCounted.hpp"

namespace BOOM {
  namespace MultinomialLogit {

    class CompleteDataSufficientStatistics : private RefCounted {
     public:
      explicit CompleteDataSufficientStatistics(uint dim);
      CompleteDataSufficientStatistics *clone() const;

      void clear();
      void update(const ChoiceData &dp, const Vector &wgts, const Vector &u);
      void combine(const CompleteDataSufficientStatistics &rhs);

      const SpdMatrix &xtwx() const;
      const Vector &xtwu() const;
      double weighted_sum_of_squares() const;

     private:
      mutable SpdMatrix xtwx_;
      Vector xtwu_;
      mutable bool sym_;
      double weighted_sum_of_squares_;

      friend void intrusive_ptr_add_ref(CompleteDataSufficientStatistics *w) {
        w->up_count();
      }
      friend void intrusive_ptr_release(CompleteDataSufficientStatistics *w) {
        w->down_count();
        if (w->ref_count() == 0) delete w;
      }
    };

  }  // namespace MultinomialLogit
}  // namespace BOOM

#endif  // BOOM_MULTINOMIAL_LOGIT_COMPLETE_DATA_SUF_HPP_
