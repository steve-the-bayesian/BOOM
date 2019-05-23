#ifndef BOOM_EXPONENTIAL_INCREMENT_MODEL_HPP_
#define BOOM_EXPONENTIAL_INCREMENT_MODEL_HPP_

/*
  Copyright (C) 2019 Steven L. Scott

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

#include "Models/ExponentialModel.hpp"
#include "Models/VectorModel.hpp"
#include "Models/Policies/CompositeParamPolicy.hpp"
#include "Models/Policies/IID_DataPolicy.hpp"
#include "Models/Policies/PriorPolicy.hpp"

namespace BOOM {

  // A model for a vector of increasing values.  The initial value of the vector
  // is an exponential random variable.  Successive increments also independent
  // exponential random variables.
  //
  // The density of vector x is \prod_i Exp(diff(x)_i, lambda_i), where Exp is
  // the exponential density.  
  class ExponentialIncrementModel
      : public VectorModel,
        public CompositeParamPolicy,
        public IID_DataPolicy<VectorData>,
        public PriorPolicy
  {
   public:
    explicit ExponentialIncrementModel(const Vector &increment_rates);
    
    explicit ExponentialIncrementModel(
        const std::vector<Ptr<ExponentialModel>> &increment_models);

    ExponentialIncrementModel(const ExponentialIncrementModel &rhs);
    ExponentialIncrementModel(ExponentialIncrementModel &&rhs);

    ExponentialIncrementModel &operator=(const ExponentialIncrementModel &rhs);
    ExponentialIncrementModel &operator=(ExponentialIncrementModel &&rhs);
    
    ExponentialIncrementModel * clone() const override;

    void add_increment_model(const Ptr<ExponentialModel> &increment_model);

    //--------------------------------------------------------------------------
    // Overrides from VectorModel.
    //--------------------------------------------------------------------------
    double logp(const Vector &x) const override;

    Vector sim(RNG &rng = GlobalRng::rng) const override;

   private:
    std::vector<Ptr<ExponentialModel>> models_;
  };
  
}  // namespace BOOM 




#endif  // BOOM_EXPONENTIAL_INCREMENT_MODEL_HPP_
