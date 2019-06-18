// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2012 Steven L. Scott

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

#ifndef BOOM_WEIGHTED_GAUSSIAN_SUF_HPP_
#define BOOM_WEIGHTED_GAUSSIAN_SUF_HPP_

#include "Models/DataTypes.hpp"
#include "Models/Sufstat.hpp"
#include "Models/WeightedData.hpp"

namespace BOOM {

  // WeightedGaussianSuf are the sufficient statistics for a Gaussian
  // model where y[i] ~ N(mu, sigsq / w[i]).
  class WeightedGaussianSuf : public SufstatDetails<WeightedDoubleData> {
   public:
    explicit WeightedGaussianSuf(double sum = 0, double sumsq = 0, double n = 0,
                                 double sumw = 0);
    WeightedGaussianSuf *clone() const override;

    void clear() override;
    void Update(const WeightedDoubleData &data) override;

    void update_raw(double data, double weight);
    void add_mixture_data(double y, double weight, double prob);

    WeightedGaussianSuf *abstract_combine(Sufstat *s) override;
    void combine(const Ptr<WeightedGaussianSuf> &);
    void combine(const WeightedGaussianSuf &);

    double ybar() const { return sum_ / sumw_; }
    double sum() const { return sum_; }
    double n() const { return n_; }
    double sumsq() const { return sumsq_; }
    double sumw() const { return sumw_; }

    Vector vectorize(bool minimal = true) const override;
    Vector::const_iterator unvectorize(Vector::const_iterator &v,
                                       bool minimal = true) override;
    Vector::const_iterator unvectorize(const Vector &v,
                                       bool minimal = true) override;
    std::ostream &print(std::ostream &) const override;

   private:
    double sum_;    // sum y[i] * w[i]
    double sumsq_;  // sum y[i]^2 * w[i]
    double n_;      // the actual count of the number of observations
    double sumw_;   // sum w[i]
  };

}  // namespace BOOM

#endif  //  BOOM_WEIGHTED_GAUSSIAN_SUF_HPP_
