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

#ifndef BOOM_HOMOGENEOUS_POISSON_PROCESS_HPP_
#define BOOM_HOMOGENEOUS_POISSON_PROCESS_HPP_

#include <functional>
#include "Models/PointProcess/PointProcess.hpp"
#include "Models/PointProcess/PoissonProcess.hpp"
#include "Models/Policies/ParamPolicy_1.hpp"
#include "Models/Policies/PriorPolicy.hpp"
#include "Models/Policies/SufstatDataPolicy.hpp"
#include "cpputil/DateTime.hpp"

namespace BOOM {

  // A PoissonProcessSuf differs from a PoissonSuf (in
  // Models/PoissonModel.hpp) because
  class PoissonProcessSuf : public SufstatDetails<PointProcess> {
   public:
    explicit PoissonProcessSuf(int count = 0, double exposure = 0);
    PoissonProcessSuf *clone() const override;

    int count() const;
    double exposure() const;

    void Update(const PointProcess &process) override;
    void clear() override;
    void update_raw(int number_of_events, double duration);
    void update_raw(const PointProcess &data);

    PoissonProcessSuf *combine(const Ptr<PoissonProcessSuf> &rhs);
    PoissonProcessSuf *combine(const PoissonProcessSuf &rhs);
    PoissonProcessSuf *abstract_combine(Sufstat *rhs) override;

    // Vectorized sufficient stats have two entries: count and
    // exposure.
    Vector vectorize(bool minimal = true) const override;
    Vector::const_iterator unvectorize(Vector::const_iterator &v,
                                       bool minimal = true) override;
    Vector::const_iterator unvectorize(const Vector &v,
                                       bool minimal = true) override;

    std::ostream &print(std::ostream &out) const override;

   private:
    int number_of_events_;
    double exposure_time_;
  };
  //======================================================================
  class HomogeneousPoissonProcess
      : public PoissonProcess,
        public ParamPolicy_1<UnivParams>,
        public SufstatDataPolicy<PointProcess, PoissonProcessSuf>,
        public PriorPolicy,
        public LoglikeModel {
   public:
    explicit HomogeneousPoissonProcess(double lambda = 1.0);
    explicit HomogeneousPoissonProcess(const std::vector<DateTime> &timestamps);

    HomogeneousPoissonProcess *clone() const override;
    double lambda() const;
    void set_lambda(double lambda);
    Ptr<UnivParams> Lambda_prm() { return ParamPolicy::prm(); }
    const Ptr<UnivParams> Lambda_prm() const { return ParamPolicy::prm(); }

    double event_rate(const DateTime &t) const override;
    double expected_number_of_events(const DateTime &t0,
                                     const DateTime &t1) const override;

    // Updates sufficient statistics, but does not allocate a new
    // Ptr<PointProcess> data element.
    void add_data_raw(int incremental_events, double incremental_duration);
    void add_data_raw(const PointProcess &);
    void add_exposure_window(const DateTime &t0, const DateTime &t1) override;
    void add_event(const DateTime &t) override;

    double loglike(const Vector &scalar_lambda_vector) const override;
    void mle() override;
    PointProcess simulate(RNG &rng, const DateTime &t0, const DateTime &t1,
                          std::function<Data *()> mark_generator =
                              NullDataGenerator()) const override;
  };

}  // namespace BOOM
#endif  // BOOM_HOMOGENEOUS_POISSON_PROCESS_HPP_
