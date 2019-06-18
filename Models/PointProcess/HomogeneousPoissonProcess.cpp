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
#include "Models/PointProcess/HomogeneousPoissonProcess.hpp"
#include "Models/SufstatAbstractCombineImpl.hpp"
#include "cpputil/report_error.hpp"
#include "distributions.hpp"

namespace BOOM {

  PoissonProcessSuf::PoissonProcessSuf(int count, double exposure)
      : number_of_events_(count), exposure_time_(exposure) {
    if (count < 0) {
      report_error("PoissonProcessSuf initialized with negative count.");
    }

    if (exposure < 0) {
      report_error("PoissonProcessSuf initialized with negative exposure.");
    }
  }

  PoissonProcessSuf *PoissonProcessSuf::clone() const {
    return new PoissonProcessSuf(*this);
  }

  int PoissonProcessSuf::count() const { return number_of_events_; }
  double PoissonProcessSuf::exposure() const { return exposure_time_; }

  void PoissonProcessSuf::clear() {
    number_of_events_ = 0;
    exposure_time_ = 0;
  }

  void PoissonProcessSuf::Update(const PointProcess &process) {
    number_of_events_ += process.number_of_events();
    exposure_time_ += process.window_duration();
  }

  void PoissonProcessSuf::update_raw(int incremental_events,
                                     double incremental_duration) {
    number_of_events_ += incremental_events;
    exposure_time_ += incremental_duration;
  }

  void PoissonProcessSuf::update_raw(const PointProcess &data) {
    number_of_events_ += data.number_of_events();
    exposure_time_ += data.window_duration();
  }

  PoissonProcessSuf *PoissonProcessSuf::combine(
      const Ptr<PoissonProcessSuf> &rhs) {
    return this->combine(*rhs);
  }

  PoissonProcessSuf *PoissonProcessSuf::combine(const PoissonProcessSuf &rhs) {
    number_of_events_ += rhs.number_of_events_;
    exposure_time_ += rhs.exposure_time_;
    return this;
  }

  PoissonProcessSuf *PoissonProcessSuf::abstract_combine(Sufstat *rhs) {
    return abstract_combine_impl(this, rhs);
  }

  Vector PoissonProcessSuf::vectorize(bool) const {
    Vector ans(2);
    ans[0] = number_of_events_;
    ans[1] = exposure_time_;
    return ans;
  }

  Vector::const_iterator PoissonProcessSuf::unvectorize(
      Vector::const_iterator &v, bool) {
    number_of_events_ = lround(*v);
    ++v;
    exposure_time_ = *v;
    return ++v;
  }

  Vector::const_iterator PoissonProcessSuf::unvectorize(const Vector &v,
                                                        bool minimal) {
    Vector::const_iterator b = v.begin();
    return this->unvectorize(b, minimal);
  }

  std::ostream &PoissonProcessSuf::print(std::ostream &out) const {
    out << number_of_events_ << " " << exposure_time_;
    return out;
  }

  //======================================================================

  HomogeneousPoissonProcess::HomogeneousPoissonProcess(double lambda)
      : ParamPolicy(new UnivParams(lambda)),
        DataPolicy(new PoissonProcessSuf) {}

  HomogeneousPoissonProcess::HomogeneousPoissonProcess(
      const std::vector<DateTime> &timestamps)
      : ParamPolicy(new UnivParams(1.0)), DataPolicy(new PoissonProcessSuf) {
    NEW(PointProcess, data)(timestamps);
    add_data(data);
    mle();
  }

  HomogeneousPoissonProcess *HomogeneousPoissonProcess::clone() const {
    return new HomogeneousPoissonProcess(*this);
  }

  double HomogeneousPoissonProcess::lambda() const {
    return ParamPolicy::prm()->value();
  }

  void HomogeneousPoissonProcess::set_lambda(double lambda) {
    ParamPolicy::prm()->set(lambda);
  }

  double HomogeneousPoissonProcess::event_rate(const DateTime &) const {
    return lambda();
  }

  double HomogeneousPoissonProcess::expected_number_of_events(
      const DateTime &then, const DateTime &now) const {
    return lambda() * (now - then);
  }

  void HomogeneousPoissonProcess::add_data_raw(int incremental_events,
                                               double incremental_duration) {
    suf()->update_raw(incremental_events, incremental_duration);
  }

  void HomogeneousPoissonProcess::add_exposure_window(const DateTime &t0,
                                                      const DateTime &t1) {
    suf()->update_raw(0, t1 - t0);
  }

  void HomogeneousPoissonProcess::add_event(const DateTime &t) {
    suf()->update_raw(1, 0);
  }

  void HomogeneousPoissonProcess::add_data_raw(const PointProcess &data) {
    suf()->update_raw(data);
  }

  double HomogeneousPoissonProcess::loglike(
      const Vector &scalar_lambda_vector) const {
    int x = suf()->count();
    double lam = scalar_lambda_vector[0] * suf()->exposure();
    return dpois(x, lam, true);
  }

  void HomogeneousPoissonProcess::mle() {
    double lambda_hat = suf()->count() / suf()->exposure();
    set_lambda(lambda_hat);
  }

  PointProcess HomogeneousPoissonProcess::simulate(
      RNG &rng, const DateTime &t0, const DateTime &t1,
      std::function<Data *()> mark_generator) const {
    PointProcess ans(t0, t1);
    int number_of_events = rpois_mt(rng, expected_number_of_events(t0, t1));
    double duration = t1 - t0;
    std::vector<double> points(number_of_events);
    for (int i = 0; i < number_of_events; ++i) {
      points[i] = runif_mt(rng, 0, duration);
    }
    std::sort(points.begin(), points.end());
    for (int i = 0; i < number_of_events; ++i) {
      Data *mark = mark_generator();
      if (mark) {
        ans.add_event(t0 + points[i], Ptr<Data>(mark));
      } else {
        ans.add_event(t0 + points[i]);
      }
    }
    return ans;
  }

}  // namespace BOOM
