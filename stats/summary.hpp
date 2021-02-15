#ifndef BOOM_STATS_VARIABLE_SUMMARY_HPP_
#define BOOM_STATS_VARIABLE_SUMMARY_HPP_

/*
  Copyright (C) 2005-2018 Steven L. Scott

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
#include "LinAlg/VectorView.hpp"
#include "cpputil/RefCounted.hpp"
#include "stats/FreqDist.hpp"
#include "stats/IQagent.hpp"

#include "stats/DataTable.hpp"

namespace BOOM {

  class DataTable;
  class NumericSummary;
  class CategoricalSummary;

  //===========================================================================
  // Base class for summarizing columns in a DataTable.
  class VariableSummary : private RefCounted {
   public:
    VariableSummary()
        : number_missing_(0),
          number_observed_(0),
          number_distinct_(0)
    {}

    virtual ~VariableSummary() {}

    int sample_size() const {return number_observed_ + number_missing_;}
    int number_missing() const {return number_missing_;}
    int number_observed() const {return number_observed_;}
    int number_distinct() const {return number_distinct_;}

    virtual VariableType type() const = 0;
    virtual std::ostream &print(std::ostream &out) const = 0;
    std::string to_string() const;

    virtual const NumericSummary &as_numeric() const;
    virtual const CategoricalSummary &as_categorical() const;

   protected:
    void set_number_missing(int n) {number_missing_ = n;}
    void set_number_observed(int n) {number_observed_ = n;}
    void set_number_distinct(int n) {number_distinct_ = n;}

   private:
    int number_missing_;
    int number_observed_;
    int number_distinct_;

    friend void intrusive_ptr_add_ref(VariableSummary *v);
    friend void intrusive_ptr_release(VariableSummary *v);
  };

  //===========================================================================
  // A summary describing the distribution of a single numeric variable.
  class NumericSummary : public VariableSummary {
   public:
    NumericSummary();
    explicit NumericSummary(
        const Vector &x,
        const Vector &probs = {.001, .01, .025, .05, .1, .25, .5, .75, .9,
              .975, .99, .999});
    void summarize(const Vector &x);
    double mean() const {return mean_;}
    double sd() const {return sd_;}
    double var() const {return sd_ * sd_;}
    double min() const {return min_;}
    double max() const {return max_;}
    double quantile(double prob) const {
      return empirical_distribution_.quantile(prob);
    }

    VariableType type() const override {return VariableType::numeric;}
    const NumericSummary &as_numeric() const override {return *this;}
    std::ostream &print(std::ostream &out) const override;

   private:
    double mean_;
    double sd_;
    double min_;
    double max_;

    IQagent empirical_distribution_;
  };

  //===========================================================================
  // A summary describing the distribution of a single categorical variable.
  class CategoricalSummary : public VariableSummary {
   public:
    CategoricalSummary();
    explicit CategoricalSummary(const CategoricalVariable &x);
    void summarize(const CategoricalVariable &x);
    void collapse(int max_levels);

    std::ostream &print(std::ostream &out) const override;

    VariableType type() const override {return VariableType::categorical;}
    const CategoricalSummary &as_categorical() const override {return *this;}

   private:
    FrequencyDistribution frequency_distribution_;
  };

  //===========================================================================
  inline std::ostream &operator<<(std::ostream &out, const NumericSummary &summary) {
    return summary.print(out);
  }

  std::vector<Ptr<VariableSummary>> summarize(const DataTable &table);

} // namespace BOOM

#endif //  BOOM_STATS_VARIABLE_SUMMARY_HPP_
