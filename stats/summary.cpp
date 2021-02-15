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

#include <iomanip>
#include <sstream>

#include "stats/summary.hpp"

#include "stats/ECDF.hpp"
#include "stats/moments.hpp"
#include "stats/DataTable.hpp"
#include "stats/FreqDist.hpp"

#include "cpputil/report_error.hpp"

namespace BOOM {

  NumericSummary empty_numeric_summary;
  CategoricalSummary empty_categorical_summary;

  void intrusive_ptr_add_ref(VariableSummary *summary) {
    summary->up_count();
  }

  void intrusive_ptr_release(VariableSummary *summary) {
    summary->down_count();
    if (summary->ref_count() == 0) {
      delete summary;
    }
  }

  std::string VariableSummary::to_string() const {
    std::ostringstream out;
    print(out);
    return out.str();
  }

  const NumericSummary & VariableSummary::as_numeric() const {
    report_error("Cannot coerce VariableSummary to numeric.");
    return empty_numeric_summary;
  }

  const CategoricalSummary & VariableSummary::as_categorical() const {
    report_error("Cannot coerce VariableSummary to categorical.");
    return empty_categorical_summary;
  }

  NumericSummary::NumericSummary()
      : mean_(0),
        sd_(0),
        min_(0),
        max_(0),
        empirical_distribution_(100000)
  {}

  NumericSummary::NumericSummary(const Vector &x, const Vector &probs)
      : mean_(0),
        sd_(0),
        min_(0),
        max_(0),
        empirical_distribution_(probs, 1000000)
  {
    summarize(x);
  }

  void NumericSummary::summarize(const Vector &data) {
    Vector non_nan = data;
    auto it = std::remove_if(
        non_nan.begin(), non_nan.end(), [](double x) {return std::isnan(x);});
    int new_size = it - non_nan.begin();
    if (new_size < non_nan.size()) {
      non_nan.resize(new_size);
    }

    set_number_observed(non_nan.size());
    set_number_missing(data.size() - number_observed());
    set_number_distinct(std::set<double>(non_nan.begin(), non_nan.end()).size());

    mean_ = BOOM::mean(non_nan);
    sd_ = BOOM::sd(non_nan);
    std::tie(min_, max_) = BOOM::range(non_nan);
    empirical_distribution_.add(non_nan);
    empirical_distribution_.update_cdf();
  }

  std::ostream &NumericSummary::print(std::ostream &out) const {
    using std::endl;
    auto precision = out.precision();
    out << "sample_size:     " << sample_size() << "\n"
        << "number observed  " << number_observed() << "\n"
        << "number missing   " << number_missing() << "\n"
        << "min:             " << std::setprecision(4) << min_ << endl
        << "lower quartile:  " << empirical_distribution_.quantile(.25) << endl
        << "median:          " << empirical_distribution_.quantile(.5) << endl
        << "mean:            " << mean_ << endl
        << "upper quartile:  " << empirical_distribution_.quantile(.75) << endl
        << "max:             " << max_ << endl;
    out << std::setprecision(precision);
    return out;
  }

  //===========================================================================
  CategoricalSummary::CategoricalSummary()
      : frequency_distribution_(std::vector<int>(0))
  {}

  CategoricalSummary::CategoricalSummary(const CategoricalVariable &x)
      : frequency_distribution_(std::vector<int>(0))
  {
    summarize(x);
  }

  void CategoricalSummary::summarize(const CategoricalVariable &x) {
    std::vector<int> category_codes;
    for (int i = 0; i < x.size(); ++i) {
      category_codes.push_back(x[i]->value());
    }
    frequency_distribution_ = FrequencyDistribution(category_codes);
    frequency_distribution_.set_labels(x.labels());
  }

  ostream &CategoricalSummary::print(ostream &out) const {
    return frequency_distribution_.print(out);
  }

  //===========================================================================
  std::vector<Ptr<VariableSummary>> summarize(const DataTable &table) {
    std::vector<Ptr<VariableSummary>> ans;
    for (int i = 0; i < table.nvars(); ++i) {
      VariableType type = table.variable_type(i);
      if (type == VariableType::numeric) {
        ans.push_back(new NumericSummary(table.getvar(i)));
      } else if (type == VariableType::categorical) {
        ans.push_back(new CategoricalSummary(table.get_nominal(i)));
      }
    }
    return ans;
  }


} // namespace BOOM
