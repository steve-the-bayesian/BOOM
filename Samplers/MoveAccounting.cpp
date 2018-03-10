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

#include "Samplers/MoveAccounting.hpp"
#include <set>

namespace BOOM {

  namespace {
    typedef std::map<std::string, std::map<std::string, int> >::const_iterator
        CountsIterator;
    typedef std::map<std::string, int>::const_iterator IntMapIterator;
    typedef std::map<std::string, double>::const_iterator TimeIterator;
  }  // namespace

  void MoveAccounting::record_acceptance(const std::string &move_type) {
    ++counts_[move_type]["accept"];
  }
  void MoveAccounting::record_rejection(const std::string &move_type) {
    ++counts_[move_type]["reject"];
  }
  void MoveAccounting::record_special(const std::string &move_type,
                                      const std::string &special_case) {
    ++counts_[move_type][special_case];
  }

  namespace {
    std::map<std::string, int> reverse_lookup(
        const std::vector<std::string> &names) {
      std::map<std::string, int> ans;
      for (int i = 0; i < names.size(); ++i) {
        ans[names[i]] = i;
      }
      return ans;
    }
  }  // namespace

  LabeledMatrix MoveAccounting::to_matrix() const {
    std::vector<std::string> move_types = compute_move_types();
    std::vector<std::string> outcome_types = compute_outcome_type_names();
    Matrix counts(move_types.size(), outcome_types.size());
    std::map<std::string, int> row_name_map = reverse_lookup(move_types);
    std::map<std::string, int> col_name_map = reverse_lookup(outcome_types);

    for (TimeIterator timing = time_in_seconds_.begin();
         timing != time_in_seconds_.end(); ++timing) {
      const std::string &move_type(timing->first);
      double seconds = timing->second;
      counts(row_name_map[move_type], col_name_map["seconds"]) = seconds;
    }

    for (CountsIterator move_type = counts_.begin(); move_type != counts_.end();
         ++move_type) {
      int row = row_name_map[move_type->first];
      for (IntMapIterator case_type = move_type->second.begin();
           case_type != move_type->second.end(); ++case_type) {
        int col = col_name_map[case_type->first];
        counts(row, col) = case_type->second;
      }
    }
    return LabeledMatrix(counts, move_types, outcome_types);
  }

  std::vector<std::string> MoveAccounting::compute_move_types() const {
    std::set<std::string> move_types;
    for (CountsIterator el = counts_.begin(); el != counts_.end(); ++el) {
      move_types.insert(el->first);
    }
    for (TimeIterator el = time_in_seconds_.begin();
         el != time_in_seconds_.end(); ++el) {
      move_types.insert(el->first);
    }
    return std::vector<std::string>(move_types.begin(), move_types.end());
  }

  std::vector<std::string> MoveAccounting::compute_outcome_type_names() const {
    std::set<std::string> names;
    if (!time_in_seconds_.empty()) {
      names.insert("seconds");
    }
    names.insert("accept");
    names.insert("reject");
    for (CountsIterator el = counts_.begin(); el != counts_.end(); ++el) {
      for (IntMapIterator m = el->second.begin(); m != el->second.end(); ++m) {
        names.insert(m->first);
      }
    }
    std::vector<std::string> ans;
    ans.reserve(names.size());
    if (!time_in_seconds_.empty()) {
      ans.push_back("seconds");
    }
    ans.push_back("accept");
    ans.push_back("reject");
    for (std::set<std::string>::const_iterator it = names.begin();
         it != names.end(); ++it) {
      const std::string &el(*it);
      if (el != "accept" && el != "reject" && el != "seconds") {
        ans.push_back(el);
      }
    }
    return ans;
  }

  double MoveAccounting::acceptance_ratio(const std::string &move_type,
                                          int &number_of_trials) {
    int accepts = counts_[move_type]["accept"];
    int rejects = counts_[move_type]["reject"];
    number_of_trials = accepts + rejects;
    double ans = accepts;
    if (number_of_trials > 0) {
      ans /= number_of_trials;
    }
    return ans;
  }

  MoveTimer MoveAccounting::start_time(const std::string &move_type) {
    return MoveTimer(move_type, this);
  }

  double MoveAccounting::stop_time(const std::string &move_type,
                                   clock_t start) {
    double dt = clock() - start;
    double seconds = dt / CLOCKS_PER_SEC;
    time_in_seconds_[move_type] += seconds;
    return seconds;
  }

  MoveTimer::MoveTimer(const std::string &move_type, MoveAccounting *accounting)
      : move_type_(move_type),
        accounting_(accounting),
        time_(clock()),
        stopped_(false) {}

  MoveTimer::~MoveTimer() { stop(); }

  void MoveTimer::stop() {
    if (!stopped_) {
      accounting_->stop_time(move_type_, time_);
      stopped_ = true;
    }
  }

}  // namespace BOOM
