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

#ifndef BOOM_MOVE_ACCOUNTING_HPP_
#define BOOM_MOVE_ACCOUNTING_HPP_

#include <ctime>
#include <map>
#include <string>
#include <vector>

#include "LinAlg/Matrix.hpp"

namespace BOOM {

  class MoveAccounting;
  
  // A MoveTimer class will record the amount of time between its
  // creation and its destruction.
  class MoveTimer {
   public:
    // Args:
    //   move_type:  The type of move that is being timed;
    //   accounting:  The object in which to record the lifetime;
    MoveTimer(const std::string &move_type, MoveAccounting *accounting);
    ~MoveTimer();
    void stop();

   private:
    const std::string move_type_;
    MoveAccounting *accounting_;
    clock_t time_;
    bool stopped_;
  };

  // A class to keep track of acceptances, rejections, and special cases
  // for Metropolis Hastings moves.
  class MoveAccounting {
   public:
    void record_acceptance(const std::string &move_type);
    void record_rejection(const std::string &move_type);
    void record_special(const std::string &move_type,
                        const std::string &special_case);

    // Rows in the matrix correspond to move types.  Column names
    // correspond to acceptances, failures, and special cases.  The
    // number of special cases must be computed.  If timings have been
    // kept, they will be stored in the first column.
    LabeledMatrix to_matrix() const;

    // Returns a vector of move type names corresponding to the row
    // names of to_matrix().
    std::vector<std::string> compute_move_types() const;
    std::vector<std::string> compute_outcome_type_names() const;

    // Returns the ratio of "accept" counts to "accept" + "reject"
    // counts for the specified move type.  If the total number of
    // counts is zero then zero is returned.  The total number of
    // counts is returned in the second argument.
    double acceptance_ratio(const std::string &move_type,
                            int &number_of_trials);

    // To time code, use eithr of the the following idioms:
    // When entering a code that is entirely devoted to an MCMC move:
    //    MoveTimer timer = this->start_time("MyMoveType");
    // Then the timer will record the time when it is destroyed.
    //
    // If other things are going on besides the move, then you can:
    // clock_t start_time = clock();
    //    .. do stuff ..
    // double time_in_seconds = this->stop_time("MyMoveType", start_time);
    MoveTimer start_time(const std::string &move_type);
    double stop_time(const std::string &move_type, clock_t start);

   private:
    // counts_ is essentially a matrix indexed by strings instead of
    // integers.  The "row" index is called a "move type".  It is
    // designed to keep track of different types of sampling
    // algorithms.  The "column" index is the "outcome type".
    // Standard outcome types are "accept", and "reject".  Other
    // outcome types can be passed in by the user through calls to
    // "record_special".
    //
    // Example: counts_["MH"]["accept"] contains the number of times a
    // move type "MH" was accepted.  counts_["MH"]["reject"] is the
    // number of times a move from "MH" was rejected.
    std::map<std::string, std::map<std::string, int> > counts_;

    // time_in_seconds_ stores the amount of time each move type has
    // consumed.  An entry for a move type (e.g. "MH") is created (if
    // it doesn't already exist) or incremented (if it does) by
    // calling start_time("MH").
    std::map<std::string, double> time_in_seconds_;
  };

}  // namespace BOOM
#endif  //  BOOM_MOVE_ACCOUNTING_HPP_
