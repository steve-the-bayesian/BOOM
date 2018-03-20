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

#ifndef BOOM_NELDER_MEAD_HPP_
#define BOOM_NELDER_MEAD_HPP_

#include "LinAlg/Vector.hpp"
#include "numopt.hpp"

namespace BOOM {
  class NelderMeadMinimizer {
   public:
    explicit NelderMeadMinimizer(const Target &f);

    // Find the minimum from the specified starting value.
    void minimize(const Vector &starting_value);

    // A vector of suggested step sizes used to initialize the algorithm.
    void set_stepsize(const Vector &stepsize);

    // How frequently should the algorithm check that convergence has
    // taken place?
    void set_convergence_check_frequency(int frequency);

    // Set the desired precision used to terminate the algorithm.  I'm
    // not entirely clear on what this means.  Search the
    // corresponding cpp file for the REQMIN argument to nelmin().
    void set_precision(double precision);

    // Set the maximum number of times the target function should be
    // evaluated before bailing out of the algorithm.
    void set_evaluation_limit(int number_of_evalutations);

    // The minimum value obtained by the Nelder Mead algorithm.
    double minimum() const;

    // The value of x that achieves the minimum.
    const Vector &minimizing_value() const;

    // Returns true if the requested level of precision was obtained
    // in less than the maximum number of function evalutations.
    bool success() const;

    // The error message produced, if any.
    std::string error_message() const;

    // The number of times the algorithm was restarted.
    int number_of_restarts() const;

    // The number of times the function was evaluated.
    int number_of_evaluations() const;

   private:
    Target f_;
    int n_;
    Vector starting_value_;
    Vector minimizing_value_;
    double minimum_;
    double precision_;

    Vector stepsize_;
    double default_step_size_;
    int frequency_of_convergence_checks_;
    int max_number_of_evaluations_;
    int number_of_evaluations_;
    int number_of_restarts_;
    int error_code_;
  };

  class NelderMeadMaximizer {
   public:
    explicit NelderMeadMaximizer(Target f);
    void maximize(const Vector &starting_value);

    void set_stepsize(const Vector &stepsize);
    void set_convergence_check_frequency(int frequency);
    void set_precision(double precision);
    void set_evaluation_limit(int number_of_evalutations);

    double minimum() const;
    const Vector &minimizing_value() const;
    bool success() const;
    std::string error_message() const;
    int number_of_restarts() const;
    int number_of_evaluations() const;

   private:
    Negate target_;
    NelderMeadMinimizer minimizer_;
  };

}  // namespace BOOM

#endif  // BOOM_NELDER_MEAD_HPP_
