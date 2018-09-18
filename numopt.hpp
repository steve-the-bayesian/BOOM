// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005 Steven L. Scott

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

#ifndef BOOM_NUMOPT_HPP
#define BOOM_NUMOPT_HPP

#include <string>
#include "LinAlg/SpdMatrix.hpp"

#include "uint.hpp"
#include <functional>
#include "cpputil/report_error.hpp"

namespace BOOM{

  // Optimizers work in terms of arbitrary function objects.
  // TODO(user): Replace these with C++ function templates once
  // C++11 becomes more widely supported.
  typedef std::function<double(const Vector &) > Target;
  typedef std::function<double(const Vector &x,
                                 Vector &g) > dTarget;
  typedef std::function<double(const Vector &x,
                                 Vector &g,
                                 Matrix &H) > d2Target;
  typedef std::function<double(double) > ScalarTarget;

  // Different flavors of conjugate gradient.
  enum ConjugateGradientMethod{
    FletcherReeves,
    PolakRibiere,
    BealeSorenson};

  // Different derivative based optimization algorithms.
  //  * BFGS is a quasi-newton method that maintains an approximate
  //    Hessian.
  //  * Conjugate gradient uses a set of 1-dimensional optimizations
  //    along a sequence of directions that are "conjugate" to one
  //    another in the sense that they provide minimal interference
  //    in the optimization.
  //  * Both: Try conjugate gradient first, and then transition to
  //    BFGS.
  //
  // Conjugate gradient is more stable far from the mode, but requires
  // more function evaluations.  BFGS can be unstable far from the
  // mode, but is faster near the mode.
  enum OptimizationMethod {
    BFGS,
    ConjugateGradient,
    Both
  };

  // Optimize a function for which no derivative information is
  // available.
  // Args:
  //   x: On input x is the initial set of function arguments of the
  //      algorithm.  On output it is the maximizing value.
  //   target:  The function to be maximized.
  // Returns:
  //   The value of 'target' at the optimizing 'x'.
  double max_nd0(Vector &x, Target target);

//======================================================================
// TODO(user): maxnd1_careful is starting to have too many
// parameters.  Replace it with an object that hides some of the
// complexity.
//======================================================================

  // Optimize a function for which gradient information is available.
  // Args:
  //   x: On input x is the initial set of function arguments of the
  //      algorithm.  On output it is the maximizing value.
  //   function_value: Not used on input.  On output this is filled
  //     with the value of the function at the maximizing x.
  //   target:  The function to be maximized.
  //   dtarget:  The function to be maximized, with gradient.
  //   eps: The convergence criterion, which will be used as both a
  //     relative and absolute measure.
  //   method:  The method that should be used to optimize target.
  // Returns:
  //   Returns true on success, and false if the optimization fails.
  bool max_nd1_careful(Vector &x,
                       double &function_value,
                       Target target,
                       dTarget gradient,
                       std::string &error_message,
                       double eps = 1e-5,
                       int max_iterations = 500,
                       OptimizationMethod method = Both);

  // As with max_nd1_careful, but does not check for a successful
  // outcome.  Returns the value of the function at the optimizing x.
  double max_nd1(Vector &x,
                 Target target,
                 dTarget differentiable_target,
                 double eps = 1e-5,
                 int max_iterations = 500,
                 OptimizationMethod method = Both);

  // Maximize a twice differentiable function.
  // Args:
  //   x: On input x is the initial set of function arguments of the
  //      algorithm.  On output it is the maximizing value.
  //   g: Gradient.  On output this will be filled with the gradient
  //      of 'target' at the optimal x.
  //   h Hessian: On output this will be filled with the Hessian
  //     (matrix of second derivatives) of 'target' evaluated at the
  //     optimal x.
  //   target:  The function to be maximized.
  //   differentiable_target:  The function to be maximized -- gradient version.
  //   twice_differentiable_target: The function to be maximized --
  //     Hessian version.
  //   convergence_epsilon: When the target function changes by less
  //     than this amount in consecutive iterations the algorithm
  //     declares convergence.
  //
  // Returns:
  //   The value of 'target' at the optimal 'x'.
  double max_nd2(Vector &x,
                 Vector &g,
                 Matrix &h,
                 Target target,
                 dTarget differentiable_target,
                 d2Target twice_differentiable_target,
                 double epsilon = 1e-5);

  // Maximize a twice differentiable function, checking for convergence.
  // Args:
  //   x: On input x is the initial set of function arguments of the
  //      algorithm.  On output it is the maximizing value.
  //   g: Gradient.  On output this will be filled with the gradient
  //      of 'target' at the optimal x.
  //   h Hessian: On output this will be filled with the Hessian
  //     (matrix of second derivatives) of 'target' evaluated at the
  //     optimal x.
  //   max_value:  The value of the function at the optimal 'x'.
  //   target:  The function to be maximized.
  //   differentiable_target:  The function to be maximized -- gradient version.
  //   twice_differentiable_target: The function to be maximized --
  //     Hessian version.
  //   convergence_epsilon: When the target function changes by less
  //     than this amount in consecutive iterations the algorithm
  //     declares convergence.
  //   error_msg: Describes the problem encountered in case of
  //     failure.  Empty if successful.
  //
  // Returns:
  //   Returns 'true' on successful execution.  If an error was
  //   encountered 'false' is returned and details are supplied in
  //   error_msg.
  bool max_nd2_careful(Vector &x,
                       Vector &g,
                       Matrix &h,
                       double &max_value,
                       Target target,
                       dTarget differentiable_target,
                       d2Target twice_differentiable_target,
                       double epsilon,
                       std::string &error_msg);

  //--------- Methods: Each includes a full interface and an inline
  //--------- function providing a simpler interface

  double nelder_mead_driver(Vector &x,
                            Vector &y,
                            const Target &target,
                            double absolute_tolerance,
                            double intol,
                            double alpha, double beta, double gamma,
                            int &fncount, int max_iterations);

  double nelder_mead(Vector &x,
                     Vector &y,
                     const Target &target,
                     double absolute_tolerance,
                     double intol,
                     double alpha, double beta, double gamma,
                     int &fncount, int max_iterations);

  // Minimize a function using the BFGS algorithm.
  // Args:
  //   x: On input x is the initial set of function arguments of the
  //      algorithm.  On output it is the maximizing value.
  //   target:  The function to be maximized.
  //   dtarget:  The function to be maximized, with gradient.
  //   max_iterations:  The maximum number of bfgs iterations allowed.
  //   absolute_tolerance: The absolute convergence criterion.
  //   relative_tolerance: The relative convergence criterion.
  //   fncount: On output this is filled with the number of function
  //     evaluations that were made.
  //   grcount: On output this is filled with the number of gradient
  //     evaluations that were made.
  //   fail:  Filled with 'true' on successful exit, and 'false' otherwise.
  //   trace_freq: If > 0 then a progress message is output each
  //     trace_freq iterations.
  //
  // Returns:
  //   The value of target at the minimizing x.
  double bfgs(Vector &x,
              const Target &target,
              const dTarget &dtarget,
              int max_iterations,
              double absolute_tolerance,
              double relative_tolerance,
              int &fncount,
              int &grcount,
              bool &fail,
              int trace_freq= -1);

  // Minimize the function f using the conjugate gradient algorithm.
  // Args:

  //   x: On input x is the initial set of function arguments for the
  //     algorithm.  On output x is the value that minimizes target.
  //   Fmin: The value of the function at the minimizing x.
  //   target:  A functor returning the value of the function to be optimized.
  //   dtarget: A functor that computes the gradient of the function
  //     to be optimized.
  //   absolute_tolerance:  Absolute tolerance.
  //   relative_tolerance:  Relative tolerance.
  //   method:  Which flavor of conjugate gradient algorithm should be used?
  //   fcnt:  The number of times target was evaluated.
  //   gcnt:  The number of times dtarget was evaluated.
  //   max_iterations:  The maximum number of iterations.
  //   error_msg:  The error message produced in case of failure.
  //
  // Returns:
  //   A return value of true indicates success.  A return value of
  //   false indicates an error or unusual condition, which will be
  //   explained in error_msg.
  bool conj_grad(Vector &x,
                 double &Fmin,
                 const Target &target,
                 const dTarget &df,
                 double absolute_tolerance,
                 double relative_tolerance,
                 ConjugateGradientMethod method,
                 int &fcnt,
                 int &gcnt,
                 int max_iterations,
                 std::string &error_message);

  // Minimize the function f using the Newton-Raphson algorithm.
  // Args:
  //   x: The argument to f.  Input specifies the starting value for
  //     the optimization algorithm.  Output gives the optimizing
  //     value of x.
  //   g: The gradient, to be computed by f.  This should either be
  //     sized appropriately on input, or f should handle the
  //     resizing.
  //   h:  The hessian, to be computed by f.  This should either be
  //     sized appropriately on input, or f should handle the
  //     resizing.
  //   f:  The function to be minimized.
  //   function_call_count:  output.  No need to initialize this.
  //   eps: The algorithm will converge when the (absolute) change in
  //     function values is less than eps.
  //   happy_ending: If true then the algorithm converged happily.  If
  //     not there was a (potential) problem.
  //   error_message: If happy_ending is false, then the reason will
  //     be written to error_message.  If happy_ending is true then
  //     error_message is not used.
  //
  // Returns:
  //   The value of the function at the conclusion of the algorithm.
  double newton_raphson_min(Vector &x,
                            Vector &g,
                            Matrix &h,
                            const d2Target &target,
                            int &function_call_count,
                            double eps,
                            bool & happy_ending,
                            std::string &error_message);

  // Minimize a function using derivative-free simulated annealing.
  // Args:

  //   x: The argument of the function to be optimized.  Input
  //     specifies the initial set of function arguments for the
  //     algorithm.  Output gives the value that optimizes f.
  //   f:  The function to be minimized.
  //   max_iterations:  The maximum number of function evaluations.
  //   tmax:  The maximum number of evaluations at each temperature step.
  //   ti: "Temperature increment".  Used to adjust the scale of the
  //     random annealing perturbations.
  //
  // Returns:
  //   On exit x is the (approximate) minimizing value of f, and the
  //   return value is f(x).
  double simulated_annealing(Vector &x,
                             const Target &target,
                             int max_iterations,
                             int tmax,
                             double ti);

  //======================================================================
  // Negations:
  //
  // Classes to use as target functions when maximizing a function
  // of a single variable.  Minimizing the ScalarNegation of f(x)
  // maximizes f(x).
  class ScalarNegation {
   public:
    explicit ScalarNegation(const ScalarTarget &target)
        : original_function_(target) {}
    double operator()(double x)const{ return -1 * original_function_(x); }
   private:
    ScalarTarget original_function_;
  };

  // Minimizing a Negate(F) maximizes F, where F is a function of many
  // variables.
  class Negate{
  public:
    explicit Negate(const Target &F) : f(F){}
    double operator()(const Vector &x)const;
   private:
    Target f;
  };

  // Use this negation when F has first derivatives.
  class dNegate : public Negate{
  public:
    dNegate(const Target &F, const dTarget &dF)
      : Negate(F), df(dF){}
    double operator()(const Vector &x)const{
      return Negate::operator()(x);}
    double operator()(const Vector &x, Vector &g)const;
  private:
    dTarget df;
  };

  // Use this Negation when F has first and second derivatives.
  class d2Negate : public dNegate{
  public:
    d2Negate(const Target &target, const dTarget &df, const d2Target &d2F)
      : dNegate(target, df), d2f(d2F){}
    double operator()(const Vector &x)const{
      return Negate::operator()(x);}
    double operator()(const Vector &x, Vector &g)const{
      return dNegate::operator()(x,g);}
    double operator()(const Vector &x, Vector &g, Matrix &h)const;
  private:
    d2Target d2f;
  };

}
#endif // BOOM_NUMOPT_HPP
