#ifndef BAYESBOOM_FUNCTIONAL_HPP_
#define BAYESBOOM_FUNCTIONAL_HPP_
/*
  Copyright (C) 2005-2026 Steven L. Scott

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

// Tools for passing Python callables into C++ code that uses std::function.
//
// C++ library code that consumes the callable needs only:
//   #include <functional>
//
// ---- Generic approach (fully type-erased) ------------------------------------
//
// Include this header in any binding file that accepts a Python callable as a
// std::function<R(Args...)> parameter.  pybind11 converts any Python callable
// automatically at the call boundary:
//
//   // C++ library:
//   void run(std::function<double(double)> fn);
//
//   // Binding layer:
//   #include "boom_functional.hpp"
//   boom.def("run", [](std::function<double(double)> fn) { run(fn); });
//
//   // Python:
//   boom.run(lambda x: x * 2)
//
// ---- Typed approach (explicit signature) ------------------------------------
//
// TypedFunctor<R(Args...)> wraps a Python callable with a concrete C++ type
// that carries the signature as a template parameter.  Register named
// instantiations in the pybind11 module so Python users construct typed
// wrapper objects rather than passing raw callables:
//
//   // Binding layer:
//   using ScalarFn = BayesBoom::TypedFunctor<double(double)>;
//   py::class_<ScalarFn>(boom, "ScalarFn")
//       .def(py::init<py::object>(), py::arg("fn"))
//       .def("__call__", &ScalarFn::operator())
//       ;
//   boom.def("run", [](const ScalarFn &fn) { run(fn); });
//
//   // Python:
//   fn = boom.ScalarFn(lambda x: x * 2)
//   boom.run(fn)
//
// TypedFunctor implicitly converts to std::function<R(Args...)>, so it can be
// passed directly to any C++ function that accepts the matching std::function
// type.  The wrapped std::function acquires the GIL before invoking the Python
// callable.

#include <functional>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace BayesBoom {

  namespace py = pybind11;

  // Primary template — only the partial specialization below is defined.
  template<typename Signature>
  class TypedFunctor;

  // TypedFunctor<R(Args...)>: a Python callable bound to a specific C++
  // call signature.
  template<typename R, typename... Args>
  class TypedFunctor<R(Args...)> {
   public:
    explicit TypedFunctor(py::object fn) : fn_(std::move(fn)) {
      if (!py::callable(fn_)) {
        throw py::type_error("TypedFunctor requires a callable");
      }
    }

    R operator()(Args... args) const {
      return fn_(std::forward<Args>(args)...).template cast<R>();
    }

    // Implicit conversion to the matching std::function type so a TypedFunctor
    // can be passed to any C++ library function that accepts std::function.
    operator std::function<R(Args...)>() const {
      py::object fn = fn_;
      return [fn](Args... args) -> R {
        return fn(std::forward<Args>(args)...).template cast<R>();
      };
    }

   private:
    py::object fn_;
  };

}  // namespace BayesBoom

#endif  // BAYESBOOM_FUNCTIONAL_HPP_
