// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2009 Steven L. Scott

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

#ifndef BOOM_DISTRIBUTIONS_RNG_HPP
#define BOOM_DISTRIBUTIONS_RNG_HPP

#include <random>
#include <cstdint>

namespace BOOM {
  // A random number generator for simulating real valued U[0, 1) deviates.
  class RNG {
   public:
    using RngIntType = std::uint_fast64_t;

    // Seed with std::random_device.
    RNG();

    // Seed with a specified value.
    explicit RNG(RngIntType seed);

    // Seed from a C++ standard random device, if one is present.
    void seed();

    // Seed using a specified value.
    void seed(RngIntType seed) {generator_.seed(seed);}

    // Simulate a U[0, 1) random deviate.
    double operator()() {return dist_(generator_);}

    std::mt19937_64 & generator() {return generator_;}

   private:
    // TODO(steve): once you can use c++17 in R and elsewhere replace this with
    // a std::variant that will choose the fastest RNG for each implementation.
    std::mt19937_64 generator_;
    std::uniform_real_distribution<double> dist_;
  };

  // The GlobalRng is a singleton.
  struct GlobalRng {
   public:
    static RNG rng;
  };

  RNG::RngIntType seed_rng(RNG &rng = GlobalRng::rng);
  // generate a random seed from the global RNG used to seed other RNG's
}  // namespace BOOM

#endif  // BOOM_DISTRIBUTIONS_RNG_HPP
