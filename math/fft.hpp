#ifndef BOOM_MATH_FFT_HPP_
#define BOOM_MATH_FFT_HPP_
/*
  Copyright (C) 2005-2023 Steven L. Scott

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

#include "LinAlg/Vector.hpp"
#include <vector>
#include <complex>

namespace FFT {
  class RealConfig;
}  // namespace FFT

namespace BOOM {

  class FastFourierTransform {
   public:

    // The discrete Fourier transform of a real valued sequence.
    //
    // Args:
    //   time_domain:  The sequence of values to transform.
    //
    // Returns:
    //   A complex vector of the same length as time_domain.  Because this
    //   vector contains twice as many numbers as the input, there is some
    //   duplication of information.  The second half of the real part of the
    //   sequence is a reflection of the first half.  The second half of the
    //   imaginary part of the sequence is the negative reflection of the first
    //   half.
    std::vector<std::complex<double>> transform(
        const Vector &time_domain) const;

    // The inverse discrete Fourier transform of a complex sequence known to
    // correspond to a real-valued time domain sequence.
    //
    // Args:
    //   frequency_domain: A sequence of complex numbers to be inverse
    //     transformed.  The second half of the sequence is not accessed, and is
    //     assumed to be a reflection of the first half, as noted in the
    //     documentation to 'transform'.
    //
    // Returns:
    //   A real sequence whose transform (after scaling) is 'frequency_domain'.
    //   Note that if x is a sequence of length n then
    //   inverse_transform(transform(x)) returns x * n.  This is the convention
    //   adopted by many other fft programs, and notably the one used by R.
    Vector inverse_transform(
        const std::vector<std::complex<double>> &frequency_domain) const;

    // Discrete Fourier transform of a complex valued input sequence.  The
    // output sequence has the same length as the input sequence.
    std::vector<std::complex<double>> complex_transform(
        const std::vector<std::complex<double>> &time_domain) const;

    // Inverse discrete Fourier transform of a complex valued sequence.
    // The returned sequence has the same length as the argument.
    std::vector<std::complex<double>> inverse_complex_transform(
        const std::vector<std::complex<double>> &frequency_domain) const;

    // Print the kiss_fft config object corresponding to a series of the given
    // data size.
    //
    // This is for debugging purposes.
    std::string print_config(int data_size, bool inverse) const;

   protected:
    // Reflect the sequence of complex numbers in the first half of the vector
    // to the second half.  The real component values match.  The imaginary
    // component matches after multiplying by -1.
    void reflect(std::vector<std::complex<double>> &freq) const;
  };

}  // namespace BOOM

#endif  //  BOOM_MATH_FFT_HPP_
