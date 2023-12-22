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

#include "math/fft.hpp"
#include "math/kissfft/kiss_fft.hpp"
#include <vector>
#include <complex>
#include <sstream>
#include "cpputil/report_error.hpp"
#include "LinAlg/Matrix.hpp"

namespace {

  using ComplexVector = std::vector<std::complex<double>>;

  // The following function is unused in active code, but is useful for
  // debugging.
  //
  // std::string print_complex_vector(const ComplexVector &z) {
  //   std::ostringstream out;
  //   for (size_t i = 0; i < z.size(); ++i) {
  //     out << std::setw(10) << z[i].real() << "   " << z[i].imag()
  //         << "  i\n";
  //   }
  //   return out.str();
  // }

} // namespace

namespace BOOM {

  // Transform a time domain vector to the frequency domain.
  ComplexVector FastFourierTransform::transform(
      const Vector &time_domain) const {
    size_t nfft = time_domain.size();
    ComplexVector freq_domain(nfft);
    if (nfft %2 == 0) {
      FFT::RealConfig config(time_domain.size(), false);
      //    ComplexVector freq_domain(nfft / 2 + 1);
      FFT::kiss_fftr(config, time_domain, freq_domain);
      reflect(freq_domain);
    } else {
      ComplexVector odd_complex;

      for (auto &el : time_domain) {
        odd_complex.push_back(std::complex<double>(el, 0.0));
      }
      FFT::Config config(time_domain.size(), false);
      FFT::kiss_fft(config, odd_complex, freq_domain);
    }
    return freq_domain;
  }

  // Transform a complex valued input sequence.
  ComplexVector FastFourierTransform::complex_transform(
      const ComplexVector &input) const {
    FFT::Config config(input.size(), false);
    ComplexVector ans(input.size());
    FFT::kiss_fft(config, input, ans);
    return ans;
  }

  // Invserse-transform a complex valued sequence.
  ComplexVector FastFourierTransform::inverse_complex_transform(
      const ComplexVector &input) const {
    FFT::Config config(input.size(), true);
    ComplexVector ans(input.size());
    FFT::kiss_fft(config, input, ans);
    return ans;
  }

  // Reflect the values of a series around the halfway point.
  void FastFourierTransform::reflect(ComplexVector &freq) const {
    size_t half_size = freq.size() / 2;
    for (size_t i = 1; i < half_size; ++i) {
      freq[half_size + i].real(freq[half_size - i].real());
      freq[half_size + i].imag(-freq[half_size - i].imag());
    }
  }

  // The inverse transform of a complex valued series (known to correspond to a
  // real-valued time domain) to the real valued time domain.
  Vector FastFourierTransform::inverse_transform(
      const ComplexVector &freq_domain) const {
    size_t nfft = freq_domain.size();
    Vector ans(nfft);

    if (nfft % 2 == 0) {
      FFT::RealConfig config(nfft, true);
      FFT::kiss_fftri(config, freq_domain, ans);
    } else {
      ComplexVector output(nfft);
      FFT::Config config(nfft, true);
      FFT::kiss_fft(config, freq_domain, output);
      for (int i = 0; i < nfft; ++i) {
        ans[i] = output[i].real();
        if (fabs(output[i].imag()) > 1e-5) {
          std::ostringstream err;
          err << "Possibly nonzero output discovered in position "
              << i << ".  " << output[i] << ".";
          report_error(err.str());
        }
      }
    }
    return ans;
  }

  std::string FastFourierTransform::print_config(
      int data_size, bool inverse) const {
    std::ostringstream out;
    if (data_size % 2 == 0) {
      FFT::RealConfig config(data_size, inverse);
      out << config;
    } else {
      FFT::Config config(data_size, inverse);
      out << config;
    }
    return out.str();
  }

}  // namespace BOOM
