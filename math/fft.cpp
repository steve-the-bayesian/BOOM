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
#include <iostream>

// namespace FFT {
//   void kiss_fftr(const RealConfig &cfg,
//                  const std::vector<double> &timedata,
//                  const std::vector<std::complex<double>> &freqdata);

//   void kiss_fftri(RealConfig &cfg,
//                   const std::vector<std::complex<double>> &freqdata,
//                   std::vector<double> &timedata);

// }  // namespace FFT

namespace BOOM {

  std::vector<std::complex<double>>
  FastFourierTransform::transform(const Vector &time_domain) {
    size_t nfft = time_domain.size();
    std::vector<std::complex<double>> freq_domain(nfft);
    if (nfft %2 == 0) {
      FFT::RealConfig config(time_domain.size(), false);
      //    std::vector<std::complex<double>> freq_domain(nfft / 2 + 1);
      FFT::kiss_fftr(config, time_domain, freq_domain);
      reflect(freq_domain);
    } else {
      std::vector<std::complex<double>> odd_complex;
      std::vector<std::complex<double>> freq_domain;
      for (auto &el : time_domain) {
        odd_complex.push_back(std::complex<double>(el, 0.0));
      }
      FFT::Config config(time_domain.size(), false);
      FFT::kiss_fft(config, odd_complex, freq_domain);
    }
    return freq_domain;
    // Vector ans(time_domain.size());
    // for (int i = 0; i < nfft; ++i) {
    //   ans[2 * i] = freq_domain[i].real();
    //   ans[2 * i + 1] = freq_domain[i].imag();
    // }
    // return ans;
  }

  void FastFourierTransform::reflect(std::vector<std::complex<double>> &freq) const {
    size_t half_size = freq.size() / 2;
    if (half_size % 2) {
      // odd case
      for (size_t i = 1; i < half_size; ++i) {
        freq[half_size + i].real(freq[half_size - i].real());
        freq[half_size + i].imag(-freq[half_size - i].imag());
      }
    } else {
      // even case
    }

  }

  Vector FastFourierTransform::inverse_transform(
      const std::vector<std::complex<double>> &freq_domain) {
    size_t nfft = freq_domain.size();
    Vector ans(nfft);
    // std::vector<std::complex<double>> complex_freq(nfft / 2 + 1);
    // for (int i = 0; i < nfft; ++i){
    //   complex_freq[i / 2].real(freq_domain[i]);
    //   complex_freq[i / 2].imag(freq_domain[i + 1]);
    // }

    FFT::RealConfig config(nfft, true);
    FFT::kiss_fftri(config, freq_domain, ans);
    return ans;
  }

  void FastFourierTransform::print_config(int data_size, bool inverse) {
    FFT::RealConfig config(data_size, inverse);
    std::cout << config;
  }


}  // namespace BOOM
