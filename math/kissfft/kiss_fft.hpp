#ifndef KISS_FFT_HPP
#define KISS_FFT_HPP

/*
 *  Copyright (c) 2003-2010, Mark Borgerding. All rights reserved.
 */

/*
 * The C++ code in this file was adapted from the KISS FFT project (written in
 * C) - found at https://github.com/mborgerding/kissfft
 *
 * The adaptation was done under the
 * under the BSD-3-Clause license, reproduced in the LICENSE section below.

 * LICENSE
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.

 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.

 * 3. Neither the name of the copyright holder nor the names of its contributors
 * may be used to endorse or promote products derived from this software without
 * specific prior written permission.

 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS”
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

/*
 *  The following notice applies to the C++ adaptation:
 *
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

#include <cmath>
#include <complex>
#include <vector>
#include <iostream>
#include <ostream>

std::ostream & operator<<(std::ostream &out, const std::vector<std::complex<double>> &cv);

namespace FFT {

  // A configuration for a complex to complex FFT.
  class Config {
   public:
    explicit Config(int nfft = 0, bool inverse = false);
    int nfft;
    int inverse;
    int factors[64];
    std::vector<std::complex<double>> twiddles;

    virtual std::ostream & print(std::ostream &out) const;
   protected:
    void resize(int nfft_arg);
  };

  // A configuration for a real to complex FFT.
  class RealConfig : public Config {
   public:
    RealConfig(int nfft, bool inverse);
    std::vector<std::complex<double>> tmpbuf;
    std::vector<std::complex<double>> super_twiddles;

    std::ostream & print(std::ostream &out) const override;
  };

  inline std::ostream & operator<<(std::ostream &out, const Config &cfg) {
    return cfg.print(out);
  }

  /*
   * kiss_fft(cfg,in_out_buf)
   *
   * Perform an FFT on a complex input buffer.
   * for a forward FFT,
   * fin should be  f[0] , f[1] , ... ,f[nfft-1]
   * fout will be   F[0] , F[1] , ... ,F[nfft-1]
   * Note that each element is complex and can be accessed like
   *    f[k].real() and f[k].imag().
   * */
  void kiss_fft(Config &cfg,
                const std::vector<std::complex<double>> &fin,
                std::vector<std::complex<double>> &fout);

  /*
   * A more generic version of the above function. It reads its input from every
   * Nth sample.
   */
  void kiss_fft_stride(Config &cfg,
                       const std::vector<std::complex<double>> &fin,
                       std::vector<std::complex<double>> &fout,
                       int fin_stride);


  /*
   * Returns the smallest integer k, such that k>=n and k has only "fast" factors (2,3,5)
   */
  inline int kiss_fft_next_fast_size(int n) {
    while(1) {
      int m = n;
      while ( (m % 2) == 0 ) m /= 2;
      while ( (m % 3) == 0 ) m /= 3;
      while ( (m % 5) == 0 ) m /= 5;
      if (m <= 1)
        break; /* n is completely factorable by twos, threes, and fives */
      n++;
    }
    return n;
  }

  /* for real ffts, we need an even size */
  inline int kiss_fftr_next_fast_size_real(int n) {
    return (kiss_fft_next_fast_size( ((n)+1)>>1)<<1);
  }


  /*
    input timedata has nfft scalar points
    output freqdata has nfft/2+1 complex points
  */
  void kiss_fftr(RealConfig &cfg,
                 const std::vector<double> &timedata,
                 std::vector<std::complex<double>> &freqdata);

  /*
    input freqdata has  nfft/2+1 complex points
    output timedata has nfft scalar points
  */
  void kiss_fftri(RealConfig &cfg,
                  const std::vector<std::complex<double>> &freqdata,
                  std::vector<double> &timedata);


}  // namespace FFT

#endif
