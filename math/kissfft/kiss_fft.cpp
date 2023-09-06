/*
 *  Copyright (c) 2003-2010, Mark Borgerding. All rights reserved.
 *  This file is part of KISS FFT - https://github.com/mborgerding/kissfft
 *
 *  SPDX-License-Identifier: BSD-3-Clause
 *  See COPYING file for more information.
 */


#include "kiss_fft.hpp"

namespace BOOM {
  void report_error(const std::string &message);
}

/* The guts header contains all the multiplication and addition macros that are defined for
 fixed or floating point complex numbers.  It also delares the kf_ internal functions.
 */

namespace FFT {
  using BOOM::report_error;

  namespace {
    // ============================================================
    // Begin Butterflies!
    // ============================================================
    void kf_bfly2(std::complex<double> *Fout,
                  const size_t fstride,
                  const Config &config,
                  int m) {
      std::complex<double> * Fout2;
      const std::complex<double> * tw1 = config.twiddles.data();
      std::complex<double> t;
      Fout2 = Fout + m;
      do{
        // C_FIXDIV(*Fout,2);
        // C_FIXDIV(*Fout2,2);
        *Fout /= 2.0;
        *Fout2 /= 2.0;

        //        C_MUL (t,  *Fout2 , *tw1);
        t = *Fout2 * *tw1;
        tw1 += fstride;

        // C_SUB( *Fout2 ,  *Fout , t );
        *Fout2 = *Fout - t;

        //         C_ADDTO( *Fout ,  t );
        *Fout += t;

        ++Fout2;
        ++Fout;
      } while (--m);
    }

    //----------------------------------------------------------------------
    void kf_bfly4(std::complex<double> * Fout,
                  const size_t fstride,
                  const Config &config,
                  const size_t m) {

      const std::complex<double> *tw1, *tw2, *tw3;
      std::complex<double> scratch[6];
      size_t k = m;
      const size_t m2 = 2 * m;
      const size_t m3 = 3 * m;

      tw3 = tw2 = tw1 = config.twiddles.data();

      do {
        // C_FIXDIV(*Fout,4);
        // C_FIXDIV(Fout[m],4);
        // C_FIXDIV(Fout[m2],4);
        // C_FIXDIV(Fout[m3],4);
        *Fout /= 4.0;
        Fout[m] /= 4.0;
        Fout[m2] /= 4.0;
        Fout[m3] /= 4.0;

        // C_MUL(scratch[0],Fout[m] , *tw1 );
        // C_MUL(scratch[1],Fout[m2] , *tw2 );
        // C_MUL(scratch[2],Fout[m3] , *tw3 );
        scratch[0] = Fout[m] * *tw1;
        scratch[1] = Fout[m2] * *tw2;
        scratch[2] = Fout[m3] * *tw3;

        //       C_SUB( scratch[5] , *Fout, scratch[1] );
        scratch[5] = *Fout - scratch[1];

        // C_ADDTO(*Fout, scratch[1]);
        *Fout += scratch[1];

        // C_ADD( scratch[3] , scratch[0] , scratch[2] );
        scratch[3] = scratch[0] + scratch[2];

        //       C_SUB( scratch[4] , scratch[0] , scratch[2] );
        scratch[4] = scratch[0] - scratch[2];

        // C_SUB( Fout[m2], *Fout, scratch[3] );
        Fout[m2] = *Fout - scratch[3];

        tw1 += fstride;
        tw2 += fstride*2;
        tw3 += fstride*3;

        // C_ADDTO( *Fout , scratch[3] );
        *Fout += scratch[3];

        if(config.inverse) {
          Fout[m].real(scratch[5].real() - scratch[4].imag());
          Fout[m].imag(scratch[5].imag() + scratch[4].real());
          Fout[m3].real(scratch[5].real() + scratch[4].imag());
          Fout[m3].imag(scratch[5].imag() - scratch[4].real());
        }else{
          Fout[m].real(scratch[5].real() + scratch[4].imag());
          Fout[m].imag(scratch[5].imag() - scratch[4].real());
          Fout[m3].real(scratch[5].real() - scratch[4].imag());
          Fout[m3].imag(scratch[5].imag() + scratch[4].real());
        }
        ++Fout;
      } while (--k);
    }

    //----------------------------------------------------------------------
    void kf_bfly3(std::complex<double> * Fout,
                  const size_t fstride,
                  const Config &config,
                  size_t m) {
      size_t k=m;
      const size_t m2 = 2 * m;
      const std::complex<double> *tw1, *tw2;
      std::complex<double> scratch[5];
      std::complex<double> epi3;
      epi3 = config.twiddles[fstride * m];

      tw1 = tw2 = config.twiddles.data();

      do{
        // C_FIXDIV(*Fout,3);
        // C_FIXDIV(Fout[m],3);
        // C_FIXDIV(Fout[m2],3);
        *Fout /= 3.0;
        Fout[m] /= 3.0;
        Fout[m2] /= 3.0;

        // C_MUL(scratch[1],Fout[m] , *tw1);
        // C_MUL(scratch[2],Fout[m2] , *tw2);
        scratch[1] = Fout[m] * *tw1;
        scratch[2] = Fout[m2] * *tw2;

        // C_ADD(scratch[3],scratch[1],scratch[2]);
        // C_SUB(scratch[0],scratch[1],scratch[2]);
        scratch[3] = scratch[1] + scratch[2];
        scratch[0] = scratch[1] - scratch[2];
        tw1 += fstride;
        tw2 += fstride * 2;

        Fout[m].real(Fout->real() - .5 * scratch[3].real());
        Fout[m].imag(Fout->imag() - .5 * scratch[3].imag());

        // C_MULBYSCALAR( scratch[0] , epi3.imag() );
        scratch[0] *= epi3.imag();

        // C_ADDTO(*Fout,scratch[3]);
        *Fout += scratch[3];

        Fout[m2].real(Fout[m].real() + scratch[0].imag());
        Fout[m2].imag(Fout[m].imag() - scratch[0].real());

        Fout[m].real(Fout[m].real() - scratch[0].imag());
        Fout[m].imag(Fout[m].imag() + scratch[0].real());

        ++Fout;
      } while (--k);
    }

    //----------------------------------------------------------------------
    void kf_bfly5(std::complex<double> * Fout,
                  const size_t fstride,
                  const Config &config,
                  int m) {
      std::complex<double> *Fout0,*Fout1,*Fout2,*Fout3,*Fout4;
      int u;
      std::complex<double> scratch[13];
      const std::complex<double> *twiddles = config.twiddles.data();
      const std::complex<double> *tw;
      std::complex<double> ya, yb;
      ya = twiddles[fstride * m];
      yb = twiddles[fstride * 2 * m];

      Fout0=Fout;
      Fout1=Fout0+m;
      Fout2=Fout0+2*m;
      Fout3=Fout0+3*m;
      Fout4=Fout0+4*m;

      tw=config.twiddles.data();
      for ( u=0; u<m; ++u ) {
        // C_FIXDIV( *Fout0,5);
        // C_FIXDIV( *Fout1,5);
        // C_FIXDIV( *Fout2,5);
        // C_FIXDIV( *Fout3,5);
        // C_FIXDIV( *Fout4,5);
        *Fout0 /= 5.0;
        *Fout1 /= 5.0;
        *Fout2 /= 5.0;
        *Fout3 /= 5.0;
        *Fout4 /= 5.0;

        scratch[0] = *Fout0;

        // C_MUL(scratch[1] ,*Fout1, tw[u*fstride]);
        // C_MUL(scratch[2] ,*Fout2, tw[2*u*fstride]);
        // C_MUL(scratch[3] ,*Fout3, tw[3*u*fstride]);
        // C_MUL(scratch[4] ,*Fout4, tw[4*u*fstride]);
        scratch[1] = *Fout1 * tw[u*fstride];
        scratch[2] = *Fout2 * tw[2*u*fstride];
        scratch[3] = *Fout3 * tw[3*u*fstride];
        scratch[4] = *Fout4 * tw[4*u*fstride];

        // C_ADD( scratch[7],scratch[1],scratch[4]);
        // C_SUB( scratch[10],scratch[1],scratch[4]);
        // C_ADD( scratch[8],scratch[2],scratch[3]);
        // C_SUB( scratch[9],scratch[2],scratch[3]);

        scratch[7] = scratch[1] + scratch[4];
        scratch[10] = scratch[1]- scratch[4];
        scratch[8] = scratch[2] + scratch[3];
        scratch[9] = scratch[2] - scratch[3];

        Fout0->real(Fout0->real() + scratch[7].real() + scratch[8].real());
        Fout0->imag(Fout0->imag() + scratch[7].imag() + scratch[8].imag());

        // scratch[5].real() = scratch[0].real() + S_MUL(scratch[7].real(),ya.real()) + S_MUL(scratch[8].real(),yb.real());
        // scratch[5].imag() = scratch[0].imag() + S_MUL(scratch[7].imag(),ya.real()) + S_MUL(scratch[8].imag(),yb.real());
        scratch[5].real(scratch[0].real() + scratch[7].real() * ya.real() + scratch[8].real() * yb.real());
        scratch[5].imag(scratch[0].imag() + scratch[7].imag() * ya.real() + scratch[8].imag() * yb.real());

        // scratch[6].real() =  S_MUL(scratch[10].imag(),ya.imag()) + S_MUL(scratch[9].imag(),yb.imag());
        // scratch[6].imag() = -S_MUL(scratch[10].real(),ya.imag()) - S_MUL(scratch[9].real(),yb.imag());
        scratch[6].real(scratch[10].imag() * ya.imag() + scratch[9].imag() * yb.imag());
        scratch[6].imag(-scratch[10].real() * ya.imag() - scratch[9].real() * yb.imag());

        // C_SUB(*Fout1,scratch[5],scratch[6]);
        // C_ADD(*Fout4,scratch[5],scratch[6]);
        *Fout1 = scratch[5] - scratch[6];
        *Fout4 = scratch[5] + scratch[6];

        // scratch[11].real() = scratch[0].real() + S_MUL(scratch[7].real(),yb.real()) + S_MUL(scratch[8].real(),ya.real());
        // scratch[11].imag() = scratch[0].imag() + S_MUL(scratch[7].imag(),yb.real()) + S_MUL(scratch[8].imag(),ya.real());
        // scratch[12].real() = - S_MUL(scratch[10].imag(),yb.imag()) + S_MUL(scratch[9].imag(),ya.imag());
        // scratch[12].imag() = S_MUL(scratch[10].real(),yb.imag()) - S_MUL(scratch[9].real(),ya.imag());

        scratch[11].real(scratch[0].real() + scratch[7].real() * yb.real() + scratch[8].real() * ya.real());
        scratch[11].imag(scratch[0].imag() + scratch[7].imag() * yb.real() + scratch[8].imag() * ya.real());
        scratch[12].real(-scratch[10].imag() * yb.imag() + scratch[9].imag() * ya.imag());
        scratch[12].imag(scratch[10].real() * yb.imag() - scratch[9].real() * ya.imag());

        // C_ADD(*Fout2,scratch[11],scratch[12]);
        // C_SUB(*Fout3,scratch[11],scratch[12]);
        *Fout2 = scratch[11] + scratch[12];
        *Fout3 = scratch[11] - scratch[12];

        ++Fout0;++Fout1;++Fout2;++Fout3;++Fout4;
      }
    }

    /* perform the butterfly for one stage of a mixed radix FFT */
    void kf_bfly_generic(std::complex<double> * Fout,
                         const size_t fstride,
                         const Config &config,
                         int m,
                         int p)
    {
      const std::complex<double> * twiddles = config.twiddles.data();
      std::complex<double> t;
      int Norig = config.nfft;

      //      std::complex<double> * scratch = (std::complex<double>*)KISS_FFT_TMP_ALLOC(sizeof(std::complex<double>)*p);
      std::vector<std::complex<double>> scratch(p);

      for (int u = 0; u < m; ++u) {
        int k = u;
        for (int q1 = 0 ; q1 < p ; ++q1 ) {
          scratch[q1] = Fout[ k  ];
          // C_FIXDIV(scratch[q1],p);
          scratch[q1] /= p;
          k += m;
        }

        k=u;
        for (int q1 = 0 ; q1 < p ; ++q1) {
          int twidx=0;
          Fout[ k ] = scratch[0];
          for (int q = 1; q < p; ++q) {
            twidx += fstride * k;
            if (twidx>=Norig) twidx-=Norig;
            t = scratch[q] * twiddles[twidx];
                // C_MUL(t,scratch[q] , twiddles[twidx] );
            Fout[k] += t;
            // C_ADDTO( Fout[ k ] ,t);
          }
          k += m;
        }
      }
    }

    //----------------------------------------------------------------------
    void kf_work(std::complex<double> * Fout,
                 const std::complex<double> * f,
                 const size_t fstride,
                 int in_stride,
                 const int * factors,
                 const Config &config) {
      std::complex<double> * Fout_beg=Fout;
      const int p=*factors++; /* the radix  */
      const int m=*factors++; /* stage's fft length/p */
      const std::complex<double> * Fout_end = Fout + p * m;

      if (m==1) {
        do {
          *Fout = *f;
          f += fstride*in_stride;
        } while(++Fout != Fout_end );
      } else {
        do{
          // recursive call:
          // DFT of size m*p performed by doing
          // p instances of smaller DFTs of size m,
          // each one takes a decimated version of the input
          kf_work( Fout , f, fstride*p, in_stride, factors, config);
          f += fstride * in_stride;
        } while ( (Fout += m) != Fout_end );
      }

      Fout=Fout_beg;

      // recombine the p smaller DFTs
      switch (p) {
        case 2: kf_bfly2(Fout, fstride, config, m); break;
        case 3: kf_bfly3(Fout, fstride, config, m); break;
        case 4: kf_bfly4(Fout, fstride, config, m); break;
        case 5: kf_bfly5(Fout, fstride, config, m); break;
        default: kf_bfly_generic(Fout, fstride, config, m, p); break;
      }
    }  // kf_work

    //----------------------------------------------------------------------
    /*  facbuf is populated by p1,m1,p2,m2, ...
        where
        p[i] * m[i] = m[i-1]
        m0 = n                  */
    void kf_factor(int n, int * facbuf) {
      int p=4;
      double floor_sqrt = floor( sqrt((double)n) );

      /*factor out powers of 4, powers of 2, then any remaining primes */
      do {
        while (n % p) {
          switch (p) {
            case 4: p = 2; break;
            case 2: p = 3; break;
            default: p += 2; break;
          }
          if (p > floor_sqrt)
            p = n;          /* no more factors, skip to end */
        }
        n /= p;
        *facbuf++ = p;
        *facbuf++ = n;
      } while (n > 1);
    }

    // ============================================================
    // End Butterflies!
    // ============================================================

    inline void kf_cexp(std::complex<double> &x, double phase) {
      x.real(cos(phase));
      x.imag(sin(phase));
    }
  }  // namespace

  Config::Config(int nfft_val, bool inverse_val)
      : twiddles(nfft_val)
  {
    nfft = nfft_val;
    inverse = inverse_val;

    for (int i = 0; i < nfft; ++i) {
      constexpr double pi=3.141592653589793238462643383279502884197169399375105820974944;
      double phase = -2 * pi * i / nfft;
      if (inverse) {
          phase *= -1;
      }
      kf_cexp(twiddles[i], phase);
    }

    kf_factor(nfft, factors);
  }


  RealConfig::RealConfig(int nfft, bool inverse)
      : Config(nfft, inverse),
        tmpbuf(nfft),
        super_twiddles(nfft)
  {
    if (nfft & 1) {
      report_error("nfft must be even.\n");
      return;
    }

    for (int i = 0; i < nfft/2; ++i) {
        double phase =
            -3.14159265358979323846264338327 * ((double) (i+1) / nfft + .5);
        if (inverse) {
            phase *= -1;
        }
        kf_cexp(super_twiddles[i], phase);
    }
  }

  void kiss_fft_stride(const Config &config,
                       const std::vector<std::complex<double>> &fin,
                       std::vector<std::complex<double>> &fout,
                       int in_stride)
  {
    if (&fin == &fout) {
      //NOTE: this is not really an in-place FFT algorithm.
      //It just performs an out-of-place FFT into a temp buffer

      std::vector<std::complex<double>> tmpbuf(config.nfft);

      kf_work(tmpbuf.data(),
              fin.data(),
              1,
              in_stride,
              config.factors,
              config);
      fout = tmpbuf;
      // memcpy(fout, tmpbuf, sizeof(std::complex<double>)* config.nfft);
    } else {
      kf_work( fout.data(), fin.data(), 1, in_stride, config.factors, config );
    }
  }

  void kiss_fft(const Config &cfg,
                const std::vector<std::complex<double>> &fin,
                std::vector<std::complex<double>> &fout)
  {
    kiss_fft_stride(cfg, fin, fout, 1);
  }

  //----------------------------------------------------------------------
  void kiss_fftr(RealConfig &config,
                 const std::vector<double> &timedata,
                 std::vector<std::complex<double>> &freqdata) {
    /* input buffer timedata is stored row-wise */
    int ncfft;
    std::complex<double> fpnk,fpk,f1k,f2k,tw,tdc;

    if (config.inverse) {
      report_error("kiss fft usage error: improper alloc");
      return;/* The caller did not call the correct function */
    }

    ncfft = config.nfft;

    /*perform the parallel fft of two real signals packed in real,imag*/
    std::vector<std::complex<double>> timedata_as_complex(ncfft / 2);
    for (int i = 0; i < config.nfft / 2; ++i) {
      timedata_as_complex[i].real(timedata[2 * i]);
      timedata_as_complex[i].imag(timedata[2 * i + 1]);
    }
    kiss_fft(config,
             timedata_as_complex,
             config.tmpbuf);
    /* The real part of the DC element of the frequency spectrum in st->tmpbuf
     * contains the sum of the even-numbered elements of the input time sequence
     * The imag part is the sum of the odd-numbered elements
     *
     * The sum of tdc.r and tdc.i is the sum of the input time sequence.
     *      yielding DC of input time sequence
     * The difference of tdc.r - tdc.i is the sum of the input (dot product) [1,-1,1,-1...
     *      yielding Nyquist bin of input time sequence
     */

    tdc.real(config.tmpbuf[0].real());
    tdc.imag(config.tmpbuf[0].imag());
    tdc /= 2.0;
    //     C_FIXDIV(tdc,2);

    // CHECK_OVERFLOW_OP(tdc.real() ,+, tdc.imag());
    double sum = tdc.real() + tdc.imag();
    if (fabs(sum) > std::numeric_limits<double>::max()) {
      report_error("Addition leads to overflow in real fft.");
    }

    double diff = tdc.real() - tdc.imag();
    //    CHECK_OVERFLOW_OP(tdc.real() ,-, tdc.imag());
    if (fabs(diff) > std::numeric_limits<double>::max()) {
      report_error("Subtraction leads to overflot in real fft.");
    }

    freqdata[0].real(tdc.real() + tdc.imag());
    freqdata[ncfft].real(tdc.real() - tdc.imag());
    freqdata[ncfft].imag(0);
    freqdata[0].imag(0);

    for (int k = 1; k <= ncfft/2 ; ++k) {
      fpk    = config.tmpbuf[k];
      fpnk.real(config.tmpbuf[ncfft-k].real());
      fpnk.imag(-config.tmpbuf[ncfft-k].imag());
      //       C_FIXDIV(fpk,2);
      //       C_FIXDIV(fpnk,2);
      fpk /= 2.0;
      fpnk /= 2.0;

      // C_ADD( f1k, fpk , fpnk );
      // C_SUB( f2k, fpk , fpnk );
      // C_MUL( tw , f2k , st->super_twiddles[k-1]);

      f1k = fpk + fpnk;
      f2k = fpk - fpnk;
      tw = f2k * config.super_twiddles[k-1];

      freqdata[k].real(.5 * (f1k.real()+ tw.real()));
      freqdata[k].imag(.5 *(f1k.imag() + tw.imag()));
      freqdata[ncfft-k].real(.5 * (f1k.real() - tw.real()));
      freqdata[ncfft-k].imag(.5 * (tw.imag() - f1k.imag()));
    }
  }

  //----------------------------------------------------------------------
  void kiss_fftri(RealConfig &config,
                  const std::vector<std::complex<double>> &freqdata,
                  std::vector<double>  &timedata) {
    /* input buffer timedata is stored row-wise */
    int k, ncfft;

    if (config.inverse == 0) {
      report_error("kiss fft usage error: improper alloc");
      return;/* The caller did not call the correct function */
    }

    ncfft = config.nfft;

    config.tmpbuf[0].real(freqdata[0].real() + freqdata[ncfft].real());
    config.tmpbuf[0].imag(freqdata[0].real() - freqdata[ncfft].real());
    //    C_FIXDIV(st->tmpbuf[0],2);
    config.tmpbuf[0] /= 2.0;

    for (k = 1; k <= ncfft / 2; ++k) {
      std::complex<double> fk, fnkc, fek, fok, tmp;
      fk = freqdata[k];
      fnkc.real(freqdata[ncfft - k].real());
      fnkc.imag(-freqdata[ncfft - k].imag());
      // C_FIXDIV( fk , 2 );
      // C_FIXDIV( fnkc , 2 );
      fk /= 2.0;
      fnkc /= 2.0;

      // C_ADD (fek, fk, fnkc);
      // C_SUB (tmp, fk, fnkc);
      // C_MUL (fok, tmp, st->super_twiddles[k-1]);
      // C_ADD (st->tmpbuf[k],     fek, fok);
      // C_SUB (st->tmpbuf[ncfft - k], fek, fok);

      fek = fk + fnkc;
      tmp = fk - fnkc;
      fok = tmp * config.super_twiddles[k-1];
      config.tmpbuf[k] = fek + fok;
      config.tmpbuf[ncfft - k] = fek - fok;
      config.tmpbuf[ncfft - k].imag(config.tmpbuf[ncfft - k].imag() * -1);
    }
    /////////////////////////////////////////////////////
    /////////////////////////////////////////////////////
    /////////////////////////////////////////////////////
    /////////////////////////////////////////////////////
    // WTF happens here???
    /////////////////////////////////////////////////////
    /////////////////////////////////////////////////////
    /////////////////////////////////////////////////////
    /////////////////////////////////////////////////////
    /////////////////////////////////////////////////////

    ///////////kiss_fft (config, config.tmpbuf, (kiss_fft_cpx *) timedata);
    std::vector<std::complex<double>> timedata_as_complex(config.nfft);
    kiss_fft (config, config.tmpbuf, timedata_as_complex);
    for (int i = 0; i < config.nfft; ++i) {
      timedata[i] = timedata_as_complex[i/2].real();
      timedata[i + 1] = timedata_as_complex[i/2].imag();
    }
  }





}  // namespace FFT
