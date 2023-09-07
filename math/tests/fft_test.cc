#include "gtest/gtest.h"

#include "math/fft.hpp"

#include "test_utils/test_utils.hpp"
#include <fstream>

namespace {
  using namespace BOOM;
  using std::endl;
  using std::cout;

  class FFTtest : public ::testing::Test {
   protected:
    FFTtest() {
      GlobalRng::rng.seed(8675309);
    }
  };


  // I generated a sequence of random numbers in R and used R's fft to take the
  // transform.
  TEST_F(FFTtest, MatchesR) {
    Vector x = {0.699069393237643, 0.243298407053864, 1.82365326326982,
      0.814619494870901, 0.914597639093842, -0.136970891187017,
      -0.104400202613791, 1.04582759831447, -0.869331165040763,
      -0.379856081183586};

    Vector real_z = {4.05050745581537, -0.209908712039394, -1.50693147439372,
      1.86167422306198, 0.886924001612595, 0.876670400078121, 0.886924001612595,
      1.86167422306198, -1.50693147439372, -0.209908712039394};

    Vector imag_z = {0, -3.30652133698535, -1.34232784427893,
      -0.114783855943944, 3.01374324761417, -2.22044604925031e-16,
      -3.01374324761417, 0.114783855943944, 1.34232784427893, 3.30652133698535};

    /*  Output from native C implementation matches R.
        x         z.real         z.imag
  0.699069      4.05051             0
  0.243298    -0.209909      -3.30652
   1.82365     -1.50693      -1.34233
  0.814619      1.86167     -0.114784
  0.914598     0.886924       3.01374
 -0.136971      0.87667             0
   -0.1044   -3.9933e-06    4.59051e-41
   1.04583            0             0
 -0.869331            0             0
 -0.379856            0             0
    */

    /*
      RealConfig printed as part of the test:
nfft: 10
inverse: 0
factors:  2  5  5  1  0  0  266652760  1  6  0  -1375714039  32753  7  0  0  0  0  0  0  0  0  0  0  0  1370245635  32760  16  0  0  0  0  0  0  0  0  0  0  0  -1340353840  32759  805310464  -156598264  -1340342273  32759  4  0  267520000  1  4  0  267517952  1  0  0  80  0  2834592  24576  2834592  24576  2834752  24576  2834752  24576
twiddles:
    1    -0
    0.809017    -0.587785
    0.309017    -0.951057
    -0.309017    -0.951057
    -0.809017    -0.587785
    -1    -1.22465e-16
    -0.809017    0.587785
    -0.309017    0.951057
    0.309017    0.951057
    0.809017    0.587785
tmpbuf:
   0   0
   0   0
   0   0
   0   0
   0   0
   0   0
   0   0
   0   0
   0   0
   0   0
super_twiddles:
   -0.309017   -0.951057
   -0.587785   -0.809017
   -0.809017   -0.587785
   -0.951057   -0.309017
   -1   -1.22465e-16
   0   0
   0   0
   0   0
   0   0
   0   0


Printed from C code-----
Config:
nfft: 5
inverse: 0
factors:  5  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0

twiddles:
            1          -0 i
     0.309017   -0.951057 i
    -0.809017   -0.587785 i
    -0.809017    0.587785 i
     0.309017    0.951057 i
tmpbuf:
      2.46359     1.58692 i
      0.77929    -3.51777 i
      2.00437   -0.441352 i
     -1.64963    0.786192 i
    -0.102275      2.8025 i
super twiddles:
      2.46359     1.58692 i
      0.77929    -3.51777 i
     */


    FastFourierTransform fft;
    std::vector<std::complex<double>> z = fft.transform(x);

    fft.print_config(x.size(), false);

    EXPECT_EQ(z.size(), x.size());
    Matrix entries(x.size(), 5);
    entries.col(0) = x;
    entries.col(1) = real_z;
    entries.col(2) = imag_z;
    for (int i = 0; i < x.size(); ++i) {
      entries(i, 3) = z[i].real();
      entries(i, 4) = z[i].imag();
    }

    for (int i = 0; i < x.size(); ++i) {
      EXPECT_NEAR(real_z[i], z[i].real(), 1e-6) << "\n" << entries;
      EXPECT_NEAR(imag_z[i], z[i].imag(), 1e-6) << "\n" << entries;
    }
  }

}  // namespace
