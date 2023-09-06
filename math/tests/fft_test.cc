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

    FastFourierTransform fft;
    std::vector<std::complex<double>> z = fft.transform(x);


    EXPECT_EQ(z.size(), x.size());
    LabelledMatrix entries(Matrix(x.size(), 5),
                           {};
    for (int i = 0; i < x.size(); ++i) {
      entries(i, 0) = x[i];
      entries(i, 1) = real_z[i];
      entries(i, 3) = z[i].real();
      entries(i, 2) = imag_z[i];
      entries(i, 4) = z[i].imag();
    }

    for (int i = 0; i < x.size(); ++i) {
      EXPECT_NEAR(real_z[i], z[i].real(), 1e-6) << "\n" << entries;
      EXPECT_NEAR(imag_z[i], z[i].imag(), 1e-6) << "\n" << entries;
    }
  }

}  // namespace
