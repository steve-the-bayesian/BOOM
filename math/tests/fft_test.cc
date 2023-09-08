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

  /*
    format.vector <- function(x, z) {
      s1 <- "Vector x = {"
      s2 <- paste(x, collapse = ", ")
      s3 <- "};"
      vx <- paste0(s1, s2, s3)

      s1 <- "Vector real_z = {"
      s2 <- paste(Re(z), collapse = ", ")
      vr <- paste0(s1, s2, s3)

      s1 <- "Vector imag_z = {"
      s2 <- paste(Im(z), collapse = ", ")
      vi <- paste0(s1, s2, s3)
      return(c(vx, vr, vi))
    }
   */
  // Check that the results of an FFT match what we get from R.
  //
  // Args:
  //   x: A sequence of real numbers.  This might be a call to rnorm(k) for some
  //     value of k.
  //   real_z:  The real component of R's fft(x).
  //   imag_z:  The imaginary component of R's fft(x).
  //
  // Effects:
  //   A BOOM::FastFourierTransform is created and used to transform x.  The
  //   test passes if the C++ real and imaginary components match the inputs
  //   from R.
  void CheckResults(const Vector &x, const Vector &real_z, const Vector &imag_z) {
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
      ASSERT_NEAR(real_z[i], z[i].real(), 1e-6)
          << "\n"
          << "Error occurred in position " << i << ".\n"
          << entries;
      EXPECT_NEAR(imag_z[i], z[i].imag(), 1e-6) << "\n" << entries;
    }
  }

  // I generated a sequence of random numbers in R and used R's fft to take the
  // transform.
  TEST_F(FFTtest, MatchesR_10) {
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

    /****************************************************************************
  Output from native C implementation matches R.
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
    ****************************************************************************/

    /****************************************************************************
This is what config looks like after the FFT in kiss_fftr (C++ version).
nfft: 5
inverse: 0
factors:  5  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
twiddles:
    1    -0
    0.309017    -0.951057
    -0.809017    -0.587785
    -0.809017    0.587785
    0.309017    0.951057
tmpbuf:
   0.504545   0.211584
   0.407472   -0.247874
   -0.0594952   -0.297532
   -0.251024   0.131235
   0.097572   0.445885
super_twiddles:
   -0.587785   -0.809017
   -0.951057   -0.309017
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
    ****************************************************************************/

    CheckResults(x, real_z, imag_z);
  }

  TEST_F(FFTtest, MatchesR_6) {
    Vector x = {-1.52121661892621, 0.558672898012533, -1.01315960277756, -2.79089580150385, -0.288372457665452, -0.249161439272905};
    Vector real_z = {-5.30413302213345, 2.07520094216896, -3.81610211957837, -0.341364336604998, -3.81610211957837, 2.07520094216896};
    Vector imag_z = {0, -0.0719209781350744, -1.32728913814204, -1.38777878078145e-17, 1.32728913814204, 0.0719209781350744};
    CheckResults(x, real_z, imag_z);
  }

  TEST_F(FFTtest, MatchesR_7) {
    Vector x = {0.721742782281823, -0.602144398993153, 0.224309920585769, 0.0949769797385891, -1.21558370999016, -0.0429372657105602, 1.79445243504823};
    Vector real_z = {0.974816742960538, 2.434407247998, -0.405668699202261, 0.00995281771037013, 0.00995281771037013, -0.405668699202261, 2.434407247998};
    Vector imag_z = {0, 1.04455714186442, 3.47710096312739, -0.0289155376661736, 0.0289155376661736, -3.47710096312739, -1.04455714186442};
    CheckResults(x, real_z, imag_z);
  }

  TEST_F(FFTtest, MatchesR_12) {

    Vector x = {-0.479975219344155, -0.260564283838167, -0.165172767583374,
      -0.822228979064739, 2.04817818851567, 0.642871290141745, 0.966415812111228,
      -0.561648397765429, 1.06880239166258, -0.218342657680286, 0.85443972238985,
      0.0848634738150466};
    Vector real_z = {3.15763857335997, -2.88275029725354, -0.423350496803669,
      0.981322593916394, -2.41001585264692, -2.43774539102901, 5.42773768214363,
      -2.43774539102901, -2.41001585264691, 0.981322593916393,
      -0.423350496803669, -2.88275029725354};
    Vector imag_z = {0, 0.209186354663792, 3.07346850064521, -1.46297825163841,
      -0.388880775873023, 0.139494357851155, 0, -0.139494357851155,
      0.388880775873023, 1.46297825163841, -3.07346850064521, -0.209186354663792};
  }

}  // namespace
