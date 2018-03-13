#include "gtest/gtest.h"
#include "Samplers/AdaptiveGaussianMixtureMhSampler.hpp"
#include "distributions.hpp"
#include <fstream>

namespace {
  using namespace BOOM;
  using std::endl;
  using std::cout;
  
  template <class V1, class V2>
  bool VectorEquals(const V1 &v1, const V2 &v2) {
    Vector v = v1 - v2;
    return v.max_abs() < 1e-8;
  }

  class AdaptiveGaussianMixtureMhSamplerTest : public ::testing::Test {
   protected:
    AdaptiveGaussianMixtureMhSamplerTest() {
      GlobalRng::rng.seed(8675309);
    }
  };


  TEST_F(AdaptiveGaussianMixtureMhSamplerTest, Mvn) {
    class MvnLogDensity {
     public:
      MvnLogDensity(const Vector &mu,
                    const SpdMatrix &Sigma)
          : mu_(mu),
            siginv_(Sigma.inv()),
            ldsi_(siginv_.logdet())
      {}
      
      double operator()(const Vector &y) const {
        return dmvn(y, mu_, siginv_, ldsi_, true);
      }
      
     private:
      Vector mu_;
      SpdMatrix siginv_;
      double ldsi_;
    };

    SpdMatrix Sigma(4);
    Sigma.randomize();
    ASSERT_TRUE(Sigma.is_pos_def())
        << "Sigma matrix is not positive definite.  Choose new values.";
    
    Vector sd = {20, 4, 6, 8};
    SpdMatrix sd_matrix(4);
    sd_matrix.diag() = sd;
    Sigma = sd_matrix * Sigma * sd_matrix;

    Vector mu = {3, 19, -27, 18};
    MvnLogDensity log_density(mu, Sigma);
    
    int niter = 5000;
    Matrix mvn_draws(niter, 4);
    AdaptiveGaussianMixtureMhSampler sampler(log_density);
    sampler.set_shared_variance(SpdMatrix(4, 1.0));

    Vector y(4, 0.0);
    for (int i = 0; i < niter; ++i) {
      cout << "---------------- iteration " << i << " ----------------"  << endl;
      y = sampler.draw(y);
      mvn_draws.row(i) = y;
    }

    std::ofstream mvn_out("mvn_draws.out");
    mvn_out << mvn_draws;

    std::ofstream mvn_direct("mvn_direct.out");
    for (int i = 0; i < niter; ++i) {
      mvn_direct << rmvn(mu, Sigma) << endl;
    }

    cout << "sampler finished with "
         << sampler.number_of_components()
         << " components. " << endl;

    cout << "sorted mixing weights = "
         << rev(sort(sampler.mixing_weights())) << endl;

  }
  
}  // namespace
