#include <stats/Resampler.hpp>
#include <LinAlg/Vector.hpp>
#include <distributions.hpp>
#include <stdexcept>
#include <numeric>
#include <cpputil/report_error.hpp>

namespace BOOM{

  Resampler::Resampler(int N){
    for(int i=0; i<N; ++i){
      double p = i+1;
      p/=N;
      cdf[p] = i;
    }
  }

  Resampler::Resampler(const Vector &probs, bool normalize){
    setup_cdf(probs, normalize);
  }

  std::vector<int> Resampler::operator()(
      int number_of_draws,
      RNG &rng) const {
    std::vector<int> ans(number_of_draws);
    for(int i = 0; i < number_of_draws; ++i){
      ans[i] = cdf.lower_bound(runif_mt(rng))->second;
    }
    return ans;
  }

  void Resampler::set_probs(const Vector &probs, bool normalize){
    cdf.clear();
    setup_cdf(probs, normalize);
  }

  void Resampler::setup_cdf(const Vector &probs, bool normalize){
    int N = probs.size();
    double nc = 1.0;
    if(normalize) {
      nc = sum(probs);
    }
    double p(0);
    for(int i=0; i<N; ++i){
      double p0 = probs[i]/nc;
      if(p0<0) report_error("negative prob");
      p+= p0;
      cdf[p] = i;
    }
  }

  int Resampler::dimension()const{ return cdf.size();}

}
