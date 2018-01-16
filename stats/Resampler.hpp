#ifndef BOOM_RESAMPLER_HPP
#define BOOM_RESAMPLER_HPP
#include <BOOM.hpp>
#include <LinAlg/Vector.hpp>

#include <vector>
#include <map>
#include <algorithm>
#include <distributions/rng.hpp>

namespace BOOM{

  // Efficiently sample with replacement from a discrete distribution.
  //
  // Typical usage:
  // Resampler resample(probs);
  //
  // std::vector<int> index = resample(number_of_draws, rng).
  // (Then use index to subsample things)
  //
  // std::vector<Things> ressampled = resample(unique_things, number_of_draws);
  class Resampler{
  public:
    // Resamples according to an equally weighted distribution of
    // dimension nvals.
    // Equivalent to Resampler(Vector(nvals, 1.0 / nvals), false);
    Resampler(int nvals = 1); // equally weighted [0..nvals-1]

    // Args:
    //   probs:  A discrete distribution (all non-negative elements).
    //   normalize: If true then the probs will be divided by its sum
    //     to ensure proper normalization before being used.  If false
    //     then it is assumed that the normalization has already
    //     occurred prior to construction, so sum(probs) == 1 already.
    Resampler(const Vector &probs, bool normalize=true);

    // Resample from a vector of objects.
    // Args:
    //   source:  The vector of (likely distinct) object to sample from.
    //   number_of_draws: The desired number of draws.  If less than 0
    //     (the default) this will be taken to be source.size().
    //   rng:  The random number generator used to create random subsamples.
    //
    // Returns:
    //   A vector of objects of size number_of_draws.  Draws will be
    //   chosen according to the probabilities passed in at
    //   construction.
    template <class T>
    std::vector<T> operator()(const std::vector<T> &source,
                              int number_of_draws = -1,
                              RNG &rng = GlobalRng::rng) const;

    // Returns a sample, with replacement from [0, ... probs.size() - 1].
    std::vector<int> operator()(int number_of_draws,
                                RNG &rng = GlobalRng::rng)const;

    // Returns the number of categories in the discrete distribution.
    int dimension()const;

    // Reset the Resampler with a new set of probabilities.
    // Equivalent to Resampler that(probs, normalize); swap(*this, that);
    void set_probs(const Vector &probs, bool normalize=true);

  private:
    typedef std::map<double, int> CDF;
    CDF cdf;
    void setup_cdf(const Vector &probs, bool normalize);
  };

  //------------------------------------------------------------

  template <class T>
  std::vector<T> Resampler::operator()(
      const std::vector<T> &things,
      int number_of_draws,
      RNG &rng) const {
    std::vector<int> index = (*this)(number_of_draws);
    std::vector<T> ans;
    ans.reserve(number_of_draws);
    for(int i = 0; i < number_of_draws; ++i) {
      ans[i] = things[index[i]];
    }
    return ans;
  }

  //______________________________________________________________________

  template<class T>
  std::vector<T> resample(const std::vector<T> & things,
                          int number_of_draws,
                          const Vector & probs){
    Vector cdf = cumsum(probs);
    double total = cdf.back();
    if(total<1.0 || total > 1.0) {
      cdf/=total;
      total=1.0;
    }

    Vector u(number_of_draws);
    u.randomize();
    std::sort(u.begin(), u.end());

    std::vector<T> ans;
    ans.reserve(number_of_draws);
    int cursor=0;
    for(int i=0; i<number_of_draws; ++i){
      while(u[i]>cdf[cursor]) ++cursor;
      ans.push_back(things[cursor]);
    }
    return(ans);
  }
}
#endif// BOOM_RESAMPLER_HPP
