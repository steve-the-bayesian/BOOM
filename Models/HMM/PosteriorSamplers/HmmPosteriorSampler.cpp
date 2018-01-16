/*
  Copyright (C) 2008 Steven L. Scott

  This library is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License as published by the Free Software Foundation; either
  version 2.1 of the License, or (at your option) any later version.

  This library is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA
*/

#include <Models/HMM/PosteriorSamplers/HmmPosteriorSampler.hpp>
#include <Models/HMM/HmmFilter.hpp>

#ifndef NO_BOOST_THREADS
#include <boost/thread.hpp>
#include <boost/ref.hpp>
#endif

namespace BOOM{

typedef HmmPosteriorSampler HS;

  HS::HmmPosteriorSampler(HiddenMarkovModel *hmm, RNG &seeding_rng)
      : PosteriorSampler(seeding_rng),
#ifndef NO_BOOST_THREADS
        hmm_(hmm),
        use_threads_(false)
#else
        hmm_(hmm)
#endif
  {}

  void HS::draw(){
    hmm_->mark()->sample_posterior();
    draw_mixture_components();
    // by drawing latent data at the end, the log likelihood stored
    // int the model matches the current set of parameters.
    hmm_->impute_latent_data();
  }

  double HS::logpri()const{
    double ans = hmm_->mark()->logpri();
    std::vector<Ptr<MixtureComponent> > mix = hmm_->mixture_components();
    uint S = mix.size();
    for(uint s=0; s<S; ++s) ans += mix[s]->logpri();
    return ans;
  }

  void HS::draw_mixture_components(){
    std::vector<Ptr<MixtureComponent> > mix = hmm_->mixture_components();
    uint S = mix.size();

#ifndef NO_BOOST_THREADS
    if(use_threads_){
      if(workers_.size()!=S) use_threads(true);
      boost::thread_group tg;
      for(uint s=0; s<S; ++s)
        tg.add_thread(new boost::thread(boost::ref(*workers_[s])));
      tg.join_all();
    }else
#endif
    {
      for(uint s=0; s<S; ++s) mix[s]->sample_posterior();
    }
  }

  void HS::use_threads(bool yn){
#ifndef NO_BOOST_THREADS
    use_threads_ = yn;
    if(!use_threads_) return;
    std::vector<Ptr<MixtureComponent> > mix = hmm_->mixture_components();
    uint S = mix.size();
    workers_.clear();
    for(uint s=0; s<S; ++s){
      std::shared_ptr<MixtureComponentSampler>
          worker(new MixtureComponentSampler(mix[s].get()));
      workers_.push_back(worker);
    }
#endif
  }
}
