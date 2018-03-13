#include <Models/HMM/HmmDataImputer.hpp>
#include <Models/HMM/HmmFilter.hpp>
#include <distributions.hpp>
#include <fstream>

namespace BOOM{
typedef HmmDataImputer HDI;

inline unsigned long getseed(){
  double u = runif() * std::numeric_limits<unsigned long>::max();
  unsigned long ans(lround(u));
  return ans;
}

HDI::HmmDataImputer(HiddenMarkovModel * hmm, uint id, uint nworkers)
    : id_(id),
      nworkers_(nworkers),
      mark_(new MarkovModel(hmm->state_space_size())),
      eng(getseed())
{
  eng.seed();
  uint S = hmm->state_space_size();
  for(uint s=0; s<S; ++s){
    Ptr<MixtureComponent> m(hmm->mixture_component(s)->clone());
    mix_.push_back(m);
  }
  filter_ = new HmmFilter(mix_, mark_);
}
//----------------------------------------------------------------------
void HDI::operator()(){
  clear_client_data();
  uint ns = dat_.size();
  loglike_ = 0;
  for(uint i= 0; i<ns; ++i){
    const std::vector<Ptr<Data> > &ts(*dat_[i]);
    loglike_ += filter_->fwd(ts);
    filter_->bkwd_sampling_mt(ts, eng);
  }
}
double HDI::loglike()const{return loglike_;}

//----------------------------------------------------------------------
Ptr<MarkovModel> HDI::mark(){return mark_;}
//----------------------------------------------------------------------
Ptr<MixtureComponent> HDI::models(uint s){return mix_[s];}
//----------------------------------------------------------------------
void HDI::clear_client_data(){
  mark_->clear_data();
  uint S = mix_.size();
  for(uint s=0; s<S; ++s) mix_[s]->clear_data();
}

//----------------------------------------------------------------------

void HDI::setup(HiddenMarkovModel *hmm){
  clear_client_data();
  uint ns = hmm->nseries();
  dat_.clear();
  dat_.reserve(1 + ns/nworkers_);
  for(uint i=id_; i<ns; i+= nworkers_){
    TimeSeries<Data> * ts = &(hmm->dat(i));
    dat_.push_back(ts);
  }

  Vector theta = hmm->mark()->vectorize_params();
  mark_->unvectorize_params(theta);

  uint S = hmm->state_space_size();
  for(uint s=0; s<S; ++s){
    theta = hmm->mixture_component(s)->vectorize_params();
    mix_[s]->unvectorize_params(theta);
  }


}
}
