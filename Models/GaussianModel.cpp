/*
  Copyright (C) 2005 Steven L. Scott

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
#include <Models/GaussianModel.hpp>
#include <Models/GaussianModelGivenSigma.hpp>
#include <Models/GammaModel.hpp>
#include <Models/PosteriorSamplers/PosteriorSampler.hpp>
#include <Models/PosteriorSamplers/GaussianConjSampler.hpp>

#include <cpputil/math_utils.hpp>
#include <distributions.hpp>
#include <cmath>
#include <typeinfo>

namespace BOOM{
  typedef GaussianConjSampler GCS;
  typedef GaussianModel GM;

  GaussianModel::GaussianModel(double mean, double sd)
    : Model(),
      ParamPolicy(new UnivParams(mean), new UnivParams(sd*sd))
  { }

  GaussianModel::GaussianModel(const std::vector<double> &v)
      : GaussianModelBase(v),
        ParamPolicy(new UnivParams(0), new UnivParams(1))
  {
    mle();
  }

  GaussianModel::GaussianModel(const GaussianModel &rhs)
    : Model(rhs),
      GaussianModelBase(rhs),
      ParamPolicy(rhs),
      PriorPolicy(rhs)
  {}

  GM * GM::clone()const{return new GM(*this);}

  Ptr<UnivParams> GM::Mu_prm(){ return prm1(); }
  Ptr<UnivParams> GM::Sigsq_prm(){ return prm2(); }
  const Ptr<UnivParams> GM::Mu_prm()const{ return prm1(); }
  const Ptr<UnivParams> GM::Sigsq_prm()const{ return prm2(); }


  void GM::set_params(double mu, double sigsq){
    set_mu(mu); set_sigsq(sigsq); }
  void GM::set_mu(double m){ Mu_prm()->set(m); }
  void GM::set_sigsq(double s){ Sigsq_prm()->set(s); }


  double GM::mu()const{return Mu_prm()->value();}
  double GM::sigsq()const{return Sigsq_prm()->value();}
  double GM::sigma()const{return sqrt(sigsq());}

  void GaussianModel::mle(){
    double n = suf()->n();
    if(n==0){
      set_params(0,1);
      return;
    }

    double m=ybar();
    if(n==1){
      set_params(ybar(), 1.0);
      return;
    }
    double v = sample_var()*(n-1)/n;
    set_params(m,v);
  }

  double GaussianModel::Loglike(const Vector &mu_sigsq,
                                Vector &g,
                                Matrix &h,
                                uint nd) const {
    double sigsq = mu_sigsq[1];
    if(sigsq<0) return BOOM::negative_infinity();

    double mu = mu_sigsq[0];
    const double log2pi = 1.8378770664093453;
    double n = suf()->n();
    double sumsq = suf()->sumsq();
    double sum = suf()->sum();
    double SS = (sumsq + ( -2*sum + n*mu)*mu);
    double ans = -0.5*(n*(log2pi + log(sigsq)) + SS/sigsq);

    if(nd>0){
      double sigsq_sq = sigsq*sigsq;
      g[0] = (sum-n*mu)/sigsq;
      g[1] = -0.5*n/sigsq + 0.5*SS/sigsq_sq;
      if(nd>1){
        h(0,0) = n/sigsq;
        h(1,0) = h(0,1) = -(sum-n*mu)/sigsq_sq;
        h(1,1) = (n/2 - SS/sigsq)/sigsq_sq;}}
    return ans;
  }

  void GM::set_conjugate_prior(double mu0, double kappa,
                               double df, double sigma_guess){
    double sum_of_squares = pow(sigma_guess, 2) * df;
    NEW(GammaModel, siginv_prior)(df / 2, sum_of_squares / 2);
    NEW(GaussianModelGivenSigma, mu_prior)(Sigsq_prm(), mu0, kappa);
    NEW(GaussianConjSampler, sampler)(this, mu_prior, siginv_prior);
    set_method(sampler);
  }

}  // namespace BOOM
