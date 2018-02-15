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

#ifndef BOOM_SUBJECT_PRIOR_HPP
#define BOOM_SUBJECT_PRIOR_HPP

#include "IRT.hpp"
#include "Subject.hpp"
#include "cpputil/math_utils.hpp"
#include <models/param_types.hpp>
#include <models/param_policies.hpp>
#include <models/sufstat_policies.hpp>
#include <models/prior_policies.hpp>
#include <models/mvn_model.hpp>
namespace BOOM{
  class mvn_suf;
  namespace IRT{

    class constrained_mvn_params : public corr_params{
      Corr Rinv_;
      double logdet_rinv;
      bool out_of_sync;
      Corr & value(){return corr_params::value();}
      bool sync();
    public:
      constrained_mvn_params(uint ThetaDim);
      constrained_mvn_params(const SpdMatrix &R);
      constrained_mvn_params(const constrained_mvn_params &rhs);
      constrained_mvn_params * clone()const;

      constrained_mvn_params& operator=(const Corr &);
      constrained_mvn_params & set_Rinv(const SpdMatrix & rinv);

      const Corr & R()const;
      const Corr & Rinv()const;
      const double & ldri()const;

      const double * unvectorize(const double *dp); //
      const double * unvectorize(const double *dp, bool &ok); //
      bool problem()const{return out_of_sync;}
      int io(const string &dname, const string &fname, IO io_prm,
              const string &sfx);

    };


    class SubjectPrior
      : public default_param_policy<constrained_mvn_params>,
        public default_sufstat_policy<mvn_suf>,
        public default_prior_policy,
        public basic_prior_details<Subject>
    {
      //  theta[i] ~ N(0, R)
    public:
      SubjectPrior(uint ThetaDim);
      SubjectPrior(const SpdMatrix &R);
      SubjectPrior(const SubjectPrior &rhs);
      SubjectPrior * clone()const;

      double pdf(const Ptr<data> &, bool logscale)const;
      const SpdMatrix & Rinv()const;
      const SpdMatrix & R()const;
      const double & ldri()const;
      double loglike(const Vector &x)const;
      void draw_children_params();
      void draw_theta_slice();
      void draw_theta_MH();

    };

    class Subject_Regression_Prior{
      // each subject with covariates X has theta\sim N(X*Beta, R);
      // part of beta must be constrained to ensure identifiability
    public:
    };
    //======================================================================
    class UniformCorrelationPrior
      : public default_param_policy<null_params>,
        public default_sufstat_policy<null_suf>,
        public default_prior_policy,
        public slice_sampling_prior<SubjectPrior>
    {
      // p(R) \propto 1
    public:
      typedef UniformCorrelationPrior UCP;
      UCP * clone()const;
      double pdf(const Ptr<data> &, bool logscale)const;
    };
  }
}
#endif // BOOM_SUBJECT_PRIOR_HPP
g
