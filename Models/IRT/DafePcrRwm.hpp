/*
  Copyright (C) 2006 Steven L. Scott

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
#ifndef BOOM_DAFE_PCR_RWM_HPP
#define BOOM_DAFE_PCR_RWM_HPP

#include "uint.hpp"
#include "Models/IRT/Subject.hpp"
#include "Models/ModelTypes.hpp"
#include "Models/VectorModel.hpp"
#include "Samplers/MetropolisHastings.hpp"

namespace BOOM {
  class MvnModel;
  class MH_Proposal;
  namespace IRT {
    class PartialCreditModel;
    class DafePcrRwmItemSampler : public PosteriorSampler {
     public:
      DafePcrRwmItemSampler(const Ptr<PartialCreditModel> &,
                            const Ptr<MvnModel> &Prior, double Tdf,
                            RNG &seeding_rng = GlobalRng::rng);
      void draw() override;
      double logpri() const override;

     private:
      Ptr<PartialCreditModel> mod;
      Ptr<MvnModel> prior;
      Ptr<MetropolisHastings> sampler;
      Ptr<MvtRwmProposal> prop;
      const double sigsq;  //  = pi^2/6 = 1.64493406684
      SpdMatrix xtx, ivar;
      Vector b;

      void get_moments();
      void accumulate_moments(const Ptr<Subject> &);
    };

    //======================================================================
    class DafePcrRwmSubjectSampler : public PosteriorSampler {
     public:
      DafePcrRwmSubjectSampler(const Ptr<Subject> &,
                               const Ptr<SubjectPrior> &Prior, double Tdf,
                               RNG &seeding_rng = GlobalRng::rng);
      void draw() override;
      double logpri() const override;

     private:
      Ptr<Subject> sub;
      Ptr<SubjectPrior> prior;
      Ptr<MetropolisHastings> sampler;
      Ptr<MvtRwmProposal> prop;

      const double sigsq;  //  = pi^2/6 = 1.64493406684
      SpdMatrix ivar;
      Vector Theta;

      void get_moments();
      void accumulate_moments(std::pair<Ptr<Item>, Response>);
    };
  }  // namespace IRT
}  // namespace BOOM

#endif  // BOOM_DAFE_PCR_RWM_HPP
