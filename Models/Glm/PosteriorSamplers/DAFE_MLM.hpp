// Copyright 2018 Google LLC. All Rights Reserved.
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
#ifndef BOOM_DAFE_MLM_HPP
#define BOOM_DAFE_MLM_HPP

#include "LinAlg/Matrix.hpp"
#include "LinAlg/SpdMatrix.hpp"
#include "LinAlg/Vector.hpp"
#include "Models/ModelTypes.hpp"
#include "Models/MvnModel.hpp"
#include "Models/PosteriorSamplers/PosteriorSampler.hpp"
#include "Samplers/MH_Proposals.hpp"
#include "Samplers/MetropolisHastings.hpp"

namespace BOOM {

  class MultinomialLogitModel;

  class DafeMlmBase : public PosteriorSampler {
   public:
    // Args:
    //   mod:  The model.
    //   SubjectPri:  Each subject beta has this prior.
    //   ChoicePri:  Prior for the choice coefficients.
    //   draw_b0:  Indicates whether the reference class should be drawn.
    //   seeding_rng: RNG for base class.
    DafeMlmBase(MultinomialLogitModel *mod, const Ptr<MvnModel> &SubjectPri,
                const Ptr<MvnModel> &ChoicePri, bool draw_b0 = false,
                RNG &seeding_rng = GlobalRng::rng);
    double logpri() const override;
    const SpdMatrix &xtx_subject() const;
    const SpdMatrix &xtx_choice() const;
    uint mlo() const { return mlo_; }

   protected:
    Ptr<MvnModel> subject_pri() const;
    Ptr<MvnModel> choice_pri() const;

   private:
    MultinomialLogitModel *mlm_;
    Ptr<MvnModel> subject_pri_;
    Ptr<MvnModel> choice_pri_;

    SpdMatrix xtx_subject_;
    SpdMatrix xtx_choice_;
    uint mlo_;

    void compute_xtx();
  };
  //------------------------------------------------------------
  class DafeMlm : public DafeMlmBase {
   public:
    // Args:
    //   mod:  The model.
    //   SubjectPri:  Each subject beta has this prior.
    //   ChoicePri:  Prior for the choice coefficients.
    //   tdf:  Degrees of freedom for the MH proposal distribution.
    //   draw_b0:  Indicates whether the reference class should be drawn.
    DafeMlm(MultinomialLogitModel *mod, const Ptr<MvnModel> &SubjectPri,
            const Ptr<MvnModel> &ChoicePri, double tdf, bool draw_b0 = false);
    void draw() override;
    void draw_choice();
    void draw_subject(uint i);
    void impute_latent_data();

   private:
    MultinomialLogitModel *mlm_;
    const double mu;     //( -0.577215664902), negative Euler's constant
    const double sigsq;  // (1.64493406685); pi^2/6
    std::vector<Ptr<MetropolisHastings> > subject_samplers_;
    std::vector<Ptr<MvtIndepProposal> > subject_proposals_;

    Ptr<MetropolisHastings> choice_sampler_;
    Ptr<MvtIndepProposal> choice_proposal_;
    Vector Ominv_mu_subject;
    Vector Ominv_mu_choice;
    Matrix U;  // latent data
    std::vector<Vector> xtu_subject;
    Vector xtu_choice;
  };

  //------------------------------------------------------------
  class DafeRMlm : public DafeMlmBase {
   public:
    // Args:
    //   mod:  The model.
    //   SubjectPri:  Each subject beta has this prior.
    //   ChoicePri:  Prior for the choice coefficients.
    //   tdf:  Degrees of freedom for the MH proposal distribution.
    DafeRMlm(MultinomialLogitModel *mod, const Ptr<MvnModel> &SubjectPri,
             const Ptr<MvnModel> &ChoicePri, double tdf);
    void draw() override;
    void draw_choice();
    void draw_subject(uint i);

   private:
    MultinomialLogitModel *mlm_;
    std::vector<Ptr<MetropolisHastings> > subject_samplers_;
    std::vector<Ptr<MvtRwmProposal> > subject_proposals_;
    Ptr<MetropolisHastings> choice_sampler_;
    Ptr<MvtRwmProposal> choice_proposal_;
  };
}  // namespace BOOM

#endif  // BOOM_DAFE_MLM_HPP
