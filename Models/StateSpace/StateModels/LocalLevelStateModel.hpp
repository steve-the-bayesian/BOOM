#ifndef BOOM_STATE_SPACE_LOCAL_LEVEL_STATE_MODEL_HPP
#define BOOM_STATE_SPACE_LOCAL_LEVEL_STATE_MODEL_HPP
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

#include <Models/StateSpace/StateModels/StateModel.hpp>
#include <Models/ZeroMeanGaussianModel.hpp>

namespace BOOM{

  class LocalLevelStateModel
      : public StateModel,
        public ZeroMeanGaussianModel
  {
   public:
    LocalLevelStateModel(double sigma=1);
    LocalLevelStateModel(const LocalLevelStateModel &rhs);
    LocalLevelStateModel * clone() const override;
    void observe_state(const ConstVectorView then,
                       const ConstVectorView now,
                       int time_now) override;

    uint state_dimension() const override;
    uint state_error_dimension() const override {return 1;}
    void simulate_state_error(RNG &rng, VectorView eta, int t)const override;
    void simulate_initial_state(RNG &rng, VectorView eta)const override;

    Ptr<SparseMatrixBlock> state_transition_matrix(int t) const override;
    Ptr<SparseMatrixBlock> state_variance_matrix(int t) const override;
    Ptr<SparseMatrixBlock> state_error_expander(int t) const override;
    Ptr<SparseMatrixBlock> state_error_variance(int t) const override;

    SparseVector observation_matrix(int t)const override;

    Vector initial_state_mean()const override;
    SpdMatrix initial_state_variance()const override;

    void set_initial_state_mean(double m);
    void set_initial_state_mean(const Vector & m);
    void set_initial_state_variance(const SpdMatrix &v);
    void set_initial_state_variance(double v);

    void update_complete_data_sufficient_statistics(
        int t,
        const ConstVectorView &state_error_mean,
        const ConstSubMatrix &state_error_variance) override;

    void increment_expected_gradient(
        VectorView gradient,
        int t,
        const ConstVectorView &state_error_mean,
        const ConstSubMatrix &state_error_variance) override;

   private:
    Ptr<IdentityMatrix> state_transition_matrix_;
    Ptr<ConstantMatrixParamView> state_variance_matrix_;
    Vector initial_state_mean_;
    SpdMatrix initial_state_variance_;
  };

}

#endif// BOOM_STATE_SPACE_LOCAL_LEVEL_STATE_MODEL_HPP
