// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2007 Steven L. Scott

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

#ifndef BOOM_HMM_FILTER_HPP
#define BOOM_HMM_FILTER_HPP

#include "LinAlg/Matrix.hpp"
#include "Models/MarkovModel.hpp"
#include "cpputil/Ptr.hpp"
#include "cpputil/RefCounted.hpp"
#include "distributions/rng.hpp"

namespace BOOM {
  class Model;
  class Data;
  class EmMixtureComponent;
  class HiddenMarkovModel;

  class HmmFilter : private RefCounted {
   public:
    friend void intrusive_ptr_add_ref(HmmFilter *d) { d->up_count(); }
    friend void intrusive_ptr_release(HmmFilter *d) {
      d->down_count();
      if (d->ref_count() == 0) delete d;
    }

    HmmFilter(const std::vector<Ptr<MixtureComponent>> &mix,
              const Ptr<MarkovModel> &mark);
    ~HmmFilter() override {}
    uint state_space_size() const;

    double initialize(const Data *);
    double loglike(const std::vector<Ptr<Data>> &);
    double fwd(const std::vector<Ptr<Data>> &);
    void bkwd_sampling(const std::vector<Ptr<Data>> &);
    void bkwd_sampling_mt(const std::vector<Ptr<Data>> &, RNG &rng);
    virtual void allocate(const Ptr<Data> &, uint);
    virtual Vector state_probs(const Ptr<Data> &) const;

    // Return the state vector that was imputed for data during the call to
    // bkwd_sampling or bkwd_sampling_mt.
    std::vector<int> imputed_state(const std::vector<Ptr<Data>> &data) const;
    
   protected:
    std::vector<Ptr<MixtureComponent>> models_;
    std::vector<Matrix> P;
    Vector pi, logp, logpi, one;
    Matrix logQ;
    Ptr<MarkovModel> markov_;
    std::map<std::vector<Ptr<Data>>, std::vector<int>> imputed_state_map_;
  };
  //----------------------------------------------------------------------
  class HmmSavePiFilter : public HmmFilter {
   public:
    HmmSavePiFilter(const std::vector<Ptr<MixtureComponent>> &mix,
                    const Ptr<MarkovModel> &mark,
                    std::map<Ptr<Data>, Vector> &pi_hist);
    void allocate(const Ptr<Data> &dp, uint h) override;
    Vector state_probs(const Ptr<Data> &) const override;

   private:
    std::map<Ptr<Data>, Vector> &pi_hist_;
  };

  //----------------------------------------------------------------------
  class HmmEmFilter : public HmmFilter {
   public:
    HmmEmFilter(const std::vector<Ptr<EmMixtureComponent>> &mix,
                const Ptr<MarkovModel> &mark);
    virtual void bkwd_smoothing(const std::vector<Ptr<Data>> &);

   private:
    std::vector<Ptr<EmMixtureComponent>> em_models_;
  };

}  // namespace BOOM

#endif  // BOOM_HMM_FILTER_HPP
