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
#ifndef BOOM_IRT_SUBJECT_SLICE_SAMPLER_HPP
#define BOOM_IRT_SUBJECT_SLICE_SAMPLER_HPP

#include "Models/ParamTypes.hpp"
#include "Models/PosteriorSamplers/PosteriorSampler.hpp"
#include "TargetFun/TargetFun.hpp"

namespace BOOM {
  class SliceSampler;
  class TargetFun;
  namespace IRT {
    class SubjectPrior;
    class Subject;

    class SubjectTF : public BOOM::TargetFun {
     public:
      SubjectTF(const Ptr<Subject> &subject, const Ptr<SubjectPrior> &prior);
      double operator()(const Vector &v) const override;

     private:
      Ptr<Subject> sub;
      Ptr<SubjectPrior> pri;
      mutable Ptr<VectorParams> prms;
      mutable Vector wsp;
    };

    class SubjectSliceSampler : public PosteriorSampler {
     public:
      SubjectSliceSampler(const Ptr<Subject> &subject,
                          const Ptr<SubjectPrior> &prior,
                          RNG &seeding_rng = GlobalRng::rng);
      SubjectSliceSampler *clone() const;
      void draw() override;
      double logpri() const override;

     private:
      Ptr<Subject> sub;
      Ptr<SubjectPrior> pri;
      SubjectTF target;
      Ptr<SliceSampler> sam;
      Vector Theta;
    };
  }  // namespace IRT
}  // namespace BOOM
#endif  // BOOM_IRT_SUBJECT_SLICE_SAMPLER_HPP
