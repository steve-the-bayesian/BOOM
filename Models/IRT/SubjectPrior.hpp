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
#ifndef BOOM_SUBJECT_PRIOR_HPP
#define BOOM_SUBJECT_PRIOR_HPP

#include "Models/IRT/Subject.hpp"
#include "Models/Policies/CompositeParamPolicy.hpp"
#include "Models/Policies/IID_DataPolicy.hpp"
#include "Models/Policies/PriorPolicy.hpp"

namespace BOOM {
  class MvnModel;
  class MvRegModel;
  namespace IRT {

    class SubjectPrior : virtual public Model {
     public:
      SubjectPrior *clone() const override = 0;
      virtual double pdf(const Ptr<Data> &dp, bool logsc) const = 0;
      virtual double pdf(const Ptr<Subject> &subject, bool logsc) const = 0;
      virtual Vector mean(const Ptr<Subject> &subject) const = 0;
      virtual SpdMatrix siginv() const = 0;
      void add_data(const Ptr<Data> &dp) override = 0;
      virtual void add_data(const Ptr<Subject> &subject) = 0;
    };
    //------------------------------------------------------------
    class MvnSubjectPrior : public SubjectPrior,
                            public CompositeParamPolicy,
                            public IID_DataPolicy<Subject>,
                            public PriorPolicy {
     public:
      explicit MvnSubjectPrior(const Ptr<MvnModel> &Mvn);
      MvnSubjectPrior(const MvnSubjectPrior &rhs);
      MvnSubjectPrior *clone() const override;

      double pdf(const Ptr<Data> &, bool logsc) const override;
      double pdf(const Ptr<Subject> &, bool logsc) const override;
      virtual void initialize_params();
      void clear_data() override;
      void add_data(const Ptr<Data> &) override;
      void add_data(const Ptr<Subject> &) override;
      Vector mean(const Ptr<Subject> &) const override;
      SpdMatrix siginv() const override;

     private:
      Ptr<MvnModel> mvn;
    };
    //------------------------------------------------------------
    class MvRegSubjectPrior : public SubjectPrior {
     public:
      explicit MvRegSubjectPrior(const Ptr<MvRegModel> &mvr);
      MvRegSubjectPrior(const MvRegSubjectPrior &rhs);
      MvRegSubjectPrior *clone() const override;

      double pdf(const Ptr<Data> &, bool logsc) const override;
      double pdf(const Ptr<Subject> &, bool logsc) const override;
      virtual void initialize_params();
      void add_data(const Ptr<Data> &) override;
      void add_data(const Ptr<Subject> &) override;
      Vector mean(const Ptr<Subject> &) const override;
      SpdMatrix siginv() const override;

     private:
      Ptr<MvRegModel> mvreg_;
    };
  }  // namespace IRT
}  // namespace BOOM
#endif  // BOOM_SUBJECT_PRIOR_HPP
