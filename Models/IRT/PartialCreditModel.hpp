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
#ifndef BOOM_PARTIAL_CREDIT_MODEL_HPP
#define BOOM_PARTIAL_CREDIT_MODEL_HPP

#include "Models/ConstrainedVectorParams.hpp"
#include "Models/Glm/MultinomialLogitModel.hpp"
#include "Models/IRT/Item.hpp"
#include "Models/IRT/Subject.hpp"
#include "Models/Policies/ParamPolicy_3.hpp"
#include "Models/Policies/PriorPolicy.hpp"
namespace BOOM {
  namespace IRT {

    class PcrBetaConstraint : public VectorConstraint {
     public:
      // Constrains the first and next_to_last elements of b so that (M+1)*b[0]
      // = b[M]. This class is used when PartialCreditModel is kept identified.
      bool check(const Vector &b) const override;
      Vector &impose(Vector &b) const override;
      Vector expand(const Vector &b_min) const override;   // adds in b0
      Vector reduce(const Vector &b_full) const override;  // omits b0
      int minimal_size_reduction() const override {return 1;}
    };

    class PcrDConstraint : public VectorConstraint {
     public:
      bool check(const Vector &d) const override;
      Vector &impose(Vector &d) const override;
      Vector expand(const Vector &d_min) const override;
      Vector reduce(const Vector &d_full) const override;
      int minimal_size_reduction() const override {return 1;}
    };

    class PartialCreditModel
        : public Item,  // knows all subjects assigned to this item
          public ParamPolicy_3<UnivParams, UnivParams,
                               ConstrainedVectorParams>,  // a,b,d
          public PriorPolicy {
      /*------------------------------------------------------------
        An item with maxscore()==M yields log score probabilities = C
        + X*beta where C is a normalizing constant X[0..M, 0..M] is an
        (M+1)x(M+2) matrix and beta[0..M+1] is an M+2 vector as follows
        (for M==4)

        X:                             beta:
        1  0  0  0  0  theta           a*(d0-b)
        0  1  0  0  0  2*theta         a*(d0+d1-2b)
        0  0  1  0  0  3*theta         a*(d0+d1+d2-3b)
        0  0  0  1  0  4*theta         a*(d0+d1+d2+d3-4b)
        0  0  0  0  1  5*theta         a*(-5b)  // sum of d's is 0
                                       a

        The redundant information is stored in d, so d[0] = 0 and
        d.sum()=0.  Among other things this makes parameter expansion
        easy.

        ------------------------------------------------------------*/

     public:
      PartialCreditModel(const std::string &Id, uint Mscore, uint which_sub,
                         uint Nscales, const std::string &Name = "",
                         bool id_d0 = true);
      PartialCreditModel(const std::string &Id, uint Mscore, uint which_sub,
                         uint Nscales, double a, double b, const Vector &d,
                         const std::string &Name = "", bool id_d0 = true);
      PartialCreditModel(const PartialCreditModel &rhs);
      PartialCreditModel *clone() const override;

      uint which_subscale() const;

      Ptr<UnivParams> A_prm(bool check = true);
      Ptr<UnivParams> B_prm(bool check = true);
      Ptr<ConstrainedVectorParams> D_prm(bool check = true);
      Ptr<ConstrainedVectorParams> Beta_prm(bool check = true);
      const Ptr<UnivParams> A_prm(bool check = true) const;
      const Ptr<UnivParams> B_prm(bool check = true) const;
      const Ptr<ConstrainedVectorParams> D_prm(bool check = true) const;
      const Ptr<ConstrainedVectorParams> Beta_prm(bool check = true) const;
      std::vector<Ptr<Params>> parameter_vector() override;
      const std::vector<Ptr<Params>> parameter_vector() const override;

      double a() const;
      double b() const;
      double d(uint m) const;
      const Vector &d() const;
      void set_a(double a);
      void set_b(double b);
      void set_d(const Vector &d);

      void fix_d0();
      void free_d0();
      bool is_d0_fixed() const;

      void initialize_params();
      void sync_params() const;

      const Vector &beta() const override;  // see note above for dimension
      void set_beta(const Vector &b);

      const Vector &fill_eta(const Vector &Theta) const;  // 0.. maxscore()
      const Matrix &X(const Vector &Theta) const;
      const Matrix &X(double theta) const;

      double response_prob(Response r, const Vector &Theta,
                           bool logsc) const override;
      double response_prob(uint r, const Vector &Theta,
                           bool logsc) const override;

      std::pair<double, double> theta_moments() const;
      // mean and variance of theta's for subjects that were assigned
      // this item

      std::ostream &display_item_params(std::ostream &,
                                   bool decorate = true) const override;

     private:
      // workspace for probability calculations
      mutable Vector b_, eta_;
      mutable Matrix X_;
      bool d0_is_fixed;

      // pointers and flags for keeping track of alternate parameterizations
      mutable Ptr<ConstrainedVectorParams> beta_, d_prm;
      mutable Ptr<UnivParams> a_prm, b_prm;
      mutable bool beta_current, a_current, b_current, d_current;

      void impose_beta_constraint();
      void fill_beta(bool first_time = false) const;
      void fill_abd() const;
      void setup_X();
      void setup_beta();

      uint which_subscale_;

      // the observers watch a, b, and d for changes

      void observe_a() const { beta_current = false; }
      void observe_b() const { beta_current = false; }
      void observe_d() const { beta_current = false; }
      void observe_beta() const { a_current = b_current = d_current = false; }

      void set_abd_current() const;

      // to be called during construction:
      void setup();
      void setup_aliases();
      void set_observers();

      // helper for theta_moments
      void increment_theta_moments(const Ptr<Subject> &, double &m, double &v,
                                   double &n) const;
    };
  }  // namespace IRT
}  // namespace BOOM
#endif  // BOOM_PARTIAL_CREDIT_MODEL_HPP
