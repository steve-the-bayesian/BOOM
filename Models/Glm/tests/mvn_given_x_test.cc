#include "gtest/gtest.h"

#include "Models/Glm/MvnGivenX.hpp"
#include "LinAlg/DiagonalMatrix.hpp"
#include "distributions.hpp"

#include "test_utils/test_utils.hpp"
#include <fstream>

namespace {
  using namespace BOOM;
  using std::endl;
  using std::cout;

  class MvnGivenXTest : public ::testing::Test {
   protected:
    MvnGivenXTest() {
      GlobalRng::rng.seed(8675309);
      sample_size_ = 10;
      xdim_ = 3;
      mu_ = Vector(xdim_);
      mu_.randomize();
      diagonal_ = Vector(xdim_);
      diagonal_.randomize();
      X_ = Matrix(sample_size_, xdim_);
      X_.randomize();
      diagonal_weight_ = .7;

      weights_ = Vector(sample_size_);
      weights_.randomize();

      xtwx_ = X_.transpose() * DiagonalMatrix(weights_) * X_ / sum(weights_);
    }

    int sample_size_;
    int xdim_;
    Vector mu_;
    Vector diagonal_;
    Matrix X_;
    double diagonal_weight_;
    Vector weights_;
    SpdMatrix xtwx_;
  };

  TEST_F(MvnGivenXTest, DirectSansDiagonal) {
    double kappa = 1.8;

    MvnGivenX model(new VectorParams(mu_),
                    new UnivParams(kappa),
                    Vector(),
                    diagonal_weight_);
    EXPECT_TRUE(model.diagonal().empty());
    for (int i = 0; i < nrow(X_); ++i) model.add_x(X_.row(i), weights_[i]);

    DiagonalMatrix D(xtwx_.diag());
    SpdMatrix precision = diagonal_weight_ * D + (1 - diagonal_weight_) * xtwx_;
    precision *= kappa;

    EXPECT_TRUE(MatrixEquals(precision, model.siginv()))
        << "direct precision = " << std::endl
        << precision << std::endl
        << "from class: " << std::endl
        << model.siginv();
  }

  TEST_F(MvnGivenXTest, DirectWithDiagonal) {
    double kappa = 1.4;
    MvnGivenX model(new VectorParams(mu_),
                    new UnivParams(kappa),
                    diagonal_,
                    diagonal_weight_);
    EXPECT_TRUE(VectorEquals(diagonal_, model.diagonal()));
    for (int i = 0; i < nrow(X_); ++i) model.add_x(X_.row(i), weights_[i]);

    DiagonalMatrix D(diagonal_);
    SpdMatrix precision = diagonal_weight_ * D + (1 - diagonal_weight_) * xtwx_;
    precision *= kappa;

    EXPECT_TRUE(MatrixEquals(precision, model.siginv()))
        << "direct precision with fixed prior diagonal = " << std::endl
        << precision << std::endl
        << "from class: " << std::endl
        << model.siginv();
  }

  TEST_F(MvnGivenXTest, RegSufNoDiagonal) {
    double kappa = 1.2;
    NEW(NeRegSuf, suf)(xdim_);
    MvnGivenXRegSuf model(new VectorParams(mu_),
                          new UnivParams(kappa),
                          Vector(),
                          diagonal_weight_,
                          suf);
    for (int i = 0; i < X_.nrow(); ++i) {
      suf->add_mixture_data(0, ConstVectorView(X_.row(i)), 1.0);
    }

    SpdMatrix xtx = suf->xtx() / suf->n();
    DiagonalMatrix D(xtx.diag());
    SpdMatrix precision = diagonal_weight_ * D + (1 - diagonal_weight_) * xtx;
    precision *= kappa;

    EXPECT_TRUE(MatrixEquals(precision, model.siginv()))
        << "reg suf precision = " << std::endl
        << precision << std::endl
        << "from class: " << std::endl
        << model.siginv();
  }

  TEST_F(MvnGivenXTest, RegSufWithDiagonal) {
    double kappa = 1.2;
    NEW(NeRegSuf, suf)(xdim_);
    MvnGivenXRegSuf model(new VectorParams(mu_),
                          new UnivParams(kappa),
                          diagonal_,
                          diagonal_weight_,
                          suf);
    for (int i = 0; i < X_.nrow(); ++i) {
      suf->add_mixture_data(0, ConstVectorView(X_.row(i)), 1.0);
    }

    SpdMatrix xtx = suf->xtx() / suf->n();
    DiagonalMatrix D(diagonal_);
    SpdMatrix precision = diagonal_weight_ * D + (1 - diagonal_weight_) * xtx;
    precision *= kappa;

    EXPECT_TRUE(MatrixEquals(precision, model.siginv()))
        << "reg suf precision with fixed diagonal = " << std::endl
        << precision << std::endl
        << "from class: " << std::endl
        << model.siginv();
  }

  TEST_F(MvnGivenXTest, WeightedRegSufSansDiagonal) {
    double kappa = 1.2;
    NEW(WeightedRegSuf, suf)(xdim_);
    MvnGivenXWeightedRegSuf model(
        new VectorParams(mu_),
        new UnivParams(kappa),
        Vector(),
        diagonal_weight_,
        suf);
    for (int i = 0; i < X_.nrow(); ++i) {
      suf->add_data(X_.row(i), 0, weights_[i]);
    }

    EXPECT_NEAR(suf->sumw(), sum(weights_), 1e-4)
        << "sum of weights is incorrect.";
    
    EXPECT_TRUE(MatrixEquals(xtwx_, suf->xtx() / suf->sumw()))
        << "Sufficient statistics don't match direct calculation."
        << std::endl
        << "suf: " << std::endl << suf->xtx() / suf->sumw()
        << std::endl
        << "direct: \n"
        << xtwx_;
    
    DiagonalMatrix D(xtwx_.diag());
    SpdMatrix precision = diagonal_weight_ * D + (1 - diagonal_weight_) * xtwx_;
    precision *= kappa;

    EXPECT_TRUE(MatrixEquals(precision, model.siginv()))
        << "weighted reg suf precision = " << std::endl
        << precision << std::endl
        << "from class: " << std::endl
        << model.siginv();
  }


  
  
}  // namespace
