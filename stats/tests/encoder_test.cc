#include "gtest/gtest.h"

#include "stats/Encoders.hpp"
#include "stats/DataTable.hpp"
#include "LinAlg/Selector.hpp"
#include "distributions.hpp"

#include "test_utils/test_utils.hpp"
#include <cmath>
#include <fstream>

namespace {
  using namespace BOOM;
  using std::endl;
  using std::cout;

  class EncoderTest : public ::testing::Test {
   protected:
    EncoderTest() {
      GlobalRng::rng.seed(8675309);
      colors_.reset(new CatKey({"red", "blue", "green"}));
      sizes_.reset(new CatKey({"xs", "small", "med", "large"}));
    }
    Ptr<CatKey> colors_;
    Ptr<CatKey> sizes_;
  };

  TEST_F(EncoderTest, EffectsEncoderTest) {
    EffectsEncoder encoder("Color", colors_);
    LabeledCategoricalData red("red", colors_);
    Vector enc = encoder.encode(red);
    EXPECT_TRUE(VectorEquals(enc, Vector{1, 0}));

    LabeledCategoricalData blue("blue", colors_);
    enc = encoder.encode(blue);
    EXPECT_TRUE(VectorEquals(enc, Vector{0, 1}));

    LabeledCategoricalData green("green", colors_);
    enc = encoder.encode(green);
    EXPECT_TRUE(VectorEquals(enc, Vector{-1, -1}));
  }

  // ---------------------------------------------------------------------------
  // Helper: build a DataTable with a single numeric column named "x".
  DataTable numeric_table(const Vector &values) {
    DataTable table;
    table.append_variable(values, "x");
    return table;
  }

  // Helper: build a MixedMultivariateData row with a single numeric field.
  Ptr<MixedMultivariateData> numeric_row(double value) {
    NEW(MixedMultivariateData, row)();
    row->add_numeric(new DoubleData(value), "x");
    return row;
  }

  // ---------------------------------------------------------------------------
  class TransformationEncoderTest : public ::testing::Test {
   protected:
    TransformationEncoderTest() {
      GlobalRng::rng.seed(8675309);
      values_ = {1.0, 2.0, 4.0, 9.0};
      table_ = numeric_table(values_);
    }
    Vector values_;
    DataTable table_;
  };

  TEST_F(TransformationEncoderTest, IdentityTransform) {
    TransformationEncoder enc("x", "identity");
    EXPECT_EQ(enc.transform_name(), "identity");
    EXPECT_EQ(enc.dim(), 1);

    auto names = enc.encoded_variable_names();
    EXPECT_EQ(names.size(), 1u);
    EXPECT_EQ(names[0], "identity(x)");

    Matrix out = enc.encode_dataset(table_);
    EXPECT_EQ(out.nrow(), 4);
    EXPECT_EQ(out.ncol(), 1);
    for (int i = 0; i < 4; ++i) {
      EXPECT_DOUBLE_EQ(out(i, 0), values_[i]);
    }
  }

  TEST_F(TransformationEncoderTest, Log1pTransform) {
    TransformationEncoder enc("x", "log1p");

    Matrix out = enc.encode_dataset(table_);
    for (int i = 0; i < 4; ++i) {
      EXPECT_DOUBLE_EQ(out(i, 0), std::log1p(values_[i]));
    }
  }

  TEST_F(TransformationEncoderTest, SqrtTransform) {
    TransformationEncoder enc("x", "sqrt");

    Matrix out = enc.encode_dataset(table_);
    for (int i = 0; i < 4; ++i) {
      EXPECT_DOUBLE_EQ(out(i, 0), std::sqrt(values_[i]));
    }
  }

  TEST_F(TransformationEncoderTest, EncodeRow) {
    TransformationEncoder enc("x", "log");

    for (double v : values_) {
      Ptr<MixedMultivariateData> row = numeric_row(v);
      Vector result = enc.encode_row(*row);
      EXPECT_EQ(result.size(), 1);
      EXPECT_DOUBLE_EQ(result[0], std::log(v));

      Vector view_result(1);
      enc.encode_row(*row, VectorView(view_result));
      EXPECT_DOUBLE_EQ(view_result[0], std::log(v));
    }
  }

  TEST_F(TransformationEncoderTest, Clone) {
    TransformationEncoder enc("x", "sqrt");
    std::unique_ptr<TransformationEncoder> copy(enc.clone());
    EXPECT_EQ(copy->transform_name(), "sqrt");
    EXPECT_EQ(copy->variable_name(), "x");

    Matrix out = copy->encode_dataset(table_);
    for (int i = 0; i < 4; ++i) {
      EXPECT_DOUBLE_EQ(out(i, 0), std::sqrt(values_[i]));
    }
  }

  TEST_F(TransformationEncoderTest, UserDefinedFunctor) {
    // A custom transform not in the built-in registry.
    auto cube = [](double x) { return x * x * x; };
    TransformationEncoder enc("x", "cube", cube);
    EXPECT_EQ(enc.transform_name(), "cube");

    Matrix out = enc.encode_dataset(table_);
    for (int i = 0; i < 4; ++i) {
      EXPECT_DOUBLE_EQ(out(i, 0), values_[i] * values_[i] * values_[i]);
    }
  }

  TEST_F(TransformationEncoderTest, RegisterCustomTransform) {
    TransformationEncoder::register_transform(
        "reciprocal", [](double x) { return 1.0 / x; });
    EXPECT_TRUE(TransformationEncoder::is_registered("reciprocal"));

    TransformationEncoder enc("x", "reciprocal");
    Matrix out = enc.encode_dataset(table_);
    for (int i = 0; i < 4; ++i) {
      EXPECT_DOUBLE_EQ(out(i, 0), 1.0 / values_[i]);
    }
  }

  TEST_F(TransformationEncoderTest, UnknownTransformThrows) {
    EXPECT_FALSE(TransformationEncoder::is_registered("no_such_transform"));
    EXPECT_THROW(
        TransformationEncoder("x", "no_such_transform"),
        std::exception);
  }

  TEST_F(TransformationEncoderTest, IsRegisteredForBuiltins) {
    for (const std::string &name :
         {"identity", "log", "log1p", "log2", "log10",
          "sqrt", "exp", "expm1", "abs", "square"}) {
      EXPECT_TRUE(TransformationEncoder::is_registered(name)) << name;
    }
  }

}  // namespace
