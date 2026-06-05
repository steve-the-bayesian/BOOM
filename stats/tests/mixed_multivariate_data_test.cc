#include "gtest/gtest.h"
#include "cpputil/seq.hpp"
#include "LinAlg/Vector.hpp"
#include "stats/Resampler.hpp"
#include "test_utils/test_utils.hpp"
#include "stats/ChiSquareTest.hpp"
#include "stats/FreqDist.hpp"
#include "stats/DataTable.hpp"

namespace {
  using namespace BOOM;
  using std::endl;

  class MixedMultivariateDataTest : public ::testing::Test {
   protected:
    MixedMultivariateDataTest()
        : color_key_(new CatKey({"red", "blue", "green"})),
          shape_key_(new CatKey({"circle", "square", "triangle", "rhombus"}))
    {
      GlobalRng::rng.seed(8675309);
    }

    Ptr<CatKey> color_key_;
    Ptr<CatKey> shape_key_;
  };

  // Checks that MixedMultivariateData can be default-constructed without error.
  TEST_F(MixedMultivariateDataTest, DefaultConstructor) {
    MixedMultivariateData data;
  }

  // Reads .txt and .csv files; checks row/column counts,
  // auto-detected variable types, and column names.
  TEST_F(MixedMultivariateDataTest, Blah) {
    bool header = false;
    std::string path = "stats/tests/autopref.txt";
    DataTable autopref(path, header, "\t");
    /*
      American	34	Male	Married	Large	Family	No
      Japanese	36	Male	Single	Small	Sporty	No
      Japanese	23	Male	Married	Small	Family	No
    */
    EXPECT_EQ(autopref.nobs(), 263);
    EXPECT_EQ(autopref.nvars(), 7);
    EXPECT_EQ(autopref.variable_type(0), VariableType::categorical);
    EXPECT_EQ(autopref.variable_type(1), VariableType::numeric);
    EXPECT_EQ(autopref.variable_type(2), VariableType::categorical);
    EXPECT_EQ(autopref.variable_type(3), VariableType::categorical);
    EXPECT_EQ(autopref.variable_type(4), VariableType::categorical);
    EXPECT_EQ(autopref.variable_type(5), VariableType::categorical);
    EXPECT_EQ(autopref.variable_type(6), VariableType::categorical);
    EXPECT_EQ(autopref.vnames()[0], "V.0");
    EXPECT_EQ(autopref.vnames()[1], "V.1");

    header=true;
    path = "stats/tests/CarsClean.csv";
    DataTable cars(path, header, ",");
    EXPECT_EQ(cars.nobs(), 94);
    EXPECT_EQ(cars.nvars(), 22);
    EXPECT_EQ(cars.vnames()[0], "Make/Model");
    EXPECT_EQ(cars.vnames()[1], "MPGCity");
    EXPECT_EQ(cars.vnames()[21], "GP1000MCity");
  }

  // Checks numeric fields are added and retrieved by index and name,
  // and that numeric_data() returns the correct vector.
  TEST_F(MixedMultivariateDataTest, AddAndAccessNumeric) {
    MixedMultivariateData row;
    row.add_numeric(new DoubleData(3.14), "pi");
    row.add_numeric(new DoubleData(2.72), "e");

    EXPECT_EQ(row.numeric_dim(), 2);
    EXPECT_EQ(row.dim(), 2);
    EXPECT_EQ(row.variable_type(0), VariableType::numeric);
    EXPECT_EQ(row.variable_type(1), VariableType::numeric);
    EXPECT_DOUBLE_EQ(row.numeric(0).value(), 3.14);
    EXPECT_DOUBLE_EQ(row.numeric(1).value(), 2.72);
    EXPECT_DOUBLE_EQ(row.numeric("pi").value(), 3.14);
    EXPECT_DOUBLE_EQ(row.numeric("e").value(), 2.72);

    Vector v = row.numeric_data();
    ASSERT_EQ(v.size(), 2);
    EXPECT_DOUBLE_EQ(v[0], 3.14);
    EXPECT_DOUBLE_EQ(v[1], 2.72);
  }

  // Checks categorical fields are added and retrieved by index and
  // name with correct integer category values.
  TEST_F(MixedMultivariateDataTest, AddAndAccessCategorical) {
    MixedMultivariateData row;
    NEW(LabeledCategoricalData, color)("blue", color_key_);
    NEW(LabeledCategoricalData, shape)("circle", shape_key_);
    row.add_categorical(color, "color");
    row.add_categorical(shape, "shape");

    EXPECT_EQ(row.categorical_dim(), 2);
    EXPECT_EQ(row.dim(), 2);
    EXPECT_EQ(row.variable_type(0), VariableType::categorical);
    EXPECT_EQ(row.variable_type(1), VariableType::categorical);
    EXPECT_EQ(row.categorical(0).value(), color_key_->findstr("blue"));
    EXPECT_EQ(row.categorical(1).value(), shape_key_->findstr("circle"));
    EXPECT_EQ(row.categorical("color").value(), color_key_->findstr("blue"));
    EXPECT_EQ(row.categorical("shape").value(), shape_key_->findstr("circle"));
  }

  // Checks datetime fields are added and retrieved by index and name.
  TEST_F(MixedMultivariateDataTest, AddAndAccessDatetime) {
    MixedMultivariateData row;
    DateTime ts1(12345.6);
    DateTime ts2(98765.4);
    row.add_datetime(new DateTimeData(ts1), "created_at");
    row.add_datetime(new DateTimeData(ts2), "updated_at");

    EXPECT_EQ(row.dim(), 2);
    EXPECT_EQ(row.variable_type(0), VariableType::datetime);
    EXPECT_EQ(row.variable_type(1), VariableType::datetime);
    EXPECT_EQ(row.datetime(0).value(), ts1);
    EXPECT_EQ(row.datetime(1).value(), ts2);
    EXPECT_EQ(row.datetime("created_at").value(), ts1);
    EXPECT_EQ(row.datetime("updated_at").value(), ts2);
  }

  // Checks dim(), numeric_dim(), categorical_dim(), and
  // variable_type() for a row mixing numeric, categorical, datetime.
  TEST_F(MixedMultivariateDataTest, MixedTypeDimensions) {
    MixedMultivariateData row;
    row.add_numeric(new DoubleData(1.0), "x");
    NEW(LabeledCategoricalData, color)("red", color_key_);
    row.add_categorical(color, "color");
    row.add_datetime(new DateTimeData(DateTime(500.0)), "ts");
    row.add_numeric(new DoubleData(2.0), "y");

    EXPECT_EQ(row.dim(), 4);
    EXPECT_EQ(row.numeric_dim(), 2);
    EXPECT_EQ(row.categorical_dim(), 1);
    EXPECT_EQ(row.variable_type(0), VariableType::numeric);
    EXPECT_EQ(row.variable_type(1), VariableType::categorical);
    EXPECT_EQ(row.variable_type(2), VariableType::datetime);
    EXPECT_EQ(row.variable_type(3), VariableType::numeric);
    EXPECT_DOUBLE_EQ(row.numeric(0).value(), 1.0);
    EXPECT_DOUBLE_EQ(row.numeric(3).value(), 2.0);
    EXPECT_EQ(row.datetime(2).value(), DateTime(500.0));
  }

  // Checks the copy constructor produces a deep copy: values match
  // but mutating the copy does not affect the original.
  TEST_F(MixedMultivariateDataTest, CopyConstructor) {
    MixedMultivariateData original;
    original.add_numeric(new DoubleData(42.0), "x");
    NEW(LabeledCategoricalData, color)("green", color_key_);
    original.add_categorical(color, "color");
    original.add_datetime(new DateTimeData(DateTime(999.0)), "ts");

    MixedMultivariateData copy(original);
    EXPECT_EQ(copy.dim(), original.dim());
    EXPECT_DOUBLE_EQ(copy.numeric(0).value(), original.numeric(0).value());
    EXPECT_EQ(copy.categorical(1).value(), original.categorical(1).value());
    EXPECT_EQ(copy.datetime(2).value(), original.datetime(2).value());

    // Mutating copy must not affect original (deep copy).
    copy.mutable_numeric(0)->set(99.0);
    EXPECT_DOUBLE_EQ(original.numeric(0).value(), 42.0);
  }

  // Checks mutable accessors update numeric and datetime values.
  TEST_F(MixedMultivariateDataTest, MutableAccess) {
    MixedMultivariateData row;
    row.add_numeric(new DoubleData(1.0), "x");
    NEW(LabeledCategoricalData, color)("red", color_key_);
    row.add_categorical(color, "color");
    row.add_datetime(new DateTimeData(DateTime(100.0)), "ts");

    row.mutable_numeric(0)->set(99.0);
    EXPECT_DOUBLE_EQ(row.numeric(0).value(), 99.0);

    row.mutable_numeric("x")->set(55.0);
    EXPECT_DOUBLE_EQ(row.numeric("x").value(), 55.0);

    DateTime new_ts(200.0);
    row.mutable_datetime(2)->set(new_ts);
    EXPECT_EQ(row.datetime(2).value(), new_ts);

    DateTime new_ts2(300.0);
    row.mutable_datetime("ts")->set(new_ts2);
    EXPECT_EQ(row.datetime("ts").value(), new_ts2);
  }

  // Checks high-cardinality fields are added and retrieved by index and name.
  TEST_F(MixedMultivariateDataTest, AddAndAccessHighCardinality) {
    MixedMultivariateData row;
    row.add_high_cardinality(new StringData("user_abc123"), "user_id");
    row.add_high_cardinality(new StringData("prod_xyz789"), "product_id");

    EXPECT_EQ(row.high_cardinality_dim(), 2);
    EXPECT_EQ(row.dim(), 2);
    EXPECT_EQ(row.variable_type(0), VariableType::high_cardinality);
    EXPECT_EQ(row.variable_type(1), VariableType::high_cardinality);
    EXPECT_EQ(row.high_cardinality(0).value(), "user_abc123");
    EXPECT_EQ(row.high_cardinality(1).value(), "prod_xyz789");
    EXPECT_EQ(row.high_cardinality("user_id").value(), "user_abc123");
    EXPECT_EQ(row.high_cardinality("product_id").value(), "prod_xyz789");
  }

  // Checks mutable accessors update high-cardinality values by index and name.
  TEST_F(MixedMultivariateDataTest, HighCardinalityMutableAccess) {
    MixedMultivariateData row;
    row.add_high_cardinality(new StringData("old_value"), "token");

    row.mutable_high_cardinality(0)->set("new_value");
    EXPECT_EQ(row.high_cardinality(0).value(), "new_value");

    row.mutable_high_cardinality("token")->set("final_value");
    EXPECT_EQ(row.high_cardinality("token").value(), "final_value");
  }

  // Checks dim, per-type counts, and values for a row mixing all
  // four variable types.
  TEST_F(MixedMultivariateDataTest, MixedTypeWithHighCardinality) {
    MixedMultivariateData row;
    row.add_numeric(new DoubleData(42.0), "score");
    NEW(LabeledCategoricalData, color)("red", color_key_);
    row.add_categorical(color, "color");
    row.add_high_cardinality(new StringData("session_9876"), "session_id");
    row.add_datetime(new DateTimeData(DateTime(1000.0)), "ts");

    EXPECT_EQ(row.dim(), 4);
    EXPECT_EQ(row.numeric_dim(), 1);
    EXPECT_EQ(row.categorical_dim(), 1);
    EXPECT_EQ(row.high_cardinality_dim(), 1);
    EXPECT_EQ(row.variable_type(0), VariableType::numeric);
    EXPECT_EQ(row.variable_type(1), VariableType::categorical);
    EXPECT_EQ(row.variable_type(2), VariableType::high_cardinality);
    EXPECT_EQ(row.variable_type(3), VariableType::datetime);
    EXPECT_DOUBLE_EQ(row.numeric(0).value(), 42.0);
    EXPECT_EQ(row.high_cardinality(2).value(), "session_9876");
    EXPECT_EQ(row.datetime(3).value(), DateTime(1000.0));
  }

  // Checks variable() returns StringData for high-cardinality fields.
  TEST_F(MixedMultivariateDataTest, VariableMethodForHighCardinality) {
    MixedMultivariateData row;
    row.add_numeric(new DoubleData(1.5), "x");
    row.add_high_cardinality(new StringData("tok_42"), "token");
    row.add_datetime(new DateTimeData(DateTime(500.0)), "ts");

    // variable(1) must be a StringData whose display output is "tok_42".
    const Data &v = row.variable(1);
    std::ostringstream out;
    v.display(out);
    EXPECT_EQ(out.str(), "tok_42");
  }

  // Checks display() contains the high-cardinality value and label.
  TEST_F(MixedMultivariateDataTest, DisplayWithHighCardinality) {
    MixedMultivariateData row;
    row.add_numeric(new DoubleData(3.14), "x");
    NEW(LabeledCategoricalData, color)("red", color_key_);
    row.add_categorical(color, "color");
    row.add_datetime(new DateTimeData(DateTime(1000.0)), "ts");
    row.add_high_cardinality(new StringData("user_999"), "user_id");

    std::ostringstream out;
    row.display(out);
    const std::string s = out.str();
    EXPECT_FALSE(s.empty());
    EXPECT_NE(s.find("user_999"), std::string::npos);
    EXPECT_NE(s.find("red"),      std::string::npos);
  }

  // Checks copying a row with high-cardinality is a deep copy:
  // mutating the copy leaves the original unchanged.
  TEST_F(MixedMultivariateDataTest, CopyConstructorWithHighCardinality) {
    MixedMultivariateData original;
    original.add_numeric(new DoubleData(7.0), "x");
    original.add_high_cardinality(new StringData("tok_abc"), "token");

    MixedMultivariateData copy(original);
    EXPECT_EQ(copy.dim(), original.dim());
    EXPECT_EQ(copy.high_cardinality(1).value(), "tok_abc");

    // Mutating copy must not affect original (deep copy).
    copy.mutable_high_cardinality(1)->set("tok_MODIFIED");
    EXPECT_EQ(original.high_cardinality(1).value(), "tok_abc");
    EXPECT_EQ(copy.high_cardinality(1).value(), "tok_MODIFIED");
  }

}  // namespace
