#include "gtest/gtest.h"
#include "cpputil/seq.hpp"
#include "LinAlg/Vector.hpp"
#include "stats/Resampler.hpp"
#include "test_utils/test_utils.hpp"
#include "stats/ChiSquareTest.hpp"
#include "stats/FreqDist.hpp"
#include "stats/DataTable.hpp"
#include "stats/fake_data_table.hpp"

namespace {
  using namespace BOOM;
  using std::endl;

  class DataTableTest : public ::testing::Test {
   protected:
    DataTableTest()
        : color_key_(new CatKey({"red", "blue", "green"})),
          shape_key_(new CatKey({"circle", "square", "triangle", "rhombus"}))
    {
      GlobalRng::rng.seed(8675309);
    }

    Ptr<CatKey> color_key_;
    Ptr<CatKey> shape_key_;
  };

  TEST_F(DataTableTest, DefaultConstructor) {
    MixedMultivariateData data;
  }

  // Read the autopref.txt data in from a .txt file.
  // Read in CarsClean.csv from a .csv file.
  // Check that the data types are correct.
  TEST_F(DataTableTest, ReadFromFile) {
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

  TEST_F(DataTableTest, TestFakeDataTable) {
    DataTable table = fake_data_table(112, 3, {4, 2, 3});

    EXPECT_EQ(table.nobs(), 112);
    EXPECT_EQ(table.ncol(), 6);
    EXPECT_EQ(table.variable_type(0), VariableType::numeric);
    EXPECT_EQ(table.variable_type(1), VariableType::numeric);
    EXPECT_EQ(table.variable_type(2), VariableType::numeric);
    EXPECT_EQ(table.variable_type(3), VariableType::categorical);
    EXPECT_EQ(table.variable_type(4), VariableType::categorical);
    EXPECT_EQ(table.variable_type(5), VariableType::categorical);

    EXPECT_EQ(table.nlevels(0), 1);
    EXPECT_EQ(table.nlevels(1), 1);
    EXPECT_EQ(table.nlevels(2), 1);
    EXPECT_EQ(table.nlevels(3), 4);
    EXPECT_EQ(table.nlevels(4), 2);
    EXPECT_EQ(table.nlevels(5), 3);
  }


  TEST_F(DataTableTest, Repeat) {
    NEW(MixedMultivariateData, row)();
    row->add_numeric(new DoubleData(3.2), "X1");
    NEW(LabeledCategoricalData, stooge)(
        "Moe", new CatKey({"Larry", "Moe", "Curly", "Shemp"}));
    row->add_categorical(stooge, "Stooges");
    row->add_numeric(new DoubleData(8675309), "Jenny");
    row->add_datetime(new DateTimeData(DateTime()), "Timestamp");


    DataTable table = repeat(*row, 12);
    EXPECT_EQ(table.nvars(), 4);
    EXPECT_EQ(table.nobs(), 12);
    EXPECT_EQ(table.variable_type(0), VariableType::numeric);
    EXPECT_EQ(table.variable_type(1), VariableType::categorical);
    EXPECT_EQ(table.variable_type(2), VariableType::numeric);
    EXPECT_EQ(table.variable_type(3), VariableType::datetime);

    EXPECT_DOUBLE_EQ(table.getvar(0, 0), table.getvar(1, 0));
    for (int i = 0; i < table.nrow(); ++i) {
      EXPECT_DOUBLE_EQ(table.getvar(0, 0), table.getvar(i, 0));
    }

    for (int i = 0; i < table.nrow(); ++i) {
      EXPECT_DOUBLE_EQ(table.getvar(0, 2), table.getvar(i, 2));
    }

    for (int i = 0; i < table.nrow(); ++i) {
      EXPECT_EQ(table.get_nominal(0, 1)->value(),
                table.get_nominal(i, 1)->value());
    }

  }
}  // namespace
