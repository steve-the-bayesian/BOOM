#include "gtest/gtest.h"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <sstream>

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

  // Checks that MixedMultivariateData can be default-constructed without error.
  TEST_F(DataTableTest, DefaultConstructor) {
    MixedMultivariateData data;
  }

  // Reads .txt and .csv files; checks row/column counts,
  // auto-detected variable types, and column names.
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

  // Verifies fake_data_table() row count, column types, and levels.
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


  // Checks repeat() creates n identical copies preserving types and values.
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

  // Verifies simulate_fake_data_table() column names, types, and counts.
  TEST_F(DataTableTest, SimulateRandomData) {
    DataTable my_data = simulate_fake_data_table(
        100,                                        // sample size
        {"height", "weight"},                       // numeric
        {{"color", {"red", "blue", "green"}},
         {"stooge", {"larry", "moe", "curly"}}},    // categorical
        {{"birthday", {DateTime(173.5),
             DateTime(5000.3)}}},                   // datetime
        {{"user_id", 6}});                          // high_cardinality

    EXPECT_EQ(my_data.ncol(), 6);
    EXPECT_EQ(my_data.nrow(), 100);
    EXPECT_EQ(my_data.vnames()[0], "height");
    EXPECT_EQ(my_data.vnames()[1], "weight");
    EXPECT_EQ(my_data.vnames()[2], "color");
    EXPECT_EQ(my_data.vnames()[3], "stooge");
    EXPECT_EQ(my_data.vnames()[4], "birthday");
    EXPECT_EQ(my_data.vnames()[5], "user_id");
    EXPECT_EQ(my_data.variable_type(0), VariableType::numeric);
    EXPECT_EQ(my_data.variable_type(1), VariableType::numeric);
    EXPECT_EQ(my_data.variable_type(2), VariableType::categorical);
    EXPECT_EQ(my_data.variable_type(3), VariableType::categorical);
    EXPECT_EQ(my_data.variable_type(4), VariableType::datetime);
    EXPECT_EQ(my_data.variable_type(5), VariableType::high_cardinality);
  }

  // Checks append_variable() correctly adds all four column types.
  TEST_F(DataTableTest, AppendAllFourTypes) {
    int n = 20;
    Vector v(n);
    v.randomize();
    CategoricalVariable cv(std::vector<int>(n, 1),
                           new CatKey({"apple", "banana", "cherry"}));
    DateTimeVariable dtv(std::vector<DateTime>(n, DateTime(86400.0)));
    HighCardinalityVariable hcv(std::vector<std::string>(n, "user_001"));

    DataTable table;
    table.append_variable(v, "score");
    table.append_variable(cv, "fruit");
    table.append_variable(dtv, "timestamp");
    table.append_variable(hcv, "user_id");

    EXPECT_EQ(table.nvars(), 4);
    EXPECT_EQ(table.nrow(), n);
    EXPECT_EQ(table.variable_type(0), VariableType::numeric);
    EXPECT_EQ(table.variable_type(1), VariableType::categorical);
    EXPECT_EQ(table.variable_type(2), VariableType::datetime);
    EXPECT_EQ(table.variable_type(3), VariableType::high_cardinality);
    EXPECT_EQ(table.numeric_dim(), 1);
    EXPECT_EQ(table.categorical_dim(), 1);
    EXPECT_EQ(table.datetime_dim(), 1);
    EXPECT_EQ(table.vnames()[0], "score");
    EXPECT_EQ(table.vnames()[1], "fruit");
    EXPECT_EQ(table.vnames()[2], "timestamp");
    EXPECT_EQ(table.vnames()[3], "user_id");
  }

  // Checks get_datetime() retrieval by index and name matches insertion.
  TEST_F(DataTableTest, GetDatetime) {
    int n = 5;
    std::vector<DateTime> values;
    for (int i = 0; i < n; ++i) {
      values.push_back(DateTime(double(i + 1) * 86400.0));
    }
    DataTable table;
    table.append_variable(DateTimeVariable(values), "event_time");

    EXPECT_EQ(table.variable_type(0), VariableType::datetime);
    EXPECT_EQ(table.nrow(), n);

    DateTimeVariable by_index = table.get_datetime(0);
    ASSERT_EQ(by_index.size(), n);
    for (int i = 0; i < n; ++i) {
      EXPECT_EQ(by_index.data()[i], values[i]);
    }

    DateTimeVariable by_name = table.get_datetime("event_time");
    ASSERT_EQ(by_name.size(), n);
    for (int i = 0; i < n; ++i) {
      EXPECT_EQ(by_name.data()[i], values[i]);
    }
  }

  // Checks get_high_cardinality() retrieval by index and name.
  TEST_F(DataTableTest, GetHighCardinality) {
    std::vector<std::string> ids = {"alpha", "beta", "gamma", "delta", "epsilon"};
    DataTable table;
    table.append_variable(HighCardinalityVariable(ids), "user_id");

    EXPECT_EQ(table.variable_type(0), VariableType::high_cardinality);
    EXPECT_EQ(table.nrow(), 5);

    HighCardinalityVariable by_index = table.get_high_cardinality(0);
    ASSERT_EQ(by_index.size(), 5);
    for (int i = 0; i < 5; ++i) {
      EXPECT_EQ(by_index.data()[i], ids[i]);
    }

    HighCardinalityVariable by_name = table.get_high_cardinality("user_id");
    ASSERT_EQ(by_name.size(), 5);
    for (int i = 0; i < 5; ++i) {
      EXPECT_EQ(by_name.data()[i], ids[i]);
    }
  }

  // Checks get_numeric() and get_nominal() in a mixed-type table.
  TEST_F(DataTableTest, GetNumericAndCategoricalFromMixedTable) {
    int n = 10;
    Vector v(n);
    v.randomize();
    Ptr<CatKey> key(new CatKey({"red", "blue", "green"}));
    CategoricalVariable cv(std::vector<int>(n, 0), key);

    DataTable table;
    table.append_variable(v, "score");
    table.append_variable(cv, "color");

    EXPECT_EQ(table.nvars(), 2);
    Vector retrieved = table.get_numeric("score");
    ASSERT_EQ(retrieved.size(), n);
    for (int i = 0; i < n; ++i) {
      EXPECT_DOUBLE_EQ(retrieved[i], v[i]);
    }

    CategoricalVariable retrieved_cat = table.get_nominal("color");
    ASSERT_EQ(retrieved_cat.size(), n);
    for (int i = 0; i < n; ++i) {
      EXPECT_EQ(retrieved_cat[i]->value(), 0);
    }
  }

  // Write 'dt' to 'out' as "YYYY-MM-DD HH:MM:SS" with zero-padded fields.
  // This format is what DataTable::read_file recognises as a datetime column.
  void write_iso8601(std::ostream &out, const DateTime &dt) {
    const Date &d = dt.date();
    out << std::setfill('0')
        << std::setw(4) << d.year()                       << "-"
        << std::setw(2) << static_cast<int>(d.month())    << "-"
        << std::setw(2) << d.day()                        << " "
        << std::setw(2) << dt.hour()                      << ":"
        << std::setw(2) << dt.minute()                    << ":"
        << std::setw(2) << dt.second();
  }

  // Verifies a DataTable with all four types survives a CSV
  // write-and-read round trip with values and types intact.
  TEST_F(DataTableTest, RoundTripCsv) {
    // cbrt(125) = 5, so the high-cardinality threshold is max(5, 5) = 5.
    // "color" and "size" have 3 levels each (< 5) -> stay categorical.
    // "user_id" generates ~125 unique 8-char strings (>> 5) -> high_cardinality.
    const int sample_size = 125;
    DataTable original = simulate_fake_data_table(
        sample_size,
        {"x1", "x2"},
        {{"color", {"red", "blue", "green"}},
         {"size",  {"small", "medium", "large"}}},
        {{"event_time", {DateTime(1000.0), DateTime(2000.0)}}},
        {{"user_id", 8}});

    ASSERT_EQ(original.nvars(), 6);
    ASSERT_EQ(original.nrow(), sample_size);

    // RAII guard: deletes the temp file when this scope exits, even on failure.
    const std::string tmp_path = "boom_data_table_roundtrip_test.csv";
    struct TempFileGuard {
      std::string path;
      explicit TempFileGuard(std::string p) : path(std::move(p)) {}
      ~TempFileGuard() { std::remove(path.c_str()); }
      TempFileGuard(const TempFileGuard &) = delete;
      TempFileGuard &operator=(const TempFileGuard &) = delete;
    } guard(tmp_path);

    // Pre-format every cell as a string.  Datetimes are written as ISO 8601
    // ("YYYY-MM-DD HH:MM:SS") so read_file can detect the datetime type.
    const int ncol = original.nvars();
    const int nrow = original.nrow();
    // cells[col][row]
    std::vector<std::vector<std::string>> cells(ncol,
                                                std::vector<std::string>(nrow));
    for (int j = 0; j < ncol; ++j) {
      VariableType vtype = original.variable_type(j);
      if (vtype == VariableType::numeric) {
        for (int i = 0; i < nrow; ++i) {
          std::ostringstream oss;
          oss << std::setprecision(17) << original.getvar(i, j);
          cells[j][i] = oss.str();
        }
      } else if (vtype == VariableType::categorical) {
        CategoricalVariable cv = original.get_nominal(j);
        for (int i = 0; i < nrow; ++i) {
          cells[j][i] = cv.label(i);
        }
      } else if (vtype == VariableType::datetime) {
        DateTimeVariable dtv = original.get_datetime(j);
        for (int i = 0; i < nrow; ++i) {
          std::ostringstream oss;
          write_iso8601(oss, dtv.data()[i]);
          cells[j][i] = oss.str();
        }
      } else {  // high_cardinality
        HighCardinalityVariable hcv = original.get_high_cardinality(j);
        for (int i = 0; i < nrow; ++i) {
          cells[j][i] = hcv.data()[i];
        }
      }
    }

    // Write CSV with header.
    {
      std::ofstream out(tmp_path);
      ASSERT_TRUE(out.is_open()) << "Could not create: " << tmp_path;
      const auto &names = original.vnames();
      for (int j = 0; j < ncol; ++j) {
        if (j > 0) out << ",";
        out << names[j];
      }
      out << "\n";
      for (int i = 0; i < nrow; ++i) {
        for (int j = 0; j < ncol; ++j) {
          if (j > 0) out << ",";
          out << cells[j][i];
        }
        out << "\n";
      }
    }

    // Read the file back and check practical equivalence.
    DataTable recovered(tmp_path, /*header=*/true, ",");

    ASSERT_EQ(recovered.nvars(), original.nvars());
    ASSERT_EQ(recovered.nrow(), original.nrow());
    EXPECT_EQ(recovered.vnames(), original.vnames());

    for (int j = 0; j < ncol; ++j) {
      const VariableType orig_type = original.variable_type(j);
      EXPECT_EQ(recovered.variable_type(j), orig_type)
          << "Type mismatch: column " << j
          << " (" << original.vnames()[j] << ")";

      if (orig_type == VariableType::numeric) {
        Vector orig_col = original.get_numeric(j);
        Vector rec_col  = recovered.get_numeric(j);
        ASSERT_EQ(orig_col.size(), rec_col.size());
        for (int i = 0; i < nrow; ++i) {
          EXPECT_DOUBLE_EQ(orig_col[i], rec_col[i])
              << "  row " << i << ", col " << j;
        }

      } else if (orig_type == VariableType::categorical) {
        CategoricalVariable orig_cv = original.get_nominal(j);
        CategoricalVariable rec_cv  = recovered.get_nominal(j);
        ASSERT_EQ(orig_cv.size(), rec_cv.size());
        for (int i = 0; i < nrow; ++i) {
          EXPECT_EQ(orig_cv.label(i), rec_cv.label(i))
              << "  row " << i << ", col " << j;
        }

      } else if (orig_type == VariableType::datetime) {
        DateTimeVariable orig_dtv = original.get_datetime(j);
        DateTimeVariable rec_dtv  = recovered.get_datetime(j);
        ASSERT_EQ(orig_dtv.size(), rec_dtv.size());
        for (int i = 0; i < nrow; ++i) {
          // Re-format the recovered DateTime and compare it with the cell
          // string that was written.  Direct second() comparison is fragile:
          // DateTime stores time as a fraction of a day, and computing
          // floor(rem(N/86400 * 86400, 60)) can lose 1 second to floating-
          // point error.  lround on the total seconds is stable because the
          // recovered fraction N/86400 multiplied by 86400 differs from N by
          // much less than 0.5.
          const DateTime &r = rec_dtv.data()[i];
          const Date &rd = r.date();
          long total_secs = std::min(86399L, std::lround(r.seconds_into_day()));
          std::ostringstream oss;
          oss << std::setfill('0')
              << std::setw(4) << rd.year()                    << "-"
              << std::setw(2) << static_cast<int>(rd.month()) << "-"
              << std::setw(2) << rd.day()                     << " "
              << std::setw(2) << (total_secs / 3600)           << ":"
              << std::setw(2) << ((total_secs % 3600) / 60)    << ":"
              << std::setw(2) << (total_secs % 60);
          EXPECT_EQ(cells[j][i], oss.str()) << "  row " << i;
        }

      } else if (orig_type == VariableType::high_cardinality) {
        HighCardinalityVariable orig_hcv = original.get_high_cardinality(j);
        HighCardinalityVariable rec_hcv  = recovered.get_high_cardinality(j);
        ASSERT_EQ(orig_hcv.size(), rec_hcv.size());
        for (int i = 0; i < nrow; ++i) {
          EXPECT_EQ(orig_hcv.data()[i], rec_hcv.data()[i])
              << "  row " << i << ", col " << j;
        }
      }
    }
  }

  // Checks append_row() seeds column structure on the first call
  // and grows the table correctly for all four variable types.
  TEST_F(DataTableTest, AppendRowAllTypes) {
    // First append_row on an empty table seeds the column structure.
    NEW(MixedMultivariateData, r0)();
    r0->add_numeric(new DoubleData(10.0), "score");
    NEW(LabeledCategoricalData, c0)("red", new CatKey({"red", "blue", "green"}));
    r0->add_categorical(c0, "color");
    r0->add_datetime(new DateTimeData(DateTime(100.0)), "ts");
    r0->add_high_cardinality(new StringData("tok_x"), "token");

    DataTable table;
    table.append_row(*r0);
    ASSERT_EQ(table.nrow(), 1);
    ASSERT_EQ(table.nvars(), 4);
    EXPECT_EQ(table.variable_type(0), VariableType::numeric);
    EXPECT_EQ(table.variable_type(1), VariableType::categorical);
    EXPECT_EQ(table.variable_type(2), VariableType::datetime);
    EXPECT_EQ(table.variable_type(3), VariableType::high_cardinality);
    EXPECT_DOUBLE_EQ(table.getvar(0, 0), 10.0);
    EXPECT_EQ(table.get_nominal(0, 1)->value(),
              c0->catkey()->findstr("red"));
    EXPECT_EQ(table.get_datetime(2).data()[0], DateTime(100.0));
    EXPECT_EQ(table.get_high_cardinality(3).data()[0], "tok_x");

    // Second append_row exercises the non-empty path.
    NEW(MixedMultivariateData, r1)();
    r1->add_numeric(new DoubleData(20.0), "score");
    NEW(LabeledCategoricalData, c1)("blue", c0->catkey());
    r1->add_categorical(c1, "color");
    r1->add_datetime(new DateTimeData(DateTime(200.0)), "ts");
    r1->add_high_cardinality(new StringData("tok_y"), "token");

    table.append_row(*r1);
    ASSERT_EQ(table.nrow(), 2);
    EXPECT_DOUBLE_EQ(table.getvar(1, 0), 20.0);
    EXPECT_EQ(table.get_datetime(2).data()[1], DateTime(200.0));
    EXPECT_EQ(table.get_high_cardinality(3).data()[1], "tok_y");
  }

  // Checks row() values match column accessors for all four types.
  TEST_F(DataTableTest, RowAccessorAllTypes) {
    // col 0: numeric, col 1: categorical, col 2: datetime, col 3: high_cardinality
    DataTable table = simulate_fake_data_table(
        5, {"score"},
        {{"color", {"red", "blue", "green"}}},
        {{"ts", {DateTime(1000.0), DateTime(5000.0)}}},
        {{"token", 6}});

    for (int i = 0; i < 5; ++i) {
      Ptr<MixedMultivariateData> r = table.row(i);
      ASSERT_EQ(r->dim(), 4);
      EXPECT_EQ(r->variable_type(0), VariableType::numeric);
      EXPECT_EQ(r->variable_type(1), VariableType::categorical);
      EXPECT_EQ(r->variable_type(2), VariableType::datetime);
      EXPECT_EQ(r->variable_type(3), VariableType::high_cardinality);

      // Cross-check row accessor against column accessors.
      EXPECT_DOUBLE_EQ(r->numeric("score").value(),      table.getvar(i, 0));
      EXPECT_EQ(r->categorical("color").value(),         table.get_nominal(i, 1)->value());
      EXPECT_EQ(r->datetime("ts").value(),               table.get_datetime(2).data()[i]);
      EXPECT_EQ(r->high_cardinality("token").value(),    table.get_high_cardinality(3).data()[i]);
    }
  }

  // Checks operator<< output contains all column names and values.
  TEST_F(DataTableTest, PrintAllTypes) {
    DataTable table = simulate_fake_data_table(
        5, {"score"},
        {{"color", {"red", "blue", "green"}}},
        {{"ts", {DateTime(1000.0), DateTime(5000.0)}}},
        {{"token", 6}});

    std::ostringstream out;
    out << table;
    const std::string s = out.str();
    EXPECT_FALSE(s.empty());

    // Every column name must appear.
    for (const auto &name : table.vnames()) {
      EXPECT_NE(s.find(name), std::string::npos)
          << "column name '" << name << "' missing from output";
    }
    // A data value from each non-numeric column type must also appear.
    EXPECT_NE(s.find(table.get_nominal(1).label(0)),          std::string::npos);
    EXPECT_NE(s.find(table.get_high_cardinality(3).data()[0]), std::string::npos);
  }

  // Checks rbind() appends rows across all four variable types,
  // preserving row order and values.
  TEST_F(DataTableTest, RbindAllTypes) {
    const std::map<std::string, std::vector<std::string>> cat =
        {{"color", {"red", "blue", "green"}}};
    const std::map<std::string, std::pair<DateTime, DateTime>> dt =
        {{"ts", {DateTime(1000.0), DateTime(5000.0)}}};
    const std::map<std::string, int> hc = {{"token", 6}};

    DataTable a = simulate_fake_data_table(3, {"score"}, cat, dt, hc);
    DataTable b = simulate_fake_data_table(2, {"score"}, cat, dt, hc);

    // Save column values before rbind so we can verify nothing was altered.
    const Vector             a_scores = a.get_numeric(0);
    const DateTimeVariable   a_dt     = a.get_datetime(2);
    const HighCardinalityVariable a_hc = a.get_high_cardinality(3);
    const Vector             b_scores = b.get_numeric(0);
    const DateTimeVariable   b_dt     = b.get_datetime(2);
    const HighCardinalityVariable b_hc = b.get_high_cardinality(3);

    a.rbind(b);
    ASSERT_EQ(a.nrow(), 5);
    ASSERT_EQ(a.nvars(), 4);

    const Vector             combined_scores = a.get_numeric(0);
    const DateTimeVariable   combined_dt     = a.get_datetime(2);
    const HighCardinalityVariable combined_hc = a.get_high_cardinality(3);

    for (int i = 0; i < 3; ++i) {
      EXPECT_DOUBLE_EQ(combined_scores[i],    a_scores[i]);
      EXPECT_EQ(combined_dt.data()[i],        a_dt.data()[i]);
      EXPECT_EQ(combined_hc.data()[i],        a_hc.data()[i]);
    }
    for (int i = 0; i < 2; ++i) {
      EXPECT_DOUBLE_EQ(combined_scores[3 + i], b_scores[i]);
      EXPECT_EQ(combined_dt.data()[3 + i],     b_dt.data()[i]);
      EXPECT_EQ(combined_hc.data()[3 + i],     b_hc.data()[i]);
    }
  }

  // Checks cbind() merges a high-cardinality column correctly.
  TEST_F(DataTableTest, CbindWithHighCardinality) {
    DataTable left;
    left.append_variable(Vector{1.0, 2.0, 3.0}, "score");

    DataTable right;
    right.append_variable(
        HighCardinalityVariable(
            std::vector<std::string>{"a", "b", "c"}),
        "token");

    left.cbind(right);
    ASSERT_EQ(left.nvars(), 2);
    EXPECT_EQ(left.variable_type(0), VariableType::numeric);
    EXPECT_EQ(left.variable_type(1), VariableType::high_cardinality);

    HighCardinalityVariable hcv = left.get_high_cardinality(1);
    ASSERT_EQ(hcv.size(), 3);
    EXPECT_EQ(hcv.data()[0], "a");
    EXPECT_EQ(hcv.data()[2], "c");
  }

  // Checks repeat() replicates high-cardinality values across rows.
  TEST_F(DataTableTest, RepeatWithHighCardinality) {
    NEW(MixedMultivariateData, row)();
    row->add_numeric(new DoubleData(7.0), "x");
    row->add_high_cardinality(new StringData("session_abc"), "session_id");

    DataTable table = repeat(*row, 5);
    ASSERT_EQ(table.nrow(), 5);
    ASSERT_EQ(table.nvars(), 2);
    EXPECT_EQ(table.variable_type(0), VariableType::numeric);
    EXPECT_EQ(table.variable_type(1), VariableType::high_cardinality);

    HighCardinalityVariable hcv = table.get_high_cardinality(1);
    ASSERT_EQ(hcv.size(), 5);
    for (int i = 0; i < 5; ++i) {
      EXPECT_DOUBLE_EQ(table.getvar(i, 0), 7.0);
      EXPECT_EQ(hcv.data()[i], "session_abc");
    }
  }

}  // namespace
