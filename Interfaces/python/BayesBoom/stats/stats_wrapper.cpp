#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "LinAlg/Vector.hpp"
#include "LinAlg/Matrix.hpp"
#include "LinAlg/SpdMatrix.hpp"
#include "LinAlg/Array.hpp"

#include "stats/Bspline.hpp"
#include "stats/DataTable.hpp"
#include "stats/Encoders.hpp"
#include "stats/IQagent.hpp"
#include "stats/Spline.hpp"

#include "stats/acf.hpp"
#include "stats/hexbin.hpp"
#include "stats/moments.hpp"
#include "stats/optimal_arm_probabilities.hpp"

#include "Models/DataTypes.hpp"
#include "cpputil/Ptr.hpp"

namespace py = pybind11;
PYBIND11_DECLARE_HOLDER_TYPE(T, BOOM::Ptr<T>, true);

namespace BayesBoom {
  using namespace BOOM;

  void stats_def(py::module &boom) {

    boom.def("mean", [](const Matrix &m){return mean(m);},
             "Returns the mean of each column of m as a boom.Vector.");
    boom.def("var", [](const Matrix &m){return var(m);},
             "Returns the variance matrix of the data in a boom.Matrix.");
    boom.def("cor", [](const Matrix &m){return cor(m);},
             "Returns the correlation matrix of the data in a boom.Matrix.");

    boom.def("mean", [](const Vector &m){return mean(m);},
             "Returns the mean of a boom.Vector.");
    boom.def("var", [](const Vector &m){return var(m);},
             "Returns the variance of a boom.Vector.");
    boom.def("sd", [](const Vector &m){return sd(m);},
             "Returns the standard deviation of a boom.Vector.");

    boom.def("acf", [](const Vector &x, int lags, bool correlation) {
      return acf(x, lags, correlation);},
      py::arg("x"),
      py::arg("lags") = 40,
      py::arg("correlation") = true,
      "Args:\n\n"
      "  x:  The time series of data whose autocorrelation function is "
      "desired.\n"
      "  lags:  The number of lags to compute.\n"
      "  correlation:  If true the autocorrelation function is returned.  "
      "If false the autocovariance function is returned.\n");

    boom.def("compute_optimal_arm_probabilities",
             [](Matrix &values) {
               return compute_optimal_arm_probabilities(values);
             },
             py::arg("values"),
             "Args:\n\n"
             "  values:  A boom.Matrix with element (i, j) giving the "
             "expected reward value for iteration i, arm j.\n\n"
             "Returns:\n"
             "  A boom.Vector containing the optimal arm probability "
             "for each arm.\n");

    boom.def("compute_user_specific_optimal_arm_probabilities",
             [](Array &values) {
               return compute_user_specific_optimal_arm_probabilities(values);
             },
             py::arg("values"),
             "Args:\n\n"
             "  values:  A boom.Array with three dimensions.  "
             "Element (i, j, a) contains the expected reward value for "
             "subject i, iteration j, arm a.\n\n"
             "Returns:\n"
             "  A boom.Vector containing the optimal arm probability "
             "for each arm.\n");


    //===========================================================================
    py::class_<SplineBase> (boom, "SplineBase")
        .def("basis", &SplineBase::basis, py::arg("x: float"),
             py::return_value_policy::copy,
             "Spline basis expansion at x.")
        .def("basis_matrix", &SplineBase::basis_matrix, py::arg("x: Vector"),
             py::return_value_policy::copy,
             "Spline basis matrix expansion of the Vector x.")
        .def_property_readonly("dim", &SplineBase::basis_dimension,
                               "The dimension of the expanded basis.")
        .def("add_knot", &SplineBase::add_knot, py::arg("knot: float"),
             "Add a knot at the specified value.  The support of the spline will be "
             "expanded to include 'knot' if necessary.")
        .def("remove_knot", &SplineBase::remove_knot, py::arg("which_knot: int"),
             "Remove the specified knot.  If which_knot corresponds to the "
             "largest or smallest knots then the support of the spline will be "
             "reduced.")
        .def("knots", &SplineBase::knots)
        .def("number_of_knots", &SplineBase::number_of_knots)
        ;

    //===========================================================================
    py::class_<Bspline, SplineBase>(boom, "Bspline")
        .def(py::init<const Vector &, int>(), py::arg("knots"), py::arg("degree") = 3,
             "Create a Bspline basis.\n\n")
        .def("basis", (Vector (Bspline::*)(double)) &Bspline::basis, py::arg("x"),
             py::return_value_policy::copy,
             "The basis function expansion at x.")
        .def_property_readonly("order", &Bspline::order,
                               "The order of the spline. (1 + degree).")
        .def_property_readonly("degree", &Bspline::degree, "The degree of the spline.")
        .def("__repr__",
             [](const Bspline &s) {
               std::ostringstream out;
               out << "A Bspline basis of degree " << s.degree() << " with knots at ["
                   << s.knots() << "].";
               return out.str();
             })
        ;

    //===========================================================================
    py::class_<IQagent>(boom, "IQagent")
        .def(py::init(
            [](int bufsize) {
              return new IQagent(bufsize);
            }),
             py::arg("bufsize") = 20,
             "Args:\n"
             "  bufsize:  The number of data points to store before triggering a CDF "
             "refresh.")
        .def(py::init(
            [](const Vector &probs, int bufsize) {
              return new IQagent(probs, bufsize);
            }),
             py::arg("probs"),
             py::arg("bufsize") = 20,
             "Args\n"
             "  probs: A vector of probabilities defining the quantiles to focus on."
             "  bufsize:  The number of data points to store before triggering a CDF "
             "refresh.")
        .def(py::init(
            [](const IqAgentState &state) {
              return new IQagent(state);
            }),
             py::arg("state"),
             "Args:\n"
             "  state:  An object of class IqAgentState, previously generated by "
             "save_state().")
        .def(py::pickle(
            [](const IQagent &agent) {
              auto state = agent.save_state();
              return py::make_tuple(
                  state.max_buffer_size,
                  state.nobs,
                  state.data_buffer,
                  state.probs,
                  state.quantiles,
                  state.ecdf_sorted_data,
                  state.fplus,
                  state.fminus);
            },
            [](const py::tuple &tup) {
              IqAgentState state;
              state.max_buffer_size = tup[0].cast<int>();
              state.nobs = tup[1].cast<int>();
              state.data_buffer = tup[2].cast<Vector>();
              state.probs = tup[3].cast<Vector>();
              state.quantiles = tup[4].cast<Vector>();
              state.ecdf_sorted_data = tup[5].cast<Vector>();
              state.fplus = tup[6].cast<Vector>();
              state.fminus = tup[7].cast<Vector>();

              return IQagent(state);
            }))
        .def("add",
             [](IQagent &agent, double x) {
               agent.add(x);
             },
             py::arg("x"),
             "Args:\n"
             "  x: A data point to add to the empirical distribution.\n")
        .def("add",
             [](IQagent &agent, const Vector &x) {
               agent.add(x);
             },
             py::arg("x"),
             "Args:\n"
             "  x:  A boom.Vector of data to add to the empirical distribution.\n")
        .def("quantile", &IQagent::quantile, py::arg("prob"),
             "Args:\n"
             "  prob:  The probability for which a quantile is desired.")
        .def("cdf", &IQagent::cdf, py::arg("x"),
             "Args:\n"
             "  x: Return the fraction of data <= x.")
        .def("update_cdf", &IQagent::update_cdf,
             "Merge the data buffer into the CDF.  Update the CDF estimate.  "
             "Clear the data buffer.")
        ;


    //===========================================================================
    py::class_<DataTable,
               Data,
               Ptr<DataTable>>(boom, "DataTable")
        .def(py::init(
            []() {return new DataTable;}),
             "Default constructor.")
        .def("add_numeric",
             [](DataTable &table,
                const Vector &values,
                const std::string &name) {
               table.append_variable(values, name);
             },
             py::arg("values"),
             py::arg("name"),
             "Args:\n"
             "  values: The numeric values to append.\n"
             "  name: The name of the numeric variable.\n")
        .def("add_categorical",
             [](DataTable &table,
                const std::vector<int> &values,
                const std::vector<std::string> &labels,
                const std::string &name) {
               NEW(CatKey, key)(labels);
               table.append_variable(CategoricalVariable(values, key), name);
             },
             py::arg("values"),
             py::arg("labels"),
             py::arg("name"),
             "Args:\n"
             "  values:  The numeric codes of the categorical variables.\n"
             "  labels:  The labels corresponding to the unique values in "
             "'values.'\n"
             "  name:  The name of the categorical variable.")
        .def("add_categorical_from_labels",
             [](DataTable &table,
                const std::vector<std::string> &values,
                const std::string &name) {
               table.append_variable(CategoricalVariable(values), name);
             },
             py::arg("values"),
             py::arg("name"),
             "Args:\n"
             "  values:  The values (as strings) of the variable to be added.\n"
             "  name:  The name of the categorical variable.")
        .def_property_readonly(
            "nrow", &DataTable::nobs,
            "Number of rows (observations) in the table.")
        .def_property_readonly(
            "ncol", &DataTable::nvars,
            "Number of columns (variables) in the table.")
        .def_property_readonly(
            "variable_names",
            [](DataTable &table) {
              return table.vnames();
            },
            "The names of the variables (columns) in the data table.")
        .def("getvar",
             [](DataTable &table, int i) {
               return table.getvar(i);
             },
             py::arg("i"),
             "Return table column 'i'.  "
             "This is an error if column 'i' is non-numeric.")
        .def("variable_type",
             [](DataTable &table, int i) {
               VariableType vtype = table.variable_type(i);
               switch (vtype) {
                 case VariableType::categorical:
                   return "categorical";
                   break;

                 case VariableType::numeric:
                   return "numeric";
                   break;

                 default:
                   return "unknown";
               }
             },
             py::arg("i"),
             "Return the variable type of column 'i' as a string.")

        .def("get_nominal_values",
             [](DataTable &table, int i) {
               CategoricalVariable var = table.get_nominal(i);
               std::vector<int> values;
               values.reserve(table.nrow());
               for (int i = 0; i < table.nrow(); ++i) {
                 values.push_back(var[i]->value());
               }
               return values;
             },
             py::arg("i"),
             "Return table column 'i'.  \n"
             "This is an error if column 'i' is not a nominal categorical "
             "variable.")
        .def("get_nominal_levels",
             [](DataTable &table, int i) {
               CategoricalVariable var = table.get_nominal(i);
               return var.labels();
             },
             py::arg("i"),
             "Return the levels associated with variable 'i'.  \n"
             "This is an error if column 'i' is not a nominal categorical "
             "variable."
             )
        ;

    //===========================================================================
    py::class_<DataEncoder, Ptr<DataEncoder>>(boom, "DataEncoder")
        .def_property_readonly(
            "dim", &DataEncoder::dim,
            "The number of columns in the encoded matrix output by the "
            "encoder.")
        .def("encode_dataset",
             &DataEncoder::encode_dataset,
             py::arg("data"),
             "Encode the (mixed type) data table into a numeric predictor "
             "matrix.\n\n"
             "Args:\n"
             "  data:  The boom.DataTable object to be encoded.\n")
        ;

    //===========================================================================
    py::class_<MainEffectsEncoder, DataEncoder, Ptr<MainEffectsEncoder>>(
        boom, "MainEffectsEncoder")
        ;

    //===========================================================================
    py::class_<EffectsEncoder, MainEffectsEncoder, Ptr<EffectsEncoder>>(
        boom, "EffectsEncoder")
        .def(py::init(
            [](int which_variable, const std::vector<std::string> &levels) {
              NEW(CatKey, key)(levels);
              return new EffectsEncoder(which_variable, key);
            }),
             py::arg("which_variable"),
             py::arg("levels"),
             "Args:\n"
             "  which_variable: The position of the input variable in the "
             "data table.\n"
             "  levels: The set of levels (as strings) to be encoded.  The \n"
             "    last level listed will be the reference level.\n")
        .def("encode", [](const EffectsEncoder &encoder, int level) {
            return encoder.encode(level);
          },
          "Encode a categorical value by its integer code.")
        ;

    //===========================================================================
    py::class_<DatasetEncoder, DataEncoder, Ptr<DatasetEncoder>>(
        boom, "DatasetEncoder")
        .def(py::init(
            [](std::vector<Ptr<DataEncoder>> &encoders,
               bool add_intercept) {
              NEW(DatasetEncoder, encoder)(add_intercept);
              for (const auto &el : encoders) {
                encoder->add_encoder(el);
              }
              return encoder;
            }),
             py::arg("encoders"),
             py::arg("add_intercept") = true,
             "Args: \n"
             "  encoders:  The encoders that produce individual effects.\n"
             "  add_intercept: If True then a column of 1's is prepended \n"
             "    to the beginning of the output matrix.\n")
        ;

    //===========================================================================
    py::class_<Hexbin>(boom, "Hexbin")
        .def(py::init<int>(),
             py::arg("gridsize") = 50,
             "Create an empty hexbin plot.")
        .def(py::init<const Vector &, const Vector &, int>(),
             py::arg("x"),
             py::arg("y"),
             py::arg("gridsize"),
             "Create a hexbin plot from x and y.\n\n"
             "Args:\n"
             "  x: Vector to plot on the horizontal axis.\n"
             "  y: Vector to plot on the vertical axis.\n"
             "  gridsize:  Number of histogram buckets in each direction.\n")
        .def("add_data",
             &Hexbin::add_data,
             py::arg("x"),
             py::arg("y"),
             "Add data to an existing hexbin plot.\n\n"
             "Args:\n"
             "  x, y: Vectors to be plotted on the x and y axes.\n")
        .def_property_readonly(
            "hexagons",
            [](const Hexbin &obj) {return obj.hexagons();},
            "A 3 column matrix containing the x and y coordinates of the hexagon centers \n"
            "(first two columns) and the hexagon counts (frequency, third column).\n")
        ;

  }  // stats_def

}  // namespace BayesBoom
