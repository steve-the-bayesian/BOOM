// Copyright 2011 Google Inc. All Rights Reserved.
// Author: steve.the.bayesian@gmail.com (Steve Scott)

#include "Models/DataTypes.hpp"
#include "Models/Glm/Glm.hpp"
#include "Models/Glm/BinomialRegressionData.hpp"
#include "Models/CategoricalData.hpp"
#include "Models/MarkovModel.hpp"
#include "Models/CompositeData.hpp"

#include "cpputil/report_error.hpp"
#include "r_interface/boom_r_tools.hpp"
#include <memory>
#include <R_ext/Arith.h>

namespace BOOM {
  namespace RInterface {

    namespace {
      template <class VEC>
      int count_missing(const VEC &v) {
        int number_missing = 0;
        for(int i = 0; i < v.size(); ++i){
          number_missing += R_IsNA(v[i]);
        }
        return number_missing;
      }
    }  // namespace

    // A class for extracting data from objects passed from R.
    class DataExtractor {
     public:
      static DataExtractor * Create(const std::string &type);
      virtual ~DataExtractor() {}

      // Extract one subject's data from supplied R object
      virtual std::vector<BOOM::Ptr<BOOM::Data>> Extract(SEXP rdata) const = 0;
    };

    class DoubleDataExtractor : public DataExtractor {
     public:
      // rdata is a an R vector
      virtual std::vector<Ptr<Data>> Extract(SEXP rdata) const {
        BOOM::Vector v = ToBoomVector(rdata);
        int n = v.size();
        std::vector<Ptr<Data>> ans;
        ans.reserve(n);
        for (int i = 0; i < n; ++i) {
          Ptr<Data> dp = new DoubleData(v[i]);
          if(R_IsNA(v[i])) dp->set_missing_status(BOOM::Data::completely_missing);
          ans.push_back(dp);
        }
        return ans;
      }
    };

    class FactorDataExtractor : public DataExtractor {
     public:
      virtual std::vector<Ptr<Data>> Extract(SEXP rdata) const {
        int n = Rf_length(rdata);
        std::vector<Ptr<Data>> ans;
        if (n == 0) {
          return ans;
        }
        SEXP rlevels = Rf_getAttrib(rdata, R_LevelsSymbol);
        std::vector<std::string> levels = StringVector(rlevels);
        NEW(BOOM::CatKey, key)(levels);

        ans.reserve(n);
        double *values = REAL(Rf_coerceVector(rdata, REALSXP));
        for(int i = 0; i < n; ++i) {
          NEW(BOOM::CategoricalData, y)(lround(values[i]) - 1, key);
          ans.push_back(y);
        }
        return ans;
      }
    };

    class MarkovDataExtractor : public DataExtractor {
     public:
      virtual std::vector<Ptr<Data>> Extract(SEXP rdata) const {
        int n = Rf_length(rdata);
        std::vector<Ptr<Data>> ans;
        if (n == 0) {
          return ans;
        }
        SEXP rlevels = Rf_getAttrib(rdata, R_LevelsSymbol);
        std::vector<std::string> levels = StringVector(rlevels);
        Ptr<CatKeyBase> key = new CatKey(levels);

        ans.reserve(n);
        double *values = REAL(Rf_coerceVector(rdata, REALSXP));

        // Assign initial value.
        uint level = lround(values[0]) - 1;
        NEW(BOOM::MarkovData, previous_data)(level, key);
        ans.push_back(previous_data);

        // Assign subsequent values.
        for(int i = 1; i < n; ++i) {
          level = lround(values[i]) - 1;
          NEW(BOOM::MarkovData, current_data)(level, previous_data);
          ans.push_back(current_data);
          previous_data = current_data;
        }
        return ans;
      }
    };

    class IntDataExtractor : public DataExtractor {
     public:
      // rdata is a an R vector
      virtual std::vector<Ptr<Data>> Extract(SEXP rdata) const {
        Vector v = ToBoomVector(rdata);
        int n = v.size();
        std::vector<Ptr<Data>> ans;
        ans.reserve(n);
        for (int i = 0; i < n; ++i) {
          bool missing = R_IsNA(v[i]);
          const int min_int = std::numeric_limits<int>::min();
          Ptr<Data> dp = new IntData(missing ? min_int : lround(v[i]));
          if(missing) dp->set_missing_status(BOOM::Data::completely_missing);
          ans.push_back(dp);
        }
        return ans;
      }
    };

    class VectorDataExtractor : public DataExtractor {
     public:
      // rdata is an R matrix with each row
      virtual std::vector<Ptr<Data>> Extract(SEXP rdata) const {
        BOOM::Matrix y = ToBoomMatrix(rdata);
        int n = nrow(y);
        std::vector<Ptr<Data>> ans;
        ans.reserve(n);
        for (int i = 0; i < n; ++i) {
          Ptr<Data> dp = new VectorData(y.row(i));
          check_missing(dp);
          ans.push_back(dp);
        }
        return(ans);
      }

      void check_missing(const Ptr<Data> &dp) const {
        Ptr<VectorData> data(dp.dcast<VectorData>());
        const Vector &y(data->value());
        int number_missing = count_missing(y);
        if(number_missing == y.size()){
          dp->set_missing_status(BOOM::Data::completely_missing);
        }else if(number_missing > 0){
          dp->set_missing_status(BOOM::Data::partly_missing);
        }
      }
    };

    class RegressionDataExtractor : public DataExtractor {
     public:
      // rdata is a list that contains two elements: a vector named y and
      // a matrix named x.
      virtual std::vector<Ptr<Data>> Extract(SEXP rdata) const {
        Vector y = ToBoomVector(getListElement(rdata, "y"));
        Matrix x = ToBoomMatrix(getListElement(rdata, "x"));
        int n = y.size();
        std::vector<Ptr<Data>> ans;
        ans.reserve(n);
        for (int i = 0; i < n; ++i) {
          Ptr<Data> dp = new RegressionData(y[i], x.row(i));
          check_missing(dp);
          ans.push_back(dp);
        }
        return ans;
      }

      void check_missing(const Ptr<Data> &d) const {
        Ptr<RegressionData> dp = d.dcast<RegressionData>();
        bool missing_response = R_IsNA(dp->y());
        const Vector &x(dp->x());
        int number_missing = count_missing(x);
        if(missing_response && number_missing == x.size()){
          dp->set_missing_status(BOOM::Data::completely_missing);
        }else if(missing_response || number_missing > 0) {
          dp->set_missing_status(BOOM::Data::partly_missing);
        }
      }
    };

    class BinomialRegressionDataExtractor : public DataExtractor {
     public:
      // rdata is a list that contains three elements: an integer vector
      // named y containing the number of successes, an integer vector
      // named n containing the number of trials, and a matrix named x.
      virtual std::vector<Ptr<Data>> Extract(SEXP rdata) const {
        int *y = INTEGER(getListElement(rdata, "y"));
        int *n = INTEGER(getListElement(rdata, "n"));
        Matrix x = ToBoomMatrix(getListElement(rdata, "x"));
        int nobs = nrow(x);
        std::vector<Ptr<Data>> ans;
        ans.reserve(nobs);
        for (int i = 0; i < nobs; ++i) {
          bool y_is_missing(y[i] == NA_INTEGER);
          bool n_is_missing(n[i] == NA_INTEGER);
          int missing_preditors = count_missing(x.row(i));
          Ptr<Data> dp = new BinomialRegressionData(y[i], n[i], x.row(i));
          if((y_is_missing || n_is_missing) && missing_preditors == ncol(x)) {
            dp->set_missing_status(BOOM::Data::completely_missing);
          } else if(y_is_missing || n_is_missing || missing_preditors > 0) {
            dp->set_missing_status(BOOM::Data::partly_missing);
          }
          ans.push_back(dp);
        }
        return ans;
      }
    };

    // Factory method for creating data extractors based on data types.
    // Args:
    //   type: A string giving the name (as supplied by R) of the type of
    //   data to be extracted
    // Returns:
    //   A pointer to a heap allocated DataExtractor of the appropriate type.
    DataExtractor * DataExtractor::Create(const std::string &type) {
      if (type == "double.data") {
        return new DoubleDataExtractor;
      } else if (type == "int.data") {
        return new IntDataExtractor;
      } else if (type == "vector.data") {
        return new VectorDataExtractor;
      } else if (type == "regression.data") {
        return new RegressionDataExtractor;
      } else if (type == "binomial.regression.data") {
        return new BinomialRegressionDataExtractor;
      } else if (type == "factor.data") {
        return new FactorDataExtractor;
      } else if (type == "markov.data") {
        return new MarkovDataExtractor;
      }
      return 0;
    }

    // Extracts all the data from all the subjects in the given
    // rmixture_composite.
    // Args:
    //   rmixture_composite: An R object with named elements 'data' and
    //     'data.type'.  'data' is a list containing a time series of data
    //     structured in a way appropriate to the type described by the R
    //     text string 'data.type.'
    // Returns:
    //   A vector of time series, each time series is represented as a
    //   vector of Ptr's to the abstract Data type.  Each time series
    //   corresponds to one entry in 'data'.
    std::vector<std::vector<BOOM::Ptr<BOOM::Data>>> ExtractMixtureComponentData(
        SEXP rmixture_composite) {
      // No PROTECT is needed because data is protected by membership in
      // rmixture_composite.
      SEXP data = getListElement(rmixture_composite, "data");
      int number_of_subjects = Rf_length(data);
      std::vector<std::vector<Ptr<Data>>> ans;
      ans.reserve(number_of_subjects);
      std::string data_type = GetStringFromList(
          rmixture_composite, "data.type").c_str();
      std::unique_ptr<DataExtractor> data_extractor(
          DataExtractor::Create(data_type));
      for (int subject = 0; subject < number_of_subjects; ++subject) {
        ans.push_back(data_extractor->Extract(VECTOR_ELT(data, subject)));
      }
      return ans;
    }

    // Extracts the data from a list of mixture component specifications.
    // Args:
    //   rmixture_component_list: An R list of MixtureComponent
    //     objects.  Each list element defines one "dimension" of the
    //     data to be modeled, where "dimension" is in quotes because
    //     some dimensions might be multivariate (e.g. for a
    //     MultivariateNormal mixture component).  All mixture
    //     components must have the same number of observations.
    //
    // Returns:
    //   If the return value is ans, then ans[i] contains the data for
    //   subject i, so that ans[i][j] is the j'th observation for
    //   subject i.
    std::vector<std::vector<BOOM::Ptr<BOOM::Data>>>
    ExtractCompositeDataFromMixtureComponentList(SEXP rmixture_component_list){
      int number_of_composites = Rf_length(rmixture_component_list);
      // The indices of packed_composite_data are
      // [composite_number][subject][time].
      std::vector<std::vector<std::vector<Ptr<Data>>>> packed_composite_data(
          number_of_composites);
      int number_of_subjects = -1;
      for (int m = 0; m < number_of_composites; ++m) {
        packed_composite_data[m] = ExtractMixtureComponentData(
            VECTOR_ELT(rmixture_component_list, m));
        int current_number_of_subjects = packed_composite_data[m].size();
        if (number_of_subjects == -1) {
          number_of_subjects = current_number_of_subjects;
        }
        if (number_of_subjects != current_number_of_subjects
            || number_of_subjects < 0) {
          std::ostringstream err;
          err << "Mixture component composite " << m << " has "
              << current_number_of_subjects
              << " subjects, but earlier composites had "
              << number_of_subjects << endl;
          report_error(err.str().c_str());
        }
      }

      std::vector<std::vector<Ptr<Data>>> ans;
      ans.reserve(std::max<int>(number_of_subjects, 0));
      for (int subject = 0; subject < number_of_subjects; ++subject) {
        // This code block counts the number of observations for each
        // subject, and verifies that each subject has the same number
        // of observations on all data composites.
        int series_length = -1;
        for (int m = 0; m < number_of_composites; ++m) {
          int current_series_length = packed_composite_data[m][subject].size();
          if (series_length == -1) {
            series_length = current_series_length;
          }
          if (series_length != current_series_length || series_length < 0) {
            std::ostringstream err;
            err << "subject " << subject << " (counting from 0) has "
                << current_series_length << " observations from data composite "
                << m
                << " (also counting from 0).  Earlier data composites had "
                << series_length << " observations for this subject.";
            report_error(err.str());
            return ans;  // Never reached.
          }
        }
        std::vector<Ptr<Data>> subject_data;
        subject_data.reserve(std::max<int>(series_length, 0));
        for (int time = 0; time < series_length; ++time) {
          Ptr<CompositeData> dp = new CompositeData;
          for (int m = 0; m < number_of_composites; ++m) {
            dp->add(packed_composite_data[m][subject][time]);
          }
          subject_data.push_back(dp);
        }
        ans.push_back(subject_data);
      }
      return ans;
    }

    //======================================================================
    // Args:
    //   rknown_source: An R vector indicating which mixture component
    //     each individual observation belongs to, or NA if this
    //     information is unavailable.
    // Returns:
    //
    std::vector<int> UnpackKnownDataSource(SEXP rknown_source){
      std::vector<int> ans;
      if(Rf_isNewList(rknown_source)){
        // if it is a list, then unlist it
        int n = Rf_length(rknown_source);
        // We will need at least n elements in ans.  We might need
        // more.
        ans.reserve(n);
        for(int i = 0; i < n; ++i){
          Vector data = ToBoomVector(VECTOR_ELT(rknown_source, i));
          for(int j = 0; j < data.size(); ++j){
            double y = data[j];
            ans.push_back(R_IsNA(y) ? -1 : lround(y));
          }
        }
      }else if(Rf_isNumeric(rknown_source)){
        // if it is a numeric vector than unpack it
        Vector data = ToBoomVector(rknown_source);
        ans.reserve(data.size());
        for(int j = 0; j < data.size(); ++j){
          double y = data[j];
          ans.push_back(R_IsNA(y) ? -1 : lround(y));
        }
      }else{
        report_error("rknown_source must be either a "
                     "numeric or a list of numerics");
      }
      return ans;
    }

  }  // namespace RInterface
} // namespace BOOM
