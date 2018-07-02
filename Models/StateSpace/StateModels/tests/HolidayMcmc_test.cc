#include "gtest/gtest.h"
#include "Models/ChisqModel.hpp"
#include "Models/PosteriorSamplers/ZeroMeanGaussianConjSampler.hpp"
#include "Models/GaussianModel.hpp"

#include "Models/StateSpace/StateSpaceModel.hpp"
#include "Models/StateSpace/PosteriorSamplers/StateSpacePosteriorSampler.hpp"
#include "Models/StateSpace/StateModels/LocalLevelStateModel.hpp"
#include "Models/StateSpace/StateModels/RandomWalkHolidayStateModel.hpp"
#include "Models/StateSpace/StateModels/RegressionHolidayStateModel.hpp"
#include "Models/StateSpace/StateModels/HierarchicalRegressionHolidayStateModel.hpp"
#include "Models/StateSpace/StateModels/Holiday.hpp"
#include "cpputil/Date.hpp"

#include "distributions.hpp"
#include "cpputil/Date.hpp"
#include <fstream>


namespace {

  using namespace BOOM;
  
  class HolidayMcmcTest : public ::testing::Test {
   protected:
    HolidayMcmcTest() {
      GlobalRng::rng.seed(8675309);
    }
  };

  
  TEST_F(HolidayMcmcTest, Mcmc) {
    int nobs = 2000;  // about 5.5 years
    Vector trend(nobs);
    double mu = 0;
    for (int i = 0; i < nobs; ++i) {
      mu = mu + rnorm(0, .1);
      trend[i] = mu;
    }

    std::vector<Date> dates(nobs);
    dates[0] = Date(Jan, 1, 2004);
    for (int i = 1; i < dates.size(); ++i) {
      dates[i] = dates[i - 1] + 1;
    }
    std::cout << "The last date is " << dates.back() << std::endl;
    
    std::vector<Ptr<Holiday>> holidays;
    NEW(NewYearsDay, new_years_day)(1, 1);
    Vector new_years_effect = {.5, 1, .25};
    holidays.push_back(new_years_day);

    NEW(FixedDateHoliday, my_bday)(Jan, 14, 1, 1);
    Vector bday_effect = {.28, 1.2, .29};
    holidays.push_back(my_bday);

    NEW(SaintPatricksDay, spday)(1, 1);
    Vector spday_effect = {.1, .25, .1};
    holidays.push_back(spday);

    std::vector<Date> eid_al_fitr_start =
        {Date(Nov, 13, 2004),
         Date(Nov, 2, 2005),
         Date(Oct, 23, 2006),
         Date(Oct, 12, 2007),
         Date(Oct, 1, 2008),
         Date(Sep, 20, 2009),
         Date(Sep, 10, 2010)};

    std::vector<Date> eid_al_fitr_end = {
      Date(Nov, 16, 2004),
      Date(Nov, 5, 2005),
      Date(Oct, 26, 2006),
      Date(Oct, 15, 2007),
      Date(Oct, 4, 2008),
      Date(Sep, 23, 2009),
      Date(Sep, 13, 2010)};

    NEW(DateRangeHoliday, eid_al_fitr)(
        eid_al_fitr_start,
        eid_al_fitr_end);
    holidays.push_back(eid_al_fitr);
    Vector eid_al_fitr_effect = {.1, .2, 3, .2, .1};

    Vector y(nobs);
    for (int i = 0; i < nobs; ++i) {
      y[i] = trend[i] + rnorm(0, 1);
      const Date &date(dates[i]);
      if (new_years_day->active(date)) {
        int d = new_years_day->days_into_influence_window(date);
        ASSERT_GE(d, 0)
            << "d = " << d << endl
            << "new_years_effect = " << new_years_effect;
        ASSERT_LT(d, new_years_effect.size())
            << "d = " << d << endl
            << "new_years_effect = " << new_years_effect;
        y[i] += new_years_effect[d];
      }

      if (my_bday->active(date)) {
        int d = my_bday->days_into_influence_window(date);
        y[i] += bday_effect[d];
      }

      if (spday->active(date)) {
        int d = spday->days_into_influence_window(date);
        y[i] += spday_effect[d];
      }

      if (eid_al_fitr->active(date)) {
        int d = eid_al_fitr->days_into_influence_window(date);
        y[i] += eid_al_fitr_effect[d];
      }
    }

    NEW(StateSpaceModel, model)(y);

    NEW(LocalLevelStateModel, trend_model);
    NEW(ChisqModel, trend_precision_prior)(1, 1);
    NEW(ZeroMeanGaussianConjSampler, trend_sampler)(
      trend_model.get(), trend_precision_prior);
    trend_model->set_method(trend_sampler);
    model->add_state(trend_model);

    NEW(GaussianModel, holiday_effect_prior)(0, 2);
    NEW(ScalarRegressionHolidayStateModel, holiday_model)(
        dates[0], model.get(), holiday_effect_prior);
    for (int i = 0; i < holidays.size(); ++i) {
      holiday_model->add_holiday(holidays[i]);
    }
    model->add_state(holiday_model);

    NEW(ChisqModel, observation_precision_prior)(1, 1);
    NEW(ZeroMeanGaussianConjSampler, observation_sampler)(
      model->observation_model(), observation_precision_prior);
    
    NEW(StateSpacePosteriorSampler, sampler)(model.get());
    model->set_method(sampler);

    for (int i = 0; i < 100; ++i) {
      model->sample_posterior();
    }
  }
  
}  // namespace


