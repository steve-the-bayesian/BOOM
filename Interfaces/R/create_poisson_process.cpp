/*
  Copyright (C) 2005-2012 Steven L. Scott

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

#include <algorithm>
#include <cpputil/report_error.hpp>

#include <Models/PointProcess/PoissonProcess.hpp>
#include <Models/PointProcess/HomogeneousPoissonProcess.hpp>
#include <Models/PointProcess/WeeklyCyclePoissonProcess.hpp>
#include <Models/PointProcess/PosteriorSamplers/HomogeneousPoissonProcessPosteriorSampler.hpp>
#include <Models/PointProcess/PosteriorSamplers/WeeklyCyclePoissonProcessSampler.hpp>

#include <r_interface/prior_specification.hpp>
#include <r_interface/boom_r_tools.hpp>
#include <r_interface/list_io.hpp>

namespace BOOM {
  namespace RInterface {

      namespace {
        class PoissonProcessFactory {
         public:
          explicit PoissonProcessFactory(RListIoManager *io_manager)
              : io_manager_(io_manager)
          {}
          Ptr<PoissonProcess> create(SEXP r_poisson_process,
                                     const std::string &name);
         private:
          Ptr<PoissonProcess> create_homogeneous(SEXP r_poisson_process,
                                                 const std::string &name);
          Ptr<PoissonProcess> create_weekly_cycle(SEXP r_poisson_process,
                                                  const std::string &name);
          RListIoManager *io_manager_;
        };

        // Create a PoissonProcess of the appropriate concrete type
        // based on the R class of the first argument
        Ptr<PoissonProcess> PoissonProcessFactory::create(
            SEXP r_poisson_process, const std::string &name) {
          if (Rf_inherits(
                  r_poisson_process, "HomogeneousPoissonProcess")) {
            return create_homogeneous(r_poisson_process, name);
          } else if (Rf_inherits(
              r_poisson_process, "WeeklyCyclePoissonProcess")) {
            return create_weekly_cycle(r_poisson_process, name);
          } else if (Rf_inherits(
              r_poisson_process, "PoissonProcess")) {
            std::vector<std::string> class_names =
                GetS3Class(r_poisson_process);
            // class_names will have at least one entry because
            // r_poisson_process inherits from PoissonProcess, so
            // class_names[0] can be accessed.
            std::ostringstream err;
            err << "PoissonProcessFactory does not know how to create a "
                << "PoissonProcess from an object of (R) class "
                << class_names[0] << "." << std::endl;
            report_error(err.str());
          } else {
            report_error("The first argument to PoissonProcessFactory::create"
                         "  must inherit from PoissonProcess.");
          }
          return Ptr<PoissonProcess>(0);
        }

        //  Create a HomogeneousPoissonProcess.
        Ptr<PoissonProcess> PoissonProcessFactory::create_homogeneous(
            SEXP r_poisson_process, const std::string &name) {
          SEXP r_prior = getListElement(r_poisson_process, "prior");
          BOOM::RInterface::GammaPrior prior_specification(r_prior);
          double initial_lambda = prior_specification.initial_value();
          NEW(BOOM::HomogeneousPoissonProcess, ans)(initial_lambda);
          NEW(BOOM::GammaModel, lambda_prior)(prior_specification.a(),
                                              prior_specification.b());
          NEW(HomogeneousPoissonProcessPosteriorSampler, sampler)(
              ans.get(),
              lambda_prior);
          ans->set_method(sampler);

          std::ostringstream lambda_name;
          lambda_name << name << ".lambda";
          io_manager_->add_list_element(new UnivariateListElement(
              ans->Lambda_prm(), lambda_name.str()));
          return ans;
        }

        Ptr<PoissonProcess> PoissonProcessFactory::create_weekly_cycle(
            SEXP r_poisson_process, const std::string &process_name) {
          NEW(WeeklyCyclePoissonProcess, ans)();

          BOOM::RInterface::GammaPrior lambda_prior_specification(
              getListElement(r_poisson_process, "average.daily.rate.prior"));
          BOOM::RInterface::DirichletPrior daily(
              getListElement(r_poisson_process, "daily.dirichlet.prior"));
          BOOM::RInterface::DirichletPrior weekday(
              getListElement(r_poisson_process,
                             "weekday.hourly.dirichlet.prior"));
          BOOM::RInterface::DirichletPrior weekend(
              getListElement(r_poisson_process,
                             "weekend.hourly.dirichlet.prior"));

          NEW(GammaModel, average_daily_rate_prior)(
              lambda_prior_specification.a(), lambda_prior_specification.b());
          NEW(DirichletModel, daily_prior)(daily.prior_counts());
          NEW(DirichletModel, weekday_prior)(weekday.prior_counts());
          NEW(DirichletModel, weekend_prior)(weekend.prior_counts());
          NEW(WeeklyCyclePoissonProcessSampler, sampler)(
              ans.get(),
              average_daily_rate_prior,
              daily_prior,
              weekday_prior,
              weekend_prior);
          ans->set_method(sampler);

          // io_manager_
          std::ostringstream lambda_name;
          lambda_name << process_name << ".average.daily.rate";
          io_manager_->add_list_element(new UnivariateListElement(
              ans->average_daily_event_rate_prm(), lambda_name.str()));

          std::ostringstream day_of_week_name;
          day_of_week_name << process_name << ".day.of.week";
          io_manager_->add_list_element(new VectorListElement(
              ans->day_of_week_cycle_prm(), day_of_week_name.str()));

          std::ostringstream weekday_name;
          weekday_name << process_name << ".weekday.hourly";
          io_manager_->add_list_element(new VectorListElement(
              ans->weekday_hour_of_day_cycle_prm(), weekday_name.str()));

          std::ostringstream weekend_name;
          weekend_name << process_name << ".weekend.hourly";
          io_manager_->add_list_element(new VectorListElement(
              ans->weekend_hour_of_day_cycle_prm(), weekend_name.str()));

          return ans;
        }

      }  // namespace
    Ptr<PoissonProcess> CreatePoissonProcess(SEXP r_poisson_process,
                                             RListIoManager *io_manager,
                                             const std::string &name) {
      PoissonProcessFactory factory(io_manager);
      return factory.create(r_poisson_process, name);
    }

  }  // namespace RInterface
}  // namespace BOOM
