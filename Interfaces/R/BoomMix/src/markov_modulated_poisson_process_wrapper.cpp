#include <set>

#include <r_interface/boom_r_tools.hpp>
#include <r_interface/create_mixture_component.hpp>
#include <r_interface/create_point_process.hpp>
#include <r_interface/create_poisson_process.hpp>
#include <r_interface/seed_rng_from_R.hpp>
#include <r_interface/print_R_timestamp.hpp>
#include <r_interface/list_io.hpp>
#include <r_interface/handle_exception.hpp>

#include <Models/PointProcess/HomogeneousPoissonProcess.hpp>
#include <Models/PointProcess/HomogeneousPoissonProcess.hpp>
#include <Models/PointProcess/MarkovModulatedPoissonProcess.hpp>
#include <Models/PointProcess/PosteriorSamplers/MmppPosteriorSampler.hpp>
#include <cpputil/report_error.hpp>

namespace BOOM {
  namespace RInterface {

    class MmppLoglikelihoodGetter : public BOOM::ScalarIoCallback {
     public:
      explicit MmppLoglikelihoodGetter(MarkovModulatedPoissonProcess *mmpp)
          : mmpp_(mmpp) {}
      virtual double get_value() const {
        return mmpp_->last_loglike();
      }
     private:
      MarkovModulatedPoissonProcess *mmpp_;
    };

    // A class for building a MarkovModulatedPoissonProcess from R inputs.
    class MmppFactory {
     public:
      // Args:
      //   r_point_process_list: A list of R objects, each inheriting
      //     from "PointProcess".  Each list element contains the point
      //     process information for a single subject (user, etc.).
      //   r_process_specification: A list of R objects of class
      //     PoissonProcessComponent.  These objets define the
      //     component processes in the MMPP, which processes they
      //     turn on and off, and the name of the associated mixture
      //     component.
      //   r_initial.state: A character vector containing three of the
      //     names from process.specification, defining any valid
      //     state in the MMPP.  The model will use this information
      //     to determine all the other states.
      //   r_mixture_components: A list of R objects inheriting from
      //     "MixtureComponent" carrying data for any marks assiciated
      //     with the point processes.  This argument can be (R-)NULL,
      //     in which case no marks are present.
      //   r_known_source: This is either NULL, if there is no known
      //     source data available, or else it is a list with the same
      //     dimensions as r_point_process_list.  The list elements
      //     are lists of character vectors naming either
      //     PoissonProcesses or MixtureComponents responsible for the
      //     event at time t for subject i.
      MmppFactory(SEXP r_point_process_list,
                  SEXP r_process_specification,
                  SEXP r_initial_state,
                  SEXP r_mixture_components,
                  SEXP r_known_source,
                  BOOM::RListIoManager *io_manager)
          : io_manager_(io_manager) {
        UnpackComponentProcesses(r_process_specification, r_initial_state);
        UnpackMixtureComponents(r_mixture_components);
        UnpackData(r_point_process_list, r_mixture_components);
        UnpackKnownSource(r_known_source);
      }

      // Create an MMPP model based on the information provided to the
      // constructor.
      Ptr<MarkovModulatedPoissonProcess> CreateMmpp() {
        MarkovModulatedPoissonProcess *mmpp =
            new MarkovModulatedPoissonProcess;
        AssignComponentProcesses(mmpp);
        AssignData(mmpp);
        NEW(MmppPosteriorSampler, sam)(mmpp);
        mmpp->set_method(sam);

        // Store the draws of log likelihood.
        io_manager_->add_list_element(new BOOM::NativeUnivariateListElement(
            new MmppLoglikelihoodGetter(mmpp),
            "log.likelihood",
            NULL));
        return mmpp;
      }

     private:
      // The "UnpackXXX" functions are each responsible for building
      // different elements of state as part of implementing the
      // constructor.
      void UnpackComponentProcesses(SEXP r_process_specification,
                                    SEXP r_initial_state);
      void UnpackMixtureComponents(SEXP r_mixture_components);
      void UnpackData(SEXP r_point_process_list,
                      SEXP r_mixture_components);
      void UnpackKnownSource(SEXP r_known_source);

      // The AssignXXX functions are used to help implement the
      // CreateMmpp method.
      void AssignComponentProcesses(MarkovModulatedPoissonProcess *mmpp);
      void AssignData(MarkovModulatedPoissonProcess *mmpp);

      // Returns the process id of the process named 'process_name'.
      // Returns -1 if the name was not found.
      int pid(const std::string & process_name) const;
      std::vector<PoissonProcess *> GetSources(
          const std::vector<std::string> &names);
      std::vector<PoissonProcess *> GetCorrespondingPoissonProcesses(
          const std::string & mixture_component_name);

      // The name of each mixture component.  This can be empty if
      // there are no mixture components.
      std::vector<std::string> mixture_component_names_;

      // The position of each process in component_processes_.
      std::map<std::string, int> process_index_;

      // The processes comprising the MMPP, and the corresponding set
      // of processes spawned and killed by an event from each.
      std::vector<Ptr<PoissonProcess> > component_processes_;
      std::vector<std::vector<Ptr<PoissonProcess> > > spawns_;
      std::vector<std::vector<Ptr<PoissonProcess> > > kills_;
      std::vector<Ptr<PoissonProcess> > initial_state_members_;

      // The the mixture component associated with each unique name in
      // mixture_component_names_.  This can be empty if there are no
      // mixture components.
      std::map<std::string, Ptr<MixtureComponent> > mixture_components_;

      // The data to be modeled.
      std::vector<Ptr<PointProcess> > data_;

      // Ground truth labels, if any are available.
      std::vector<MarkovModulatedPoissonProcess::SourceVector> known_source_;

      // The mapping between mixture component names and the set of
      // processes associated with those mixture components.
      std::map<std::string, std::vector<PoissonProcess *> >
      processes_with_mixture_component_named_;

      // The io_manager responsible for recording or streaming the
      // MCMC draws.
      BOOM::RListIoManager *io_manager_;
    };

    //----------------------------------------------------------------------
    // Returns the process id of the process named 'process_name'.
    int MmppFactory::pid(const std::string &process_name) const {
      std::map<std::string, int>::const_iterator it =
          process_index_.find(process_name);
      return (it == process_index_.end()) ? -1 : it->second;
    }

    //----------------------------------------------------------------------
    // A partial implementation of the MmppFactory constructor.
    // Args:
    //   r_process_specification: The R object detailing the set of
    //     component processes in the MMPP.  This is a list, where
    //     each entry contains a PoissonProcess specification, the
    //     names of the processes it spawns and kills, and the name of
    //     the mixture component (if any) it is associated with.
    // Returns:
    //   void.  On exit the fields 'process_name', 'process_index_',
    //   'component_processes_', 'spawns_', and 'kills_' will be
    //   populated.
    void MmppFactory::UnpackComponentProcesses(SEXP r_process_specification,
                                               SEXP r_initial_state) {
      int nproc = Rf_length(r_process_specification);
      component_processes_.reserve(nproc);
      std::vector<std::string> process_names =
          BOOM::getListNames(r_process_specification);

      // Step 1 is to to build the list of component processes and
      // associate them with their names.
      for (int i = 0; i < nproc; ++i) {
        SEXP r_component = VECTOR_ELT(r_process_specification, i);
        component_processes_.push_back(CreatePoissonProcess(
            getListElement(r_component, "process"),
            io_manager_,
            process_names[i]));
        process_index_[process_names[i]] = i;
        SEXP r_mixture_component_name =
            getListElement(r_component, "mixture.component");
        if (Rf_length(r_mixture_component_name) == 0) {
          // What to do if there are no mixture components?
        } else {
          mixture_component_names_.push_back(CHAR(STRING_ELT(
              r_mixture_component_name, 0)));
        }
      }

      // Step 2 is to build the vectors 'spawns' and 'kills'.
      spawns_.reserve(nproc);
      kills_.reserve(nproc);
      for (int i = 0; i < nproc; ++i) {
        SEXP r_component = VECTOR_ELT(r_process_specification, i);
        std::vector<Ptr<PoissonProcess> > this_process_spawns;
        std::vector<std::string> spawn_names = StringVector(
            getListElement(r_component, "spawns"));
        this_process_spawns.reserve(spawn_names.size());
        for (int i = 0; i < spawn_names.size(); ++i) {
          this_process_spawns.push_back(
              component_processes_[pid(spawn_names[i])]);
        }
        spawns_.push_back(this_process_spawns);

        std::vector<Ptr<PoissonProcess> > this_process_kills;
        std::vector<std::string> kill_names = StringVector(
            getListElement(r_component, "kills"));
        this_process_kills.reserve(kill_names.size());
        for (int i = 0; i < kill_names.size(); ++i) {
          this_process_kills.push_back(
              component_processes_[pid(kill_names[i])]);
        }
        kills_.push_back(this_process_kills);
      }

      // Step 3 is to fill the vector of initial_state_members_.
      std::vector<std::string> initial_state_names =
          StringVector(r_initial_state);
      initial_state_members_.reserve(initial_state_names.size());
      for (int i = 0; i < initial_state_names.size(); ++i) {
        initial_state_members_.push_back(
            component_processes_[pid(initial_state_names[i])]);
      }
    }

    //----------------------------------------------------------------------
    // A partial implementation of the MmppFactory constructor.
    // Args:
    //   r_mixture_components: An R list containing the information
    //     needed to build the mixture components for the model.  This
    //     can be an R NULL object if no mixture components are
    //     present.
    //
    // Returns:
    //   void.  On exit the object 'mixture_components_' is populated
    //     if r_mixture_components was not NULL.  Otherwise it is left
    //     empty.
    void MmppFactory::UnpackMixtureComponents(SEXP r_mixture_components) {
      if (Rf_isNull(r_mixture_components)
          || mixture_component_names_.empty()) {
        return;
      }
      // To make the name strings unique we dump them into a set and
      // then dump them back out.  This is slightly wasteful, but
      // seems easier and more readable than a sequence of calls to
      // the std library algorithms sort, unique, and erase that would
      // be required to accomplish the same thing.
      std::set<std::string> set_of_mixture_component_names(
          mixture_component_names_.begin(),
          mixture_component_names_.end());

      std::vector<std::string>
          minimal_mixture_component_names(
              set_of_mixture_component_names.begin(),
              set_of_mixture_component_names.end());

      mixture_components_ = UnpackNamedCompositeMixtureComponents(
          r_mixture_components,
          minimal_mixture_component_names,
          io_manager_);
    }

    //----------------------------------------------------------------------
    // A component of the MmppFactory constructor.  Extracts the point
    // process start, end, and event times, and any associated mark
    // data that go along with them.
    // Args:
    //   r_point_process_list: A list of R objects, each inheriting
    //     from "PointProcess".  Each list element contains the point
    //     process information for a single subject (user, etc.).
    //   r_mixture_components: A list of R objects inheriting from
    //     "MixtureComponent" carrying data for any marks assiciated
    //     with the point processes.  This argument can be (R-)NULL,
    //     in which case no marks are present.
    // Returns:
    //   void.  On exit the data_ member variable is populated.  Each
    //   element of data_ contains the point process information for a
    //   single subject, including mark (if any).
    void MmppFactory::UnpackData(
        SEXP r_point_process_list, SEXP r_mixture_components) {
      bool have_mixture_components = !Rf_isNull(r_mixture_components);
      std::vector<std::vector<Ptr<Data> > > mixture_data;
      if (have_mixture_components) {
        mixture_data = ExtractCompositeDataFromMixtureComponentList(
            r_mixture_components);
      }

      int number_of_subjects = Rf_length(r_point_process_list);
      if (have_mixture_components &&
          (number_of_subjects != mixture_data.size())) {
        ostringstream err;
        err << "There were " << number_of_subjects << " point processes, but "
            << mixture_data.size() << " groups of mixture component data.";
        report_error(err.str());
      }

      data_.reserve(number_of_subjects);
      for (int subject = 0; subject < number_of_subjects; ++subject) {
        SEXP r_point_process = VECTOR_ELT(r_point_process_list, subject);
        if (have_mixture_components) {
          data_.push_back(CreatePointProcess(
              r_point_process, mixture_data[subject]));
        } else {
          data_.push_back(CreatePointProcess(r_point_process));
        }
      }
    }

    //----------------------------------------------------------------------
    // Unpacks the vector of known source information, and populates
    // the known_source_ data element.
    void MmppFactory::UnpackKnownSource(SEXP r_known_source) {
      // Bail out and leave the whole vector empty if there is no
      // known source data anywhere.
      if (Rf_isNull(r_known_source)) return;

      int number_of_subjects = data_.size();
      known_source_.reserve(number_of_subjects);
      for (int i = 0; i < number_of_subjects; ++i) {
        // Get the list of character vectors for subject i.
        SEXP r_known_source_i = VECTOR_ELT(r_known_source, i);
        MarkovModulatedPoissonProcess::SourceVector known_source_i;
        int number_of_events = Rf_length(r_known_source_i);
        for (int t = 0; t < number_of_events; ++t) {
          std::vector<std::string> source_names =
              StringVector(VECTOR_ELT(r_known_source_i, t));
          std::vector<PoissonProcess *> sources = GetSources(source_names);
          known_source_i.push_back(sources);
        }
        known_source_.push_back(known_source_i);
      }
    }

    //----------------------------------------------------------------------
    // Given a vector of names, return the set of PoissonProcesses to
    // which they belong.
    //
    // Example: If p0, p3, and p7 produce events of type "blue", p1,
    // p4, and p6 produce events of type "green, and p2 and p5 produce
    // events of type "yellow", then passing ["blue" "yellow"] returns
    // (p0, p3, p7, p2, p5).
    std::vector<PoissonProcess *> MmppFactory::GetSources(
        const std::vector<std::string> &names) {
      std::vector<PoissonProcess *> ans;
      ans.reserve(names.size());
      for (int i = 0; i < names.size(); ++i) {
        int index = pid(names[i]);
        if (index > 0) {
          ans.push_back(component_processes_[pid(names[i])].get());
        } else {
          std::vector<PoissonProcess *> sources =
              GetCorrespondingPoissonProcesses(names[i]);
          std::vector<PoissonProcess *> tmp;
          std::merge(ans.begin(), ans.end(),
                     sources.begin(), sources.end(),
                     std::back_inserter(tmp));
          ans = tmp;
        }
      }
      return make_unique_inplace(ans);
    }

    //----------------------------------------------------------------------
    // Returns the vector of PoissonProcess pointers corresponding to
    // mixture components named 'mixture_component_name'
    std::vector<PoissonProcess *> MmppFactory::GetCorrespondingPoissonProcesses(
        const std::string &mixture_component_name) {
      std::map<std::string, std::vector<PoissonProcess *> >::iterator it =
          processes_with_mixture_component_named_.find(mixture_component_name);
      if (it != processes_with_mixture_component_named_.end()) {
        return it->second;
      }
      // If the name was not found...
      std::vector<PoissonProcess *> ans;
      for (int i = 0; i < mixture_component_names_.size(); ++i) {
        if (mixture_component_names_[i] == mixture_component_name) {
          ans.push_back(component_processes_[i].get());
        }
      }
      std::sort(ans.begin(), ans.end());
      processes_with_mixture_component_named_[mixture_component_name] = ans;
      return ans;
    }

    void MmppFactory::AssignComponentProcesses(
        MarkovModulatedPoissonProcess *mmpp) {
      bool have_mixture_components = !(mixture_components_.empty());
      for (int i = 0; i < component_processes_.size(); ++i) {
        Ptr<MixtureComponent> mix;
        if (have_mixture_components) {
          const std::string &name = mixture_component_names_[i];
          mix = mixture_components_[name];
        }
        mmpp->add_component_process(
            component_processes_[i],
            spawns_[i],
            kills_[i],
            mix);
      }
      mmpp->make_hmm_states(initial_state_members_);
    }

    // Assigns data to the model.
    void MmppFactory::AssignData(MarkovModulatedPoissonProcess *mmpp) {
      bool have_known_source = !(known_source_.empty());
      for (int i = 0; i < data_.size(); ++i) {
        if (have_known_source) {
          mmpp->add_supervised_data(data_[i], known_source_[i]);
        } else {
          mmpp->add_data(data_[i]);
        }
      }
    }

    // Append in the probability of activity and the probability of
    // responsibility.
    //
    // Args:
    //   r_input_list:  An R list to which the probabilities will be appended.
    //   mmpp:  The object generating the probabilities to be stored.
    //
    // Returns:
    //   An R list containing the elements of r_input_list, with
    //   "prob.active" and "prob.responsible" appended at the end.
    SEXP AppendProbabilities(
        SEXP r_input_list,
        const MarkovModulatedPoissonProcess &mmpp) {
      int nproc = mmpp.dat().size();
      SEXP r_probability_of_responsibility;
      PROTECT(r_probability_of_responsibility = Rf_allocVector(VECSXP, nproc));
      for (int i = 0; i < nproc; ++i) {
        BOOM::Matrix pr = mmpp.probability_of_responsibility(i);

        SET_VECTOR_ELT(r_probability_of_responsibility,
                       i,
                       ToRMatrix(pr));
      }

      SEXP r_probability_of_activity;
      PROTECT(r_probability_of_activity = Rf_allocVector(VECSXP, nproc));
      for (int i = 0; i < nproc; ++i) {
        SET_VECTOR_ELT(r_probability_of_activity,
                       i,
                       ToRMatrix(mmpp.probability_of_activity(i)));
      }

      int input_list_size = Rf_length(r_input_list);
      SEXP r_ans;
      PROTECT(r_ans = Rf_allocVector(VECSXP, input_list_size + 2));
      for (int i = 0; i < input_list_size; ++i) {
        SET_VECTOR_ELT(r_ans, i, VECTOR_ELT(r_input_list, i));
      }
      SET_VECTOR_ELT(r_ans, input_list_size, r_probability_of_activity);
      SET_VECTOR_ELT(r_ans, input_list_size + 1, r_probability_of_responsibility);

      SEXP r_input_list_names = Rf_getAttrib(r_input_list, R_NamesSymbol);
      SEXP r_ans_names;
      PROTECT(r_ans_names = Rf_allocVector(STRSXP, input_list_size + 2));
      for (int i = 0; i < input_list_size; ++i) {
        SET_STRING_ELT(r_ans_names, i, STRING_ELT(r_input_list_names, i));
      }
      SET_STRING_ELT(r_ans_names, input_list_size, Rf_mkChar("prob.active"));
      SET_STRING_ELT(r_ans_names, input_list_size + 1,
                     Rf_mkChar("prob.responsible"));

      Rf_namesgets(r_ans, r_ans_names);
      UNPROTECT(4);
      return r_ans;
    }

  }   // namespace RInterface
}  // namespace BOOM

extern "C" {
  SEXP markov_modulated_poisson_process_wrapper_(
      SEXP r_point_process_list,
      SEXP r_process_specification,
      SEXP r_initial_state,
      SEXP r_mixture_components,
      SEXP r_known_source,
      SEXP r_niter,
      SEXP r_ping,
      SEXP r_seed) {
    using namespace BOOM;
    try {
      BOOM::RListIoManager io_manager;
      BOOM::RInterface::seed_rng_from_R(r_seed);

      BOOM::RInterface::MmppFactory factory(r_point_process_list,
                                            r_process_specification,
                                            r_initial_state,
                                            r_mixture_components,
                                            r_known_source,
                                            &io_manager);
      Ptr<MarkovModulatedPoissonProcess> mmpp = factory.CreateMmpp();
      int niter = Rf_asInteger(r_niter);
      int ping = Rf_asInteger(r_ping);
      SEXP r_ans;
      PROTECT(r_ans = io_manager.prepare_to_write(niter));
      for (int i = 0; i < niter; ++i) {
        // TODO(stevescott): There is a potentially large memory leak
        // here as boom objects will not be freed.
        R_CheckUserInterrupt();
        BOOM::print_R_timestamp(i, ping);
        mmpp->sample_posterior();
        io_manager.write();
      }

      // Append in the probability of activity and the probability of
      // responsibility.
      r_ans = BOOM::RInterface::AppendProbabilities(r_ans, *mmpp);
      UNPROTECT(1);
      return r_ans;
    } catch(std::exception &e) {
      BOOM::RInterface::handle_exception(e);
    } catch(...) {
      BOOM::RInterface::handle_unknown_exception();
    }
    return R_NilValue;
  }
}  // extern "C"
