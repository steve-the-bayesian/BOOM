#ifndef BOOM_DEFERRED_DATA_POLICY_HPP_
#define BOOM_DEFERRED_DATA_POLICY_HPP_

/*
  Copyright (C) 2005-2018 Steven L. Scott

  This library is free software; you can redistribute it and/or modify it under
  the terms of the GNU Lesser General Public License as published by the Free
  Software Foundation; either version 2.1 of the License, or (at your option)
  any later version.

  This library is distributed in the hope that it will be useful, but WITHOUT
  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
  FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
  details.

  You should have received a copy of the GNU Lesser General Public License along
  with this library; if not, write to the Free Software Foundation, Inc., 51
  Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA
*/

namespace BOOM {

  // A data policy to be used when the data for a model is acutally owned by a
  // single sub-model.  For example if a model class was a wrapper around an
  // implementation class.
  //
  // Classes that inherit from this class must define their own copy
  // constructors and assignment operators, which must call set_model() to pass
  // the implementation model.  Move construction and assignment can be defaulted.
  class DeferredDataPolicy : virtual public Model {
   public:
    typedef DeferredDataPolicy DataPolicy;

    DeferredDataPolicy() {}
    explicit DeferredDataPolicy(const Ptr<Model> &model) : model_(model) {}

    // Copy and assignment reset the model pointer to nullptr, with the
    // expectation that it will be set in the downstream implementation using
    // set_model.
    DeferredDataPolicy(const DeferredDataPolicy &rhs)
        : Model(rhs),
          model_(nullptr)
    {}

    DeferredDataPolicy & operator=(const DeferredDataPolicy &rhs) {
      if (&rhs != this) {
        Model::operator=(rhs);
        model_ = nullptr;
      }
      return *this;
    }

    DeferredDataPolicy(DeferredDataPolicy &&rhs) = default;
    DeferredDataPolicy & operator=(DeferredDataPolicy &&rhs) = default;

    void set_model(const Ptr<Model> &model) {
      model_ = model;
    }

    void add_data(const Ptr<Data> &dp) override {
      check_model("add_data");
      model_->add_data(dp);
    }

    void clear_data() override {
      if (!!model_) {
        model_->clear_data();
      }
    }

    void combine_data(const Model &other_model, bool just_suf = true) override {
      check_model("combine_data");
      model_->combine_data(other_model, just_suf);
    }

   private:
    // If the model pointer is nullptr then report an error.
    // Args:
    //   msg:  The name of the function in which the error occurred.
    void check_model(const char *msg) {
      if (!model_) {
        std::ostringstream err;
        err << "No model was assigned to DeferredDataPolicy prior to the call: "
            << *msg;
        report_error(err.str());
      }
    }
    Ptr<Model> model_;
  };

}

#endif //  BOOM_DEFERRED_DATA_POLICY_HPP_
