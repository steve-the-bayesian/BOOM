/*
  Copyright (C) 2005 Steven L. Scott

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

#include "Models/IRT/IrtModel.hpp"

#include <algorithm>
#include <cstring>
#include <string>
#include <fstream>
#include <functional>
#include <iomanip>

#include "LinAlg/CorrelationMatrix.hpp"
#include "Models/IRT/Item.hpp"
#include "Models/IRT/Subject.hpp"
#include "Models/IRT/SubjectPrior.hpp"
#include "Models/MvnModel.hpp"
#include "Models/PosteriorSamplers/PosteriorSampler.hpp"
#include "cpputil/math_utils.hpp"
#include "cpputil/random_element.hpp"
#include "cpputil/string_utils.hpp"

namespace BOOM {
  namespace IRT {

    typedef std::vector<IrtModel::ModelTypeName> ModelVec;

    typedef std::vector<std::string> StringVector;

    inline void set_default_names(StringVector &s) {
      for (uint i = 0; i < s.size(); ++i) {
        std::ostringstream out;
        out << "subscale[" << i << "]";
        s[i] = out.str();
      }
    }
    //------------------------------------------------------------
    IrtModel::IrtModel()
        : subscale_names_(1),
          theta_freq(1),
          item_freq(1),
          R_freq(1),
          niter(0),
          theta_suppressed(false),
          subject_subset(0),
          subject_search_helper(new Subject("", 1)),
          item_search_helper(new NullItem) {
      set_default_names(subscale_names_);
    }

    //------------------------------------------------------------
    IrtModel::IrtModel(uint nsub)
        : subscale_names_(nsub),
          //      response_prototype(new ordinal_data<uint>),
          theta_freq(1),
          item_freq(1),
          R_freq(1),
          niter(0),
          theta_suppressed(false),
          subject_subset(0),
          subject_search_helper(new Subject("", 1)),
          item_search_helper(new NullItem) {
      set_default_names(subscale_names_);
    }

    //------------------------------------------------------------
    IrtModel::IrtModel(const StringVector &SubscaleNames)
        : subscale_names_(SubscaleNames),
          theta_freq(1),
          item_freq(1),
          R_freq(1),
          niter(0),
          theta_suppressed(false),
          subject_subset(0),
          subject_search_helper(new Subject("", 1)),
          item_search_helper(new NullItem) {}

    IrtModel::IrtModel(const IrtModel &rhs)
        : Model(rhs), ParamPolicy(rhs), DataPolicy(rhs), PriorPolicy(rhs) {
      report_error("need to implement copy constructor for IrtModel");
    }

    //------------------------------------------------------------
    IrtModel *IrtModel::clone() const { return new IrtModel(*this); }
    //------------------------------------------------------------
    double IrtModel::pdf(const Ptr<Subject> &s, bool logscale) const {
      const ItemResponseMap &resp(s->item_responses());
      double ans = 0;
      for (IrIterC it = resp.begin(); it != resp.end(); ++it) {
        Ptr<Item> item = it->first;
        Response r = it->second;
      }
      report_error("need to implement 'pdf' for IrtModel");
      return logscale ? ans : exp(ans);
    }
    //------------------------------------------------------------

    double IrtModel::pdf(const Ptr<Data> &dp, bool logscale) const {
      return pdf(DAT(dp), logscale);
    }
    //------------------------------------------------------------
    void IrtModel::set_subscale_names(const StringVector &names) {
      subscale_names_ = names;
    }

    //------------------------------------------------------------
    const StringVector &IrtModel::subscale_names() { return subscale_names_; }

    //------------------------------------------------------------
    inline uint find_max_length(const StringVector &v) {
      uint n = v.size();
      uint sz = 0;
      for (uint i = 0; i < n; ++i) {
        sz = std::max<uint>(sz, v[i].size());
      }
      return sz;
    }

    //------------------------------------------------------------
    std::ostream &IrtModel::print_subscales(
        std::ostream &out, bool nl, bool decorate) {
      std::string sep = "   ";
      if (decorate) {
        uint sz = find_max_length(subscale_names());
        out << std::string(2, '-') << sep << std::string(sz, '-') << endl;
      }

      for (uint i = 0; i < nscales(); ++i) {
        if (decorate) out << std::setw(2) << i << sep;
        out << subscale_names()[i];
        if (nl) {
          out << endl;
        } else {
          out << " ";
        }
      }
      return out;
    }

    //------------------------------------------------------------
    uint IrtModel::nscales() const { return subscale_names_.size(); }
    uint IrtModel::nsubjects() const { return subjects_.size(); }
    uint IrtModel::nitems() const { return items.size(); }

    //------------------------------------------------------------
    void IrtModel::add_item(const Ptr<Item> &item) {
      items.insert(item);
      ParamPolicy::add_model(item);
    }

    //------------------------------------------------------------
    ItemIt IrtModel::item_begin() { return items.begin(); }
    ItemIt IrtModel::item_end() { return items.end(); }
    ItemItC IrtModel::item_begin() const { return items.begin(); }
    ItemItC IrtModel::item_end() const { return items.end(); }

    //------------------------------------------------------------
    Ptr<Item> IrtModel::find_item(const std::string &id, bool nag) const {
      item_search_helper->id_ = id;
      ItemItC it = items.lower_bound(item_search_helper);
      if (it == items.end() || (*it)->id() != id) {
        if (nag) {
          std::ostringstream msg;
          msg << "item with id " << id << " not found in IrtModel::find_item";
          report_error(msg.str());
        }
        return Ptr<Item>();
      }
      return *it;
    }

    //------------------------------------------------------------
    void IrtModel::add_subject(const Ptr<Subject> &s) {
      BOOM::IRT::add_subject(subjects_, s);
      DataPolicy::add_data(s);
      if (!!subject_prior_) subject_prior_->add_data(s);
    }
    //------------------------------------------------------------
    SI IrtModel::subject_begin() { return subjects_.begin(); }
    SI IrtModel::subject_end() { return subjects_.end(); }
    CSI IrtModel::subject_begin() const { return subjects_.begin(); }
    CSI IrtModel::subject_end() const { return subjects_.end(); }

    //------------------------------------------------------------
    Ptr<Subject> IrtModel::find_subject(const std::string &id, bool nag) const {
      subject_search_helper->id_ = id;
      CSI it = std::lower_bound(subject_begin(), subject_end(),
                                subject_search_helper, SubjectLess());
      if (it == subject_end() || (*it)->id() != id) {
        if (nag) {
          std::ostringstream msg;
          msg << "subject with id " << id
              << " not found in IrtModel::find_subject";
          report_error(msg.str());
        }
        return Ptr<Subject>();
      }
      return *it;
    }

    //------------------------------------------------------------
    void IrtModel::set_subject_prior(const Ptr<MvnModel> &p) {
      subject_prior_ = new MvnSubjectPrior(p);
      allocate_subjects();
    }

    void IrtModel::set_subject_prior(const Ptr<SubjectPrior> &sp) {
      subject_prior_ = sp;
      allocate_subjects();
    }

    IrtModel::PriPtr IrtModel::subject_prior() { return subject_prior_; }

    void IrtModel::allocate_subjects() {
      if (!subject_prior_) return;
      for (SI s = subject_begin(); s != subject_end(); ++s) {
        subject_prior_->add_data(*s);
      }
    }

    //------------------------------------------------------------
    void read_subject_info_file(const std::string &fname,
                                const Ptr<IrtModel> &m,
                                const char delim) {
      std::ifstream in(fname.c_str());
      while (in) {
        std::string line;
        getline(in, line);
        if (!in || is_all_white(line)) break;
        StringVector fields =
            (delim == ' ') ? split_string(line) : split_delimited(line, delim);

        uint nf = fields.size();
        std::string id = fields[0];
        if (!!m->find_subject(id, false)) {
          std::ostringstream msg;
          msg << "IrtModel::read_subject_info_file..." << endl
              << "subject identifiers must be unique" << endl
              << "offending id: " << id;
          report_error(msg.str().c_str());
        }

        if (nf == 1) {
          NEW(Subject, s)(id, m->nscales());
          m->add_subject(s);
        } else if (nf > 1) {
          Vector x(nf - 1);
          for (uint i = 1; i < nf; ++i) {
            std::istringstream(fields[i]) >> x[i - 1];
          }
          NEW(Subject, s)(id, m->nscales(), x);
          m->add_subject(s);
        } else {
          std::ostringstream out;
          out << "0 fields in IrtModel::read_subject_info_file";
          report_error(out.str().c_str());
        }
      }
    }

    void read_item_response_file(const std::string &fname,
                                 const Ptr<IrtModel> &m) {
      std::ifstream in(fname.c_str());
      while (in) {
        std::string line;
        getline(in, line);
        if (!in || is_all_white(line)) break;

        std::string subject_id;
        std::string item_id;
        std::string response_str;
        std::istringstream sin(line);
        sin >> subject_id >> item_id >> response_str;
        Ptr<Subject> sub = m->find_subject(subject_id, false);
        if (!sub) {
          sub = new Subject(subject_id, m->nscales());
          m->add_subject(sub);
        }

        const Ptr<Item> &item = m->find_item(item_id, false);
        if (!item) {
          std::ostringstream msg;
          msg << "item " << item_id
              << " present in IrtModel::read_item_response_file," << endl
              << "but not in IrtModel::read_item_info_file." << endl;
          report_error(msg.str().c_str());
        }

        Response r = item->make_response(response_str);
        item->add_subject(sub);
        sub->add_item(item, r);  // response levels are shared here
      }
    }

    void IrtModel::item_report(std::ostream &out, uint max_name_width) const {
      uint maxw = 0;
      for (ItemItC it = items.begin(); it != items.end(); ++it) {
        maxw = std::max<uint>(maxw, (*it)->name().size());
      }
      maxw = std::min(maxw, max_name_width);
      for (ItemItC it = items.begin(); it != items.end(); ++it) {
        (*it)->report(out, maxw);
      }
    }
  }  // namespace IRT
}  // namespace BOOM
