// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2017 Steven L. Scott

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

#include "cpputil/AsciiGraph.hpp"
#include <cmath>
#include "cpputil/report_error.hpp"

namespace BOOM {

  AsciiGraph::AsciiGraph()
      : xlo_(0),
        xhi_(0),
        ylo_(0),
        yhi_(0),
        xbuckets_(-1),
        ybuckets_(-1),
        throw_on_error_(true) {}

  AsciiGraph::AsciiGraph(double xlo, double xhi, double ylo, double yhi,
                         int xbuckets, int ybuckets, bool throw_on_error)
      : xlo_(xlo),
        xhi_(xhi),
        ylo_(ylo),
        yhi_(yhi),
        xbuckets_(xbuckets),
        ybuckets_(ybuckets),
        throw_on_error_(throw_on_error) {
    if (xlo_ >= xhi_) {
      report_error("Illegal X limits.");
    }
    if (ylo_ >= yhi_) {
      report_error("Illegal Y limits.");
    }
    if (xbuckets_ <= 1 || ybuckets_ <= 1) {
      report_error("Need more pixels to draw a graph.");
    }
    clear();
  }

  void AsciiGraph::plot(double x, double y, char plotting_character) {
    int xbucket = which_bucket(x, xlo_, xhi_, xbuckets_);
    int ybucket = which_bucket(y, ylo_, yhi_, ybuckets_);
    if (xbucket >= 0 && ybucket >= 0) {
      graph_[ybucket][xbucket] = plotting_character;
    }
  }

  void AsciiGraph::plot_horizontal_line(double value, char plotting_character) {
    int bucket = which_bucket(value, ylo_, yhi_, ybuckets_);
    if (bucket >= 0) {
      graph_[bucket] = std::string(xbuckets_, plotting_character);
    }
  }

  void AsciiGraph::plot_vertical_line(double value, char plotting_character) {
    int bucket = which_bucket(value, xlo_, xhi_, xbuckets_);
    if (bucket >= 0) {
      for (int i = 0; i < ybuckets_; ++i) {
        graph_[i][bucket] = plotting_character;
      }
    }
  }

  std::string AsciiGraph::print() const {
    // Make sure graph starts on a fresh line.
    std::string ans = "\n";
    for (auto it = graph_.rbegin(); it != graph_.rend(); ++it) {
      ans += *it + '\n';
    }
    return ans;
  }

  void AsciiGraph::clear() {
    graph_.assign(ybuckets_, std::string(xbuckets_, ' '));
  }

  int AsciiGraph::which_bucket(double value, double lo, double hi,
                               int buckets) const {
    if (buckets <= 0) {
      report_error(
          "Can't plot to a zero-sized graph.  "
          "Try increasing the number of pixels.");
    }
    int ans = std::lround(
        std::floor(buckets * ((value - lo) / (1.00001 * (hi - lo)))));
    if (ans < 0 || ans >= buckets) {
      if (throw_on_error_) {
        std::ostringstream err;
        err << "Illegal value " << value << " outside the legal range: [" << lo
            << ", " << hi << "].";
        report_error(err.str());
      }
      return -1;
    }
    return ans;
  }

}  // namespace BOOM
