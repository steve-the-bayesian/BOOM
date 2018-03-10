// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2014 Steven L. Scott

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

#ifndef BOOM_PARAM_FILE_IO_MANAGER_HPP_
#define BOOM_PARAM_FILE_IO_MANAGER_HPP_

#include <deque>
#include <memory>
#include <string>
#include "Models/ParamTypes.hpp"

namespace BOOM {

  namespace ParameterFileIO {

    // A SingleParameterIoManager manages file I/O for a single
    // parameter.  This class buffers reading from and writing to
    // files.
    class SingleParameterIoManager {
     public:
      // Args:
      //   parameter:  The parameter to be managed.
      //   filename:  The name of the file used to store parameter values.
      //   buffer_size_in_iterations:  The size of the I/O buffer to use.
      SingleParameterIoManager(const Ptr<Params> &parameter,
                               const std::string &filename,
                               int buffer_size_in_iterations);

      // The destructor will flush any remaining output in the buffer.
      ~SingleParameterIoManager();

      // Sets the size of the I/O buffer to the given number of
      // iterations.
      void set_bufsize(int iterations);

      // Clears the data file (the file exists, but is empty).
      void clear_file();

      // Write any remaining data in the buffer to the data file.
      void flush();

      // Add the current value of the parameter to the buffer.
      void write();

      // Reset the pointer to the beginning of the data file.
      void rewind();

      // Reads the next parameter value in the data file.  If no data
      // has been read, then rewind() is called first, so that reading
      // starts from the beginning of the file.
      void read_next_value();

      // Reads the last line of the data file.  Upon completion, the
      // file may be used for writing (with future output appended at
      // the end).
      void read_last_line();

     private:
      // Read the next batch of data from the data file into the
      // buffer_.
      void read_to_fill_buffer();

      Ptr<Params> parameter_;

      // The name of the file where parameter values are stored.
      std::string filename_;

      // The buffer holding the output iterations or the input stream.
      // Its size must be buffer_size_in_iterations_ *
      // parameter_->size(false);
      //
      // TODO: Consider making this a deque<Vector>.
      std::deque<double> buffer_;
      int buffer_limit_;

      ifstream input_;
      ofstream output_;
    };

  }  // namespace ParameterFileIO

  // The ParamFileIoManager manages a collection of Params objects
  // (each internally assigned its own SingleParameterIoManager).
  class ParamFileIoManager {
   public:
    ParamFileIoManager();

    // Adds a parameter to the set of parameters being managed.
    void add_parameter(const Ptr<Params> &parameter,
                       const std::string &filename);

    // Sets the size of the input/output buffer to the specified
    // number of iterations.
    void set_bufsize(int iterations);

    // Removes any data from the parameter files.
    void clear_files();

    // Any data currently in the buffer is written to the file, and
    // the buffer is made empty.
    void flush();

    // Appends parameter values to the end of the buffer.  If the
    // buffer is full then it will be flushed.
    void write();

    // Resets the read position to the start of the data files.
    void rewind();

    // Reads the next entry from the history file.  If no reading has
    // been done thus far, then rewind() is called so that reading
    // starts from the beginning of the file
    void read_next_value();

    // Reads the last line of the data file.  Upon completion, the
    // file may be used for writing (with future output appended at
    // the end).
    void read_last_line();

   private:
    // There is one element in io_ for each parameter added to the
    // IoManager.
    std::vector<std::shared_ptr<ParameterFileIO::SingleParameterIoManager> >
        io_;
    int buffer_size_in_iterations_;
  };

}  // namespace BOOM

#endif  // BOOM_PARAM_FILE_IO_MANAGER_HPP_
