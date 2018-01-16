#ifndef BOOM_PROGRESS_TRACKER_CLASS_HPP
#define BOOM_PROGRESS_TRACKER_CLASS_HPP

#include <BOOM.hpp>
#include <cpputil/Ptr.hpp>
#include <cpputil/RefCounted.hpp>

namespace BOOM{
  class ProgressTracker:  private RefCounted{
    string fname;
    ostream * msg_;
    uint nskip;
    uint n;
    string sep;
    bool owns_msg;
    void start(const string & prog_name);
    ProgressTracker(const ProgressTracker &) : RefCounted(){}
  public:
    // Write progress messages to a file named "msg" in directory dname.
    ProgressTracker(const string &dname, uint nskip=100,
                    bool restart=false, const string & prog_name="",
                    bool keep_existing_msg=false);

    // Write progress messages to std::cout
    ProgressTracker(uint nskip=100, const string & prog_name="");

    // Write progress to an arbitrary stream
    ProgressTracker(ostream &out, uint nskip=100, const string & prog_name="");
    ~ProgressTracker() override;
    ProgressTracker & operator++(){update(); return *this;}
    ProgressTracker & operator++(int){update(); return *this;}
    void update();
    uint restart();
    void set_niter(uint n);

    ostream & msg();

    friend void intrusive_ptr_add_ref(ProgressTracker *m);
    friend void intrusive_ptr_release(ProgressTracker *m);
  };
  void intrusive_ptr_add_ref(ProgressTracker *m);
  void intrusive_ptr_release(ProgressTracker *m);
}
#endif// BOOM_PROGRESS_TRACKER_CLASS_HPP
