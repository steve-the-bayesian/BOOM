#include <string>
#include <vector>

namespace BOOM{

  using namespace std;
  using std::string;
//   vector<string> split_delimited(const string &s, char delim){
//     string d(1, delim);
//     return split_delimited(s, d);
//   }

  vector<string> split_delimited(const string &s, const string &delims){
    vector<string> ans;
    typedef std::string::size_type sz;
    sz b=0;
    bool done=false;

    while(!done){
      sz e = s.find_first_of(delims, b);
      if(e==std::string::npos){
        done=true;
        ans.push_back(s.substr(b));
      }else{
        ans.push_back(s.substr(b,e-b));
      }
      b=e+1;
    }
    return ans;
  }
}
