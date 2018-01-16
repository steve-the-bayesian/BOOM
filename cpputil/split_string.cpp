#include <string>
#include <vector>
#include <cassert>

namespace BOOM{
  using namespace std;
  using std::string;
  vector<string> split_string(const string &s){
    typedef std::string::size_type sz;
    vector<string> ans;
    const string ws(" \n\r\t\f\v");

    sz b = s.find_first_not_of(ws);
    if(b==string::npos) return ans;

    while(1){
      sz e = s.find_first_of(ws,b);
      assert(e>=b);
      if(e==string::npos){
        ans.push_back(s.substr(b));
        return ans;
      }else{
        ans.push_back(s.substr(b,e-b));
        b = s.find_first_not_of(ws, e);
        if(b==string::npos) return ans;}}}
}
