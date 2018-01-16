#include <fstream>
#include <string>
#include <vector>

namespace BOOM{
  using namespace std;
  using std::string;
  vector<string> read_file(istream &in){
    vector<string> ans;
    while(in){
      string line;
      getline(in, line);
      if(!in) break;
      ans.push_back(line);
    }
    return ans;
  }


  vector<string> read_file(const string &fname){
    ifstream in(fname.c_str());
    return read_file(in);
  }
}
