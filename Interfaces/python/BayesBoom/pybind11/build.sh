# Vitali's script
c++ -O3 -Wall -shared -std=c++11 -fPIC -undefined dynamic_lookup `python3 -m pybind11 --includes` -I../../ pybsts.cpp -L../.. -lboom -o pybsts`python3-config --extension-suffix`

# Mine
c++ -O3 -Wall -shared -std=c++11 -fPIC -undefined dynamic_lookup -I'/usr/local/include/BOOM' `python3 -m pybind11 --includes` -lboom Models/GaussianModel.cpp -o BayesBoom`python3-config --extension-suffix`
