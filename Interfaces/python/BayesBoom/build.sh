# Vitali's script
#c++ -O3 -Wall -shared -std=c++11 -fPIC -undefined dynamic_lookup `python3 -m pybind11 --includes` -I../../ pybsts.cpp -L../.. -lboom -o pybsts`python3-config --extension-suffix`

# Mine
#c++ -O3 -Wall -shared -std=c++11 -fPIC -undefined dynamic_lookup -I'/usr/local/include/BOOM' `python3 -m pybind11 --includes` -lboom Models/GaussianModel.cpp -oBayesBoom`python3-config --extension-suffix`

c++ -shared -O3 -Wall  -std=c++11 -fPIC -undefined dynamic_lookup -I'/usr/local/include/BOOM' `python3 -m pybind11 --includes` -lboom module.cpp -o module.o

c++ -shared -O3 -Wall  -std=c++11 -fPIC -undefined dynamic_lookup -I'/usr/local/include/BOOM' `python3 -m pybind11 --includes` -lboom LinAlg/LinAlgWrapper.cpp -o LinAlgWrapper.o

c++ -shared -O3 -Wall  -std=c++11 -fPIC -undefined dynamic_lookup -I'/usr/local/include/BOOM' `python3 -m pybind11 --includes` -lboom Models/GaussianModel.cpp -o GaussianModel.o

c++ -v -Wall -shared -std=c++11 -fPIC -undefined dynamic_lookup module.o GaussianModel.o LinAlgWrapper.o -oBayesBoom`python3-config --extension-suffix`

