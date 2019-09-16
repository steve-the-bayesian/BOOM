# Vitali's script
#c++ -O3 -Wall -shared -std=c++11 -fPIC -undefined dynamic_lookup `python3 -m pybind11 --includes` -I../../ pybsts.cpp -L../.. -lboom -o pybsts`python3-config --extension-suffix`

# Mine
#c++ -O3 -Wall -shared -std=c++11 -fPIC -undefined dynamic_lookup -I'/usr/local/include/BOOM' `python3 -m pybind11 --includes` -lboom Models/GaussianModel.cpp -oBayesBoom`python3-config --extension-suffix`

c++ -shared -O3 -Wall  -std=c++11 -fPIC -undefined dynamic_lookup -I'/usr/local/include/BOOM' `python3 -m pybind11 --includes` -lboom module.cpp -o module.o

c++ -shared -O3 -Wall  -std=c++11 -fPIC -undefined dynamic_lookup -I'/usr/local/include/BOOM' `python3 -m pybind11 --includes` -lboom LinAlg/LinAlgWrapper.cpp -o LinAlgWrapper.o

c++ -shared -O3 -Wall  -std=c++11 -fPIC -undefined dynamic_lookup -I'/usr/local/include/BOOM' `python3 -m pybind11 --includes` -lboom stats/BsplineWrapper.cpp -o BsplineWrapper.o

c++ -shared -O3 -Wall  -std=c++11 -fPIC -undefined dynamic_lookup -I'/usr/local/include/BOOM' `python3 -m pybind11 --includes` -lboom Models/ParameterWrapper.cpp -o ParameterWrapper.o

c++ -shared -O3 -Wall  -std=c++11 -fPIC -undefined dynamic_lookup -I'/usr/local/include/BOOM' `python3 -m pybind11 --includes` -lboom Models/GaussianModelWrapper.cpp -o GaussianModelWrapper.o

c++ -shared -O3 -Wall  -std=c++11 -fPIC -undefined dynamic_lookup -I'/usr/local/include/BOOM' `python3 -m pybind11 --includes` -lboom Models/GammaModelWrapper.cpp -o GammaModelWrapper.o

c++ -Wall -shared -std=c++11 -fPIC -undefined dynamic_lookup module.o GaussianModelWrapper.o GammaModelWrapper.o LinAlgWrapper.o BsplineWrapper.o -oBayesBoom`python3-config --extension-suffix`

