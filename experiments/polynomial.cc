#include "cpputil/Polynomial.hpp"
#include "LinAlg/Vector.hpp"
#include <iostream>

using namespace BOOM;

int main() {
  Vector coef = {1, .3, -.2};
  Polynomial poly(coef);
  std::cout << "poly = " << poly << std::endl;

  Vector roots = poly.real_roots();
  std::cout << "roots = " << roots << std::endl;

  std::cout << "poly(" << roots[0] << ") = "
            << poly(roots[0]) << std::endl
            << "poly(" << roots[1] << ") = "
            << poly(roots[1]) << std::endl;
}

