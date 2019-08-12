#include <pybind11/pybind11.h>
#include <string>

namespace py = pybind11;

int add(int x = 1, int y = 2) {
  return x + y;
}

class Pet {
 public:
  Pet(const std::string &name) : name_(name) {}
  void set_name(const std::string &name) {name_ = name;}
  const std::string &name() const {return name_;}

  std::string bark() const {return "woof!";}
  
 private:
  std::string name_;
};


PYBIND11_MODULE(example, m) {
  m.doc() = "A docstring for the example module";

  m.attr("jenny") = 867.5309;

  m.attr("author") = std::string("Steve");
  
  m.def("add",
        &add,
        "Add two integers.",
        py::arg("first_term") = 1,
        py::arg("second_term") = 2,
        "Docstring for the function");

  
  py::class_<Pet>(m, "Pet")
      .def(py::init<const std::string &>())
      .def_property("name", &Pet::name, &Pet::set_name)
      .def("bark", &Pet::bark)
      .def("__repr__",
           [](const Pet &pet) {
             return "<A pet named '" + pet.name() + "'>";
           });

}
