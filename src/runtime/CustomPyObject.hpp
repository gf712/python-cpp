#pragma once

#include "PyObject.hpp"

namespace py {
class CustomPyObject : public PyBaseObject
{
  public:
	CustomPyObject(const PyType *type);
	
	std::string to_string() const override;
};
}// namespace py