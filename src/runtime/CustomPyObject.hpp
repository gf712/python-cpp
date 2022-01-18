#pragma once

#include "PyObject.hpp"

class CustomPyObject : public PyBaseObject
{
	const PyType *m_type_obj;

  public:
	CustomPyObject(const PyType *type);
	
	std::string to_string() const override;
	
	PyType *type() const override;
};