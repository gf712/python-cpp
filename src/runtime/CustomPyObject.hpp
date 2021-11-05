#pragma once

#include "PyObject.hpp"

struct CustomPyObjectContext
{
	const std::string name;
	const PyDict *attributes;
};

// a user defined PyObject
class CustomPyObject : public PyObject
{
  public:
	CustomPyObject(const CustomPyObjectContext &ctx, const PyTuple *);

	std::string to_string() const override { return fmt::format("CustomPyObject"); }

  private:
	bool update_slot_if_special(const std::string &name, PyObject *value);
};