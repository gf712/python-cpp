#pragma once

#include "PyObject.hpp"

struct CustomPyObjectContext
{
	const std::string name;
	const PyDict *attributes;
};

// a user defined PyObject
class CustomPyObject : public PyBaseObject
{
  public:
	CustomPyObject(const CustomPyObjectContext &ctx, const PyTuple *);

	std::string to_string() const override { return fmt::format("CustomPyObject"); }

	static std::unique_ptr<TypePrototype> register_type();
	PyType *type_() const override;

  private:
	bool update_slot_if_special(const std::string &name, PyObject *value);
};