#include "PyList.hpp"
#include "PyTuple.hpp"
#include "PyType.hpp"
#include "types/builtin.hpp"
#include "vm/VM.hpp"

#include "gtest/gtest.h"

using namespace py;

TEST(PyType, ObjectClassParent)
{
	[[maybe_unused]] auto scope = VirtualMachine::the().heap().scoped_gc_pause();
	const auto &bases = types::object()->underlying_type().__bases__;
	EXPECT_TRUE(bases.empty());

	auto mro_ = types::object()->mro();
	ASSERT_TRUE(mro_.is_ok());
	auto *mro = mro_.unwrap();
	ASSERT_TRUE(mro);
	EXPECT_EQ(mro->elements().size(), 1);

	EXPECT_TRUE(std::holds_alternative<PyObject *>(mro->elements()[0]));
	auto *mro_0 = std::get<PyObject *>(mro->elements()[0]);
	EXPECT_EQ(mro_0, types::object());
}

namespace {
TypePrototype *new_type(const std::string &name, std::vector<PyType *> bases)
{
	auto *new_type = new TypePrototype{};
	new_type->__name__ = name;
	new_type->__bases__ = std::move(bases);
	return new_type;
}
}// namespace

TEST(PyType, InheritanceTriangle)
{
	[[maybe_unused]] auto scope = VirtualMachine::the().heap().scoped_gc_pause();
	PyType *B1 =
		PyType::initialize(std::unique_ptr<TypePrototype>(new_type("B1", { types::object() })));
	PyType *B2 =
		PyType::initialize(std::unique_ptr<TypePrototype>(new_type("B2", { types::object() })));

	PyType *C = PyType::initialize(std::unique_ptr<TypePrototype>(new_type("C", { B1, B2 })));

	auto C_mro_ = C->mro();
	ASSERT_TRUE(C_mro_.is_ok());
	auto *C_mro = C_mro_.unwrap();
	ASSERT_TRUE(C_mro);
	EXPECT_EQ(C_mro->elements().size(), 4);
	EXPECT_EQ(std::get<PyObject *>(C_mro->elements()[0]), C);
	EXPECT_EQ(std::get<PyObject *>(C_mro->elements()[1]), B1);
	EXPECT_EQ(std::get<PyObject *>(C_mro->elements()[2]), B2);
	EXPECT_EQ(std::get<PyObject *>(C_mro->elements()[3]), types::object());
}

TEST(PyType, InheritanceDiamond)
{
	[[maybe_unused]] auto scope = VirtualMachine::the().heap().scoped_gc_pause();
	PyType *A =
		PyType::initialize(std::unique_ptr<TypePrototype>(new_type("A", { types::object() })));
	PyType *B1 = PyType::initialize(std::unique_ptr<TypePrototype>(new_type("B1", { A })));
	PyType *B2 = PyType::initialize(std::unique_ptr<TypePrototype>(new_type("B2", { A })));
	PyType *C = PyType::initialize(std::unique_ptr<TypePrototype>(new_type("C", { B1, B2 })));

	auto C_mro_ = C->mro();
	ASSERT_TRUE(C_mro_.is_ok());
	auto *C_mro = C_mro_.unwrap();
	ASSERT_TRUE(C_mro);
	EXPECT_EQ(C_mro->elements().size(), 5);
	EXPECT_EQ(std::get<PyObject *>(C_mro->elements()[0]), C);
	EXPECT_EQ(std::get<PyObject *>(C_mro->elements()[1]), B1);
	EXPECT_EQ(std::get<PyObject *>(C_mro->elements()[2]), B2);
	EXPECT_EQ(std::get<PyObject *>(C_mro->elements()[3]), A);
	EXPECT_EQ(std::get<PyObject *>(C_mro->elements()[4]), types::object());
}
