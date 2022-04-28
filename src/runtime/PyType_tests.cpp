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
	auto *bases = object()->underlying_type().__bases__;
	EXPECT_TRUE(bases->elements().empty());

	auto mro_ = object()->mro();
	ASSERT_TRUE(mro_.is_ok());
	auto *mro = mro_.unwrap_as<PyList>();
	ASSERT_TRUE(mro);
	EXPECT_EQ(mro->elements().size(), 1);

	EXPECT_TRUE(std::holds_alternative<PyObject *>(mro->elements()[0]));
	auto *mro_0 = std::get<PyObject *>(mro->elements()[0]);
	EXPECT_EQ(mro_0, object());
}

TEST(PyType, InheritanceTriangle)
{
	[[maybe_unused]] auto scope = VirtualMachine::the().heap().scoped_gc_pause();
	PyType *B1 = PyType::initialize(TypePrototype{
		.__name__ = "B1", .__bases__ = PyTuple::create(object()).unwrap_as<PyTuple>() });
	PyType *B2 = PyType::initialize(TypePrototype{
		.__name__ = "B2", .__bases__ = PyTuple::create(object()).unwrap_as<PyTuple>() });

	PyType *C = PyType::initialize(TypePrototype{
		.__name__ = "C", .__bases__ = PyTuple::create(B1, B2).unwrap_as<PyTuple>() });

	auto C_mro_ = C->mro();
	ASSERT_TRUE(C_mro_.is_ok());
	auto *C_mro = C_mro_.unwrap_as<PyList>();
	ASSERT_TRUE(C_mro);
	EXPECT_EQ(C_mro->elements().size(), 4);
	EXPECT_EQ(std::get<PyObject *>(C_mro->elements()[0]), C);
	EXPECT_EQ(std::get<PyObject *>(C_mro->elements()[1]), B1);
	EXPECT_EQ(std::get<PyObject *>(C_mro->elements()[2]), B2);
	EXPECT_EQ(std::get<PyObject *>(C_mro->elements()[3]), object());
}

TEST(PyType, InheritanceDiamond)
{
	[[maybe_unused]] auto scope = VirtualMachine::the().heap().scoped_gc_pause();
	PyType *A = PyType::initialize(TypePrototype{
		.__name__ = "A", .__bases__ = PyTuple::create(object()).unwrap_as<PyTuple>() });
	PyType *B1 = PyType::initialize(
		TypePrototype{ .__name__ = "B1", .__bases__ = PyTuple::create(A).unwrap_as<PyTuple>() });
	PyType *B2 = PyType::initialize(
		TypePrototype{ .__name__ = "B2", .__bases__ = PyTuple::create(A).unwrap_as<PyTuple>() });
	PyType *C = PyType::initialize(TypePrototype{
		.__name__ = "C", .__bases__ = PyTuple::create(B1, B2).unwrap_as<PyTuple>() });

	auto C_mro_ = C->mro();
	ASSERT_TRUE(C_mro_.is_ok());
	auto *C_mro = C_mro_.unwrap_as<PyList>();
	ASSERT_TRUE(C_mro);
	EXPECT_EQ(C_mro->elements().size(), 5);
	EXPECT_EQ(std::get<PyObject *>(C_mro->elements()[0]), C);
	EXPECT_EQ(std::get<PyObject *>(C_mro->elements()[1]), B1);
	EXPECT_EQ(std::get<PyObject *>(C_mro->elements()[2]), B2);
	EXPECT_EQ(std::get<PyObject *>(C_mro->elements()[3]), A);
	EXPECT_EQ(std::get<PyObject *>(C_mro->elements()[4]), object());
}