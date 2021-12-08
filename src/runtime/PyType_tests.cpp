#include "PyList.hpp"
#include "PyTuple.hpp"
#include "PyType.hpp"
#include "types/builtin.hpp"

#include "gtest/gtest.h"


TEST(PyType, ObjectClassParent)
{
	PyType *object = custom_object();
	auto *bases = object->underlying_type().__bases__;
	EXPECT_TRUE(bases->elements().empty());

	auto *mro = object->mro();
	EXPECT_EQ(mro->elements().size(), 1);

	EXPECT_TRUE(std::holds_alternative<PyObject *>(mro->elements()[0]));
	auto *mro_0 = std::get<PyObject *>(mro->elements()[0]);
	EXPECT_EQ(mro_0, custom_object());
}

TEST(PyType, InheritanceTriangle)
{
	PyType *B1 = PyType::initialize(
		TypePrototype{ .__name__ = "B1", .__bases__ = PyTuple::create(custom_object()) });
	PyType *B2 = PyType::initialize(
		TypePrototype{ .__name__ = "B2", .__bases__ = PyTuple::create(custom_object()) });

	PyType *C =
		PyType::initialize(TypePrototype{ .__name__ = "C", .__bases__ = PyTuple::create(B1, B2) });

	auto C_mro = C->mro();
	EXPECT_EQ(C_mro->elements().size(), 4);
	EXPECT_EQ(std::get<PyObject *>(C_mro->elements()[0]), C);
	EXPECT_EQ(std::get<PyObject *>(C_mro->elements()[1]), B1);
	EXPECT_EQ(std::get<PyObject *>(C_mro->elements()[2]), B2);
	EXPECT_EQ(std::get<PyObject *>(C_mro->elements()[3]), custom_object());
}

TEST(PyType, InheritanceDiamond)
{
	PyType *A = PyType::initialize(
		TypePrototype{ .__name__ = "A", .__bases__ = PyTuple::create(custom_object()) });
	PyType *B1 =
		PyType::initialize(TypePrototype{ .__name__ = "B1", .__bases__ = PyTuple::create(A) });
	PyType *B2 =
		PyType::initialize(TypePrototype{ .__name__ = "B2", .__bases__ = PyTuple::create(A) });
	PyType *C =
		PyType::initialize(TypePrototype{ .__name__ = "C", .__bases__ = PyTuple::create(B1, B2) });

	auto C_mro = C->mro();
	EXPECT_EQ(C_mro->elements().size(), 5);
	EXPECT_EQ(std::get<PyObject *>(C_mro->elements()[0]), C);
	EXPECT_EQ(std::get<PyObject *>(C_mro->elements()[1]), B1);
	EXPECT_EQ(std::get<PyObject *>(C_mro->elements()[2]), B2);
	EXPECT_EQ(std::get<PyObject *>(C_mro->elements()[3]), A);
	EXPECT_EQ(std::get<PyObject *>(C_mro->elements()[4]), custom_object());
}