#include "PyString.hpp"

#include <gtest/gtest.h>

using namespace py;

TEST(PyString, hash)
{
	String foo{ "foo" };
	auto foo_obj_ = PyString::create(foo.s);
	ASSERT_TRUE(foo_obj_.is_ok());
	auto object_hash = foo_obj_.unwrap()->hash();
	ASSERT_TRUE(object_hash.is_ok());
	ASSERT_EQ(ValueHash{}(foo), object_hash.unwrap());
}