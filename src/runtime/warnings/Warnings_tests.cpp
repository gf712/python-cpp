#include "gtest/gtest.h"

import py.runtime;
import py.types;

using namespace py;

// Each warning header declares make_<name>_warning out of line and calls it from an inline
// <name>_warning wrapper. Referencing the wrappers here is what pins the out-of-line
// definitions down: three of them were declared but never defined, which nothing caught
// because no call site existed.

namespace {
void assert_warning(BaseException *exc, PyType *expected_type, std::string_view message)
{
	ASSERT_TRUE(exc);
	EXPECT_EQ(exc->type(), expected_type);
	auto str = exc->str();
	ASSERT_TRUE(str.is_ok());
	EXPECT_EQ(str.unwrap()->value(), message);
}
}// namespace

TEST(Warnings, DeprecationWarning)
{
	[[maybe_unused]] auto scope = VirtualMachine::the().heap().scoped_gc_pause();
	assert_warning(deprecation_warning("spam is deprecated"),
		types::deprecation_warning(),
		"spam is deprecated");
}

TEST(Warnings, PendingDeprecationWarning)
{
	[[maybe_unused]] auto scope = VirtualMachine::the().heap().scoped_gc_pause();
	assert_warning(pending_deprecation_warning("spam will be deprecated"),
		types::pending_deprecation_warning(),
		"spam will be deprecated");
}

TEST(Warnings, ResourceWarning)
{
	[[maybe_unused]] auto scope = VirtualMachine::the().heap().scoped_gc_pause();
	assert_warning(resource_warning("unclosed file"), types::resource_warning(), "unclosed file");
}

TEST(Warnings, ImportWarning)
{
	[[maybe_unused]] auto scope = VirtualMachine::the().heap().scoped_gc_pause();
	assert_warning(import_warning("bad import"), types::import_warning(), "bad import");
}

TEST(Warnings, FormatsArguments)
{
	[[maybe_unused]] auto scope = VirtualMachine::the().heap().scoped_gc_pause();
	assert_warning(deprecation_warning("{} is deprecated, use {}", "spam", "eggs"),
		types::deprecation_warning(),
		"spam is deprecated, use eggs");
}
