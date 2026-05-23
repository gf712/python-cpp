#include "SourceManager.hpp"

#include "gtest/gtest.h"

#include <cstdio>
#include <filesystem>
#include <fstream>

namespace {
std::filesystem::path write_temp(std::string_view name, std::string_view contents)
{
	auto path = std::filesystem::temp_directory_path() / name;
	std::ofstream{ path } << contents;
	return path;
}
}// namespace

TEST(SourceManager, ReturnsRequestedLineOneIndexed)
{
	const auto path = write_temp("source_manager_basic.py",
		"alpha\n"
		"beta\n"
		"gamma\n");
	auto &sm = py::SourceManager::the();
	EXPECT_EQ(sm.line(path.string(), 1), "alpha");
	EXPECT_EQ(sm.line(path.string(), 2), "beta");
	EXPECT_EQ(sm.line(path.string(), 3), "gamma");
}

TEST(SourceManager, ReturnsEmptyForZeroOrOutOfRangeLine)
{
	const auto path = write_temp("source_manager_oob.py", "only\n");
	auto &sm = py::SourceManager::the();
	EXPECT_TRUE(sm.line(path.string(), 0).empty());
	EXPECT_TRUE(sm.line(path.string(), 2).empty());
	EXPECT_TRUE(sm.line(path.string(), 999).empty());
}

TEST(SourceManager, ReturnsEmptyForUnreadableFile)
{
	auto &sm = py::SourceManager::the();
	EXPECT_TRUE(sm.line("/nonexistent/path/source_manager_missing.py", 1).empty());
}

TEST(SourceManager, StripLeadingWhitespaceRemovesSpacesAndTabs)
{
	using py::SourceManager;
	EXPECT_EQ(SourceManager::strip_leading_whitespace("    x = 1"), "x = 1");
	EXPECT_EQ(SourceManager::strip_leading_whitespace("\t\tcall()"), "call()");
	EXPECT_EQ(SourceManager::strip_leading_whitespace("no_indent"), "no_indent");
	EXPECT_TRUE(SourceManager::strip_leading_whitespace("    \t").empty());
	EXPECT_TRUE(SourceManager::strip_leading_whitespace("").empty());
}
