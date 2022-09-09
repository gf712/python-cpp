#include "VariablesResolver.hpp"

#include "lexer/Lexer.hpp"
#include "parser/Parser.hpp"

#include "gtest/gtest.h"


namespace {
VariablesResolver::VisibilityMap generate_resolver(std::string_view program)
{
	auto lexer = Lexer::create(std::string(program), "_bytecode_generator_tests_.py");
	parser::Parser p{ lexer };
	p.parse();

	auto module = as<ast::Module>(p.module());
	ASSERT(module)

	return VariablesResolver::resolve(module.get());
}
}// namespace

TEST(VariablesResolver, GlobalNamespace)
{
	static constexpr std::string_view program =
		"a = foo()\n"
		"b = z.bar()\n"
		"c = a + b\n"
		"print(c)\n";

	auto visibility = generate_resolver(program);
	ASSERT_EQ(visibility.size(), 1);

	ASSERT_TRUE(visibility.contains("_bytecode_generator_tests_"));
	auto &main = visibility.at("_bytecode_generator_tests_");
	ASSERT_EQ(main->visibility.size(), 6);
	ASSERT_TRUE(main->visibility.contains("a"));
	ASSERT_EQ(main->visibility.at("a"), VariablesResolver::Visibility::NAME);
	ASSERT_TRUE(main->visibility.contains("foo"));
	ASSERT_EQ(main->visibility.at("foo"), VariablesResolver::Visibility::NAME);
	ASSERT_TRUE(main->visibility.contains("b"));
	ASSERT_EQ(main->visibility.at("b"), VariablesResolver::Visibility::NAME);
	ASSERT_TRUE(main->visibility.contains("z"));
	ASSERT_EQ(main->visibility.at("z"), VariablesResolver::Visibility::NAME);
	ASSERT_TRUE(main->visibility.contains("c"));
	ASSERT_EQ(main->visibility.at("c"), VariablesResolver::Visibility::NAME);
	ASSERT_TRUE(main->visibility.contains("print"));
	ASSERT_EQ(main->visibility.at("print"), VariablesResolver::Visibility::NAME);
}

TEST(VariablesResolver, FunctionDefinition)
{
	static constexpr std::string_view program =
		"def foo():\n"
		"   a = 1\n"
		"   return a\n"
		"b = foo()\n";

	auto visibility = generate_resolver(program);
	ASSERT_EQ(visibility.size(), 2);

	ASSERT_TRUE(visibility.contains("_bytecode_generator_tests_"));
	auto &main = visibility.at("_bytecode_generator_tests_");
	ASSERT_EQ(main->visibility.size(), 2);
	ASSERT_TRUE(main->visibility.contains("b"));
	ASSERT_EQ(main->visibility.at("b"), VariablesResolver::Visibility::NAME);
	ASSERT_TRUE(main->visibility.contains("foo"));
	ASSERT_EQ(main->visibility.at("foo"), VariablesResolver::Visibility::NAME);

	ASSERT_TRUE(visibility.contains("_bytecode_generator_tests_.foo.0:0"));
	auto &foo = visibility.at("_bytecode_generator_tests_.foo.0:0");
	ASSERT_EQ(foo->visibility.size(), 1);
	ASSERT_TRUE(foo->visibility.contains("a"));
	ASSERT_EQ(foo->visibility.at("a"), VariablesResolver::Visibility::LOCAL);
}


TEST(VariablesResolver, Closure)
{
	static constexpr std::string_view program =
		"def foo(a):\n"
		"   print(a)\n"
		"   global c\n"
		"   b = 1\n"
		"   c = 1\n"
		"   def bar():\n"
		"       return a + b + c\n"
		"b = foo()\n";

	auto visibility = generate_resolver(program);
	ASSERT_EQ(visibility.size(), 3);

	ASSERT_TRUE(visibility.contains("_bytecode_generator_tests_"));
	auto &main = visibility.at("_bytecode_generator_tests_");
	ASSERT_EQ(main->visibility.size(), 2);
	ASSERT_TRUE(main->visibility.contains("b"));
	ASSERT_EQ(main->visibility.at("b"), VariablesResolver::Visibility::NAME);
	ASSERT_TRUE(main->visibility.contains("foo"));
	ASSERT_EQ(main->visibility.at("foo"), VariablesResolver::Visibility::NAME);

	ASSERT_TRUE(visibility.contains("_bytecode_generator_tests_.foo.0:0"));
	auto &foo = visibility.at("_bytecode_generator_tests_.foo.0:0");
	ASSERT_EQ(foo->visibility.size(), 5);
	ASSERT_TRUE(foo->visibility.contains("a"));
	ASSERT_EQ(foo->visibility.at("a"), VariablesResolver::Visibility::CELL);
	ASSERT_TRUE(foo->visibility.contains("b"));
	ASSERT_EQ(foo->visibility.at("b"), VariablesResolver::Visibility::CELL);
	ASSERT_TRUE(foo->visibility.contains("c"));
	ASSERT_EQ(foo->visibility.at("c"), VariablesResolver::Visibility::GLOBAL);
	ASSERT_TRUE(foo->visibility.contains("print"));
	ASSERT_EQ(foo->visibility.at("print"), VariablesResolver::Visibility::GLOBAL);
	ASSERT_TRUE(foo->visibility.contains("bar"));
	ASSERT_EQ(foo->visibility.at("bar"), VariablesResolver::Visibility::LOCAL);

	ASSERT_TRUE(visibility.contains("_bytecode_generator_tests_.foo.bar.5:3"));
	auto &foo_bar = visibility.at("_bytecode_generator_tests_.foo.bar.5:3");
	ASSERT_EQ(foo_bar->visibility.size(), 3);
	ASSERT_TRUE(foo_bar->visibility.contains("a"));
	ASSERT_EQ(foo_bar->visibility.at("a"), VariablesResolver::Visibility::FREE);
	ASSERT_TRUE(foo_bar->visibility.contains("b"));
	ASSERT_EQ(foo_bar->visibility.at("b"), VariablesResolver::Visibility::FREE);
	ASSERT_TRUE(foo_bar->visibility.contains("c"));
	ASSERT_EQ(foo_bar->visibility.at("c"), VariablesResolver::Visibility::GLOBAL);
}

TEST(VariablesResolver, ClassDefinition)
{
	static constexpr std::string_view program =
		"class A:\n"
		"   pass\n"
		"\n"
		"class B:\n"
		"  def a(self):\n"
		"		return A\n";

	auto visibility = generate_resolver(program);
	(void)visibility;
}

TEST(VariablesResolver, LambdaDefinition)
{
	static constexpr std::string_view program = "a = lambda c: a + b + c\n";

	auto visibility = generate_resolver(program);
	ASSERT_EQ(visibility.size(), 2);

	ASSERT_TRUE(visibility.contains("_bytecode_generator_tests_"));
	auto &main = visibility.at("_bytecode_generator_tests_");
	ASSERT_EQ(main->visibility.size(), 1);
	ASSERT_TRUE(main->visibility.contains("a"));
	ASSERT_EQ(main->visibility.at("a"), VariablesResolver::Visibility::NAME);

	ASSERT_TRUE(visibility.contains("_bytecode_generator_tests_.<lambda>.0:4"));
	auto &lambda_ = visibility.at("_bytecode_generator_tests_.<lambda>.0:4");
	ASSERT_EQ(lambda_->visibility.size(), 3);
	ASSERT_TRUE(lambda_->visibility.contains("a"));
	ASSERT_EQ(lambda_->visibility.at("a"), VariablesResolver::Visibility::GLOBAL);
	ASSERT_TRUE(lambda_->visibility.contains("b"));
	ASSERT_EQ(lambda_->visibility.at("b"), VariablesResolver::Visibility::GLOBAL);
	ASSERT_TRUE(lambda_->visibility.contains("c"));
	ASSERT_EQ(lambda_->visibility.at("c"), VariablesResolver::Visibility::LOCAL);
}