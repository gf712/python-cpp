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
	ASSERT_EQ(main->symbol_map.symbols.size(), 7);
	ASSERT_TRUE(main->symbol_map.get_visible_symbol("a").has_value());
	ASSERT_EQ(main->symbol_map.get_visible_symbol("a")->get().visibility,
		VariablesResolver::Visibility::IMPLICIT_GLOBAL);
	ASSERT_TRUE(main->symbol_map.get_visible_symbol("foo").has_value());
	ASSERT_EQ(main->symbol_map.get_visible_symbol("foo")->get().visibility,
		VariablesResolver::Visibility::NAME);
	ASSERT_TRUE(main->symbol_map.get_visible_symbol("b").has_value());
	ASSERT_EQ(main->symbol_map.get_visible_symbol("b")->get().visibility,
		VariablesResolver::Visibility::IMPLICIT_GLOBAL);
	ASSERT_TRUE(main->symbol_map.get_visible_symbol("z").has_value());
	ASSERT_EQ(main->symbol_map.get_visible_symbol("z")->get().visibility,
		VariablesResolver::Visibility::NAME);
	ASSERT_TRUE(main->symbol_map.get_visible_symbol("c").has_value());
	ASSERT_EQ(main->symbol_map.get_visible_symbol("c")->get().visibility,
		VariablesResolver::Visibility::IMPLICIT_GLOBAL);
	ASSERT_TRUE(main->symbol_map.get_visible_symbol("print").has_value());
	ASSERT_EQ(main->symbol_map.get_visible_symbol("print")->get().visibility,
		VariablesResolver::Visibility::NAME);
	ASSERT_EQ(main->symbol_map.get_visible_symbol("__name__")->get().visibility,
		VariablesResolver::Visibility::IMPLICIT_GLOBAL);
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
	ASSERT_EQ(main->symbol_map.symbols.size(), 3);
	ASSERT_TRUE(main->symbol_map.get_visible_symbol("b").has_value());
	ASSERT_EQ(main->symbol_map.get_visible_symbol("b")->get().visibility,
		VariablesResolver::Visibility::IMPLICIT_GLOBAL);
	ASSERT_TRUE(main->symbol_map.get_visible_symbol("foo").has_value());
	ASSERT_EQ(main->symbol_map.get_visible_symbol("foo")->get().visibility,
		VariablesResolver::Visibility::IMPLICIT_GLOBAL);

	ASSERT_TRUE(visibility.contains("_bytecode_generator_tests_.foo.0:0"));
	auto &foo = visibility.at("_bytecode_generator_tests_.foo.0:0");
	ASSERT_EQ(foo->symbol_map.symbols.size(), 1);
	ASSERT_TRUE(foo->symbol_map.get_visible_symbol("a").has_value());
	ASSERT_EQ(foo->symbol_map.get_visible_symbol("a")->get().visibility,
		VariablesResolver::Visibility::LOCAL);
	ASSERT_EQ(main->symbol_map.get_visible_symbol("__name__")->get().visibility,
		VariablesResolver::Visibility::IMPLICIT_GLOBAL);
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
	ASSERT_EQ(main->symbol_map.symbols.size(), 3);
	ASSERT_TRUE(main->symbol_map.get_visible_symbol("b").has_value());
	ASSERT_EQ(main->symbol_map.get_visible_symbol("b")->get().visibility,
		VariablesResolver::Visibility::IMPLICIT_GLOBAL);
	ASSERT_TRUE(main->symbol_map.get_visible_symbol("foo").has_value());
	ASSERT_EQ(main->symbol_map.get_visible_symbol("foo")->get().visibility,
		VariablesResolver::Visibility::IMPLICIT_GLOBAL);
	ASSERT_EQ(main->symbol_map.get_visible_symbol("__name__")->get().visibility,
		VariablesResolver::Visibility::IMPLICIT_GLOBAL);

	ASSERT_TRUE(visibility.contains("_bytecode_generator_tests_.foo.0:0"));
	auto &foo = visibility.at("_bytecode_generator_tests_.foo.0:0");
	ASSERT_EQ(foo->symbol_map.symbols.size(), 5);
	ASSERT_TRUE(foo->symbol_map.get_visible_symbol("a").has_value());
	ASSERT_EQ(foo->symbol_map.get_visible_symbol("a")->get().visibility,
		VariablesResolver::Visibility::CELL);
	ASSERT_TRUE(foo->symbol_map.get_visible_symbol("b").has_value());
	ASSERT_EQ(foo->symbol_map.get_visible_symbol("b")->get().visibility,
		VariablesResolver::Visibility::CELL);
	ASSERT_TRUE(foo->symbol_map.get_visible_symbol("c").has_value());
	ASSERT_EQ(foo->symbol_map.get_visible_symbol("c")->get().visibility,
		VariablesResolver::Visibility::EXPLICIT_GLOBAL);
	ASSERT_TRUE(foo->symbol_map.get_visible_symbol("print").has_value());
	ASSERT_EQ(foo->symbol_map.get_visible_symbol("print")->get().visibility,
		VariablesResolver::Visibility::IMPLICIT_GLOBAL);
	ASSERT_TRUE(foo->symbol_map.get_visible_symbol("bar").has_value());
	ASSERT_EQ(foo->symbol_map.get_visible_symbol("bar")->get().visibility,
		VariablesResolver::Visibility::LOCAL);

	ASSERT_TRUE(visibility.contains("_bytecode_generator_tests_.foo.bar.5:3"));
	auto &foo_bar = visibility.at("_bytecode_generator_tests_.foo.bar.5:3");
	ASSERT_EQ(foo_bar->symbol_map.symbols.size(), 3);
	ASSERT_TRUE(foo_bar->symbol_map.get_visible_symbol("a").has_value());
	ASSERT_EQ(foo_bar->symbol_map.get_visible_symbol("a")->get().visibility,
		VariablesResolver::Visibility::FREE);
	ASSERT_TRUE(foo_bar->symbol_map.get_visible_symbol("b").has_value());
	ASSERT_EQ(foo_bar->symbol_map.get_visible_symbol("b")->get().visibility,
		VariablesResolver::Visibility::FREE);
	ASSERT_TRUE(foo_bar->symbol_map.get_visible_symbol("c").has_value());
	ASSERT_EQ(foo_bar->symbol_map.get_visible_symbol("c")->get().visibility,
		VariablesResolver::Visibility::EXPLICIT_GLOBAL);
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

TEST(VariablesResolver, NonLocal)
{
	static constexpr std::string_view program =
		"a = 1\n"
		"b = 1\n"
		"def outer():\n"
		"   a = 1\n"
		"   b = 1\n"
		"   def inner():\n"
		"      nonlocal b\n"
		"      a = 2\n"
		"      b = 2\n";

	auto visibility = generate_resolver(program);
	ASSERT_EQ(visibility.size(), 3);

	ASSERT_TRUE(visibility.contains("_bytecode_generator_tests_"));
	auto &main = visibility.at("_bytecode_generator_tests_");
	ASSERT_EQ(main->symbol_map.symbols.size(), 4);
	ASSERT_TRUE(main->symbol_map.get_visible_symbol("a").has_value());
	ASSERT_TRUE(main->symbol_map.get_visible_symbol("b").has_value());
	ASSERT_EQ(main->symbol_map.get_visible_symbol("a")->get().visibility,
		VariablesResolver::Visibility::IMPLICIT_GLOBAL);
	ASSERT_EQ(main->symbol_map.get_visible_symbol("b")->get().visibility,
		VariablesResolver::Visibility::IMPLICIT_GLOBAL);
	ASSERT_EQ(main->symbol_map.get_visible_symbol("__name__")->get().visibility,
		VariablesResolver::Visibility::IMPLICIT_GLOBAL);

	ASSERT_TRUE(visibility.contains("_bytecode_generator_tests_.outer.2:0"));
	auto &outer = visibility.at("_bytecode_generator_tests_.outer.2:0");
	ASSERT_EQ(outer->symbol_map.symbols.size(), 3);
	ASSERT_TRUE(outer->symbol_map.get_visible_symbol("inner").has_value());
	ASSERT_EQ(outer->symbol_map.get_visible_symbol("inner")->get().visibility,
		VariablesResolver::Visibility::LOCAL);
	ASSERT_TRUE(outer->symbol_map.get_visible_symbol("a").has_value());
	ASSERT_EQ(outer->symbol_map.get_visible_symbol("a")->get().visibility,
		VariablesResolver::Visibility::LOCAL);
	ASSERT_TRUE(outer->symbol_map.get_visible_symbol("b").has_value());
	ASSERT_EQ(outer->symbol_map.get_visible_symbol("b")->get().visibility,
		VariablesResolver::Visibility::CELL);

	ASSERT_TRUE(visibility.contains("_bytecode_generator_tests_.outer.inner.5:3"));
	auto &inner = visibility.at("_bytecode_generator_tests_.outer.inner.5:3");
	ASSERT_EQ(inner->symbol_map.symbols.size(), 2);
	ASSERT_TRUE(inner->symbol_map.get_visible_symbol("a").has_value());
	ASSERT_EQ(inner->symbol_map.get_visible_symbol("a")->get().visibility,
		VariablesResolver::Visibility::LOCAL);
	ASSERT_TRUE(inner->symbol_map.get_visible_symbol("b").has_value());
	ASSERT_EQ(inner->symbol_map.get_visible_symbol("b")->get().visibility,
		VariablesResolver::Visibility::FREE);
}

TEST(VariablesResolver, LambdaDefinition)
{
	static constexpr std::string_view program = "a = lambda c: a + b + c\n";

	auto visibility = generate_resolver(program);
	ASSERT_EQ(visibility.size(), 2);

	ASSERT_TRUE(visibility.contains("_bytecode_generator_tests_"));
	auto &main = visibility.at("_bytecode_generator_tests_");
	ASSERT_EQ(main->symbol_map.symbols.size(), 2);
	ASSERT_TRUE(main->symbol_map.get_visible_symbol("a").has_value());
	ASSERT_EQ(main->symbol_map.get_visible_symbol("a")->get().visibility,
		VariablesResolver::Visibility::IMPLICIT_GLOBAL);
	ASSERT_EQ(main->symbol_map.get_visible_symbol("__name__")->get().visibility,
		VariablesResolver::Visibility::IMPLICIT_GLOBAL);

	ASSERT_TRUE(visibility.contains("_bytecode_generator_tests_.<lambda>.0:4"));
	auto &lambda_ = visibility.at("_bytecode_generator_tests_.<lambda>.0:4");
	ASSERT_EQ(lambda_->symbol_map.symbols.size(), 3);
	ASSERT_TRUE(lambda_->symbol_map.get_visible_symbol("a").has_value());
	ASSERT_EQ(lambda_->symbol_map.get_visible_symbol("a")->get().visibility,
		VariablesResolver::Visibility::IMPLICIT_GLOBAL);
	ASSERT_TRUE(lambda_->symbol_map.get_visible_symbol("b").has_value());
	ASSERT_EQ(lambda_->symbol_map.get_visible_symbol("b")->get().visibility,
		VariablesResolver::Visibility::IMPLICIT_GLOBAL);
	ASSERT_TRUE(lambda_->symbol_map.get_visible_symbol("c").has_value());
	ASSERT_EQ(lambda_->symbol_map.get_visible_symbol("c")->get().visibility,
		VariablesResolver::Visibility::LOCAL);
}
