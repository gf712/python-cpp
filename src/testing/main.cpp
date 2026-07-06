#include "interpreter/Interpreter.hpp"
#include "vm/VM.hpp"

#include "gtest/gtest.h"

#include <cxxopts.hpp>

#if defined(__SANITIZE_ADDRESS__)
#define PYTHON_ASAN_ENABLED
#elif defined(__has_feature)
#if __has_feature(address_sanitizer)
#define PYTHON_ASAN_ENABLED
#endif
#endif

#ifdef PYTHON_ASAN_ENABLED
// The garbage collector finds roots by conservatively scanning the machine
// stack. ASan's use-after-return detection relocates locals to a heap-backed
// fake stack that the scan never sees, so live GC pointers go unnoticed and
// stale ones linger, breaking the collector's reachability tests.
extern "C" const char *__asan_default_options() { return "detect_stack_use_after_return=0"; }
#endif

class PythonVMEnvironment : public ::testing::Environment
{
	char **m_argv;

  public:
	PythonVMEnvironment(char **argv) : m_argv(argv) {}
	~PythonVMEnvironment() override {}

	void SetUp() override
	{
		auto &vm = VirtualMachine::the();
		vm.heap().set_start_stack_pointer(bit_cast<uintptr_t *>(m_argv));
		initialize_types();
	}

	void TearDown() override {}
};

int main(int argc, char **argv)
{
	::testing::InitGoogleTest(&argc, argv);

	(void)AddGlobalTestEnvironment(new PythonVMEnvironment(argv));

	cxxopts::Options options("python-testing", "The C++ Python interpreter tests");

	// clang-format off
	options.add_options()
		("d,debug", "Enable debug logging", cxxopts::value<bool>()->default_value("false"))
		("trace", "Enable trace logging", cxxopts::value<bool>()->default_value("false"))
		("h,help", "Print usage");

	auto result = options.parse(argc, argv);

	if (result.count("help")) {
		std::cout << options.help() << std::endl;
		return EXIT_SUCCESS;
	}

	const bool debug = result["debug"].as<bool>();
	const bool trace = result["trace"].as<bool>();
	if (debug) { spdlog::set_level(spdlog::level::debug); }
	if (trace) { spdlog::set_level(spdlog::level::trace); }

	return RUN_ALL_TESTS();
}