#include "vm/VM.hpp"

#include "gtest/gtest.h"

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
	}

	void TearDown() override {}
};

int main(int argc, char **argv)
{
	::testing::InitGoogleTest(&argc, argv);

	(void)AddGlobalTestEnvironment(new PythonVMEnvironment(argv));

	return RUN_ALL_TESTS();
}