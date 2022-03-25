#include "vm/VM.hpp"

#include "gtest/gtest.h"

class PythonVMEnvironment : public ::testing::Environment
{
  public:
	~PythonVMEnvironment() override {}

	void SetUp() override { (void)VirtualMachine::the(); }

	void TearDown() override {}
};

int main(int argc, char **argv)
{
	::testing::InitGoogleTest(&argc, argv);

	(void)AddGlobalTestEnvironment(new PythonVMEnvironment);

	return RUN_ALL_TESTS();
}