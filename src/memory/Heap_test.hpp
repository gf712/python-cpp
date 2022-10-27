#pragma once

#include "Heap.hpp"
#include "vm/VM.hpp"

#include <gtest/gtest.h>

struct TestHeap : ::testing::Test
{
	std::unique_ptr<Heap> m_heap;
	uintptr_t *m_start_stack_pointer{ nullptr };

	void SetUp() final
	{
		if (!m_start_stack_pointer) {
			// VirtualMachine is initialized in PythonVMEnvironment (testing/main.cpp)
			// which knows the address of the initial stack
			m_start_stack_pointer = VirtualMachine::the().heap().start_stack_pointer();
		}
		m_heap = Heap::create();
		m_heap->set_start_stack_pointer(m_start_stack_pointer);
	}

	void TearDown() final { m_heap->reset(); }
};