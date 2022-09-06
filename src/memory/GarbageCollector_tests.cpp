#include "gtest/gtest.h"

#include "GarbageCollector.hpp"
#include "Heap.hpp"

namespace {
struct Data : Cell
{
	int64_t foo;
	int64_t &m_counter;
	Data(int64_t foo_, int64_t &counter) : foo(foo_), m_counter(counter) {}
	~Data() { m_counter++; }
	std::string to_string() const override { return "Data"; }
	void visit_graph(Visitor &visitor) override { visitor.visit(*this); }
};

static_assert(
	sizeof(Data) + sizeof(GarbageCollected) > 16 && sizeof(Data) + sizeof(GarbageCollected) <= 32);

// helper function that makes sure all the allocations are performed in a new stack frame that is
// popped (and therefore the allocated GC pointers can be GC'ed)
#if defined(__clang__)
__attribute__((noinline, optnone)) void new_stack_frame_function(int64_t &counter)
#elif defined(__GNUC__)
__attribute__((noinline, optimize("-O0"))) void new_stack_frame_function(int64_t &counter)
#else
static_assert(false, "compiler not supported");
#endif
{
	auto &heap = Heap::the();

	auto *ptr1 = heap.allocate<Data>(1, counter);
	heap.collect_garbage();
	ASSERT_EQ(counter, 0);
	auto *ptr2 = heap.allocate<Data>(2, counter);
	heap.collect_garbage();
	ASSERT_EQ(counter, 0);
	auto *ptr3 = heap.allocate<Data>(3, counter);
	heap.collect_garbage();
	ASSERT_EQ(counter, 0);
	auto *ptr4 = heap.allocate<Data>(4, counter);
	heap.collect_garbage();
	ASSERT_EQ(counter, 0);
	auto *ptr5 = heap.allocate<Data>(5, counter);
	heap.collect_garbage();
	ASSERT_EQ(counter, 0);

	ASSERT_EQ(ptr1->foo, 1);
	ASSERT_EQ(ptr2->foo, 2);
	ASSERT_EQ(ptr3->foo, 3);
	ASSERT_EQ(ptr4->foo, 4);
	ASSERT_EQ(ptr5->foo, 5);
};
}// namespace

TEST(GarbageCollector, DoesNotDeallocateGCPointersOnTheStack)
{
	int64_t counter = 0;

	auto &heap = Heap::the();
	heap.garbage_collector().set_frequency(1);

	ASSERT_EQ(counter, 0);
	heap.collect_garbage();

	new_stack_frame_function(counter);

	heap.reset();
}

TEST(GarbageCollector, DeallocatesGCPointersWhenStackFrameIsPopped)
{
	int64_t counter = 0;

	auto &heap = Heap::the();
	heap.garbage_collector().set_frequency(1);

	ASSERT_EQ(counter, 0);
	heap.collect_garbage();

	new_stack_frame_function(counter);

	heap.collect_garbage();

	ASSERT_EQ(counter, 5);

	heap.reset();
}