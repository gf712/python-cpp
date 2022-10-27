#include "GarbageCollector.hpp"
#include "Heap_test.hpp"

namespace {

static int64_t g_counter = 0;

struct Data : Cell
{
	int64_t foo;
	Data(int64_t foo_) : foo(foo_) {}
	~Data() { g_counter++; }
	std::string to_string() const override { return "Data"; }
	void visit_graph(Visitor &visitor) override { visitor.visit(*this); }
};

static_assert(
	sizeof(Data) + sizeof(GarbageCollected) > 16 && sizeof(Data) + sizeof(GarbageCollected) <= 32);

// helper function that makes sure all the allocations are performed in a new stack frame that is
// popped (and therefore the allocated GC pointers can be GC'ed)
#if defined(__clang__)
__attribute__((noinline, optnone)) void new_stack_frame_function(Heap &heap)
#elif defined(__GNUC__)
__attribute__((noinline, optimize("-O0"))) void new_stack_frame_function(Heap &heap)
#else
static_assert(false, "compiler not supported");
#endif
{
	auto *ptr1 = heap.allocate<Data>(1);
	heap.collect_garbage();
	ASSERT_EQ(g_counter, 0);
	auto *ptr2 = heap.allocate<Data>(2);
	heap.collect_garbage();
	ASSERT_EQ(g_counter, 0);
	auto *ptr3 = heap.allocate<Data>(3);
	heap.collect_garbage();
	ASSERT_EQ(g_counter, 0);
	auto *ptr4 = heap.allocate<Data>(4);
	heap.collect_garbage();
	ASSERT_EQ(g_counter, 0);
	auto *ptr5 = heap.allocate<Data>(5);
	heap.collect_garbage();
	ASSERT_EQ(g_counter, 0);

	ASSERT_EQ(ptr1->foo, 1);
	ASSERT_EQ(ptr2->foo, 2);
	ASSERT_EQ(ptr3->foo, 3);
	ASSERT_EQ(ptr4->foo, 4);
	ASSERT_EQ(ptr5->foo, 5);
}
}// namespace

TEST_F(TestHeap, GarbageCollectorDoesNotDeallocateGCPointersOnTheStack)
{
	g_counter = 0;

	m_heap->garbage_collector().set_frequency(1);

	ASSERT_EQ(g_counter, 0);
	m_heap->collect_garbage();

	new_stack_frame_function(*m_heap);
}

TEST_F(TestHeap, GarbageCollectorDeallocatesGCPointersWhenStackFrameIsPopped)
{
	g_counter = 0;

	m_heap->garbage_collector().set_frequency(1);

	ASSERT_EQ(g_counter, 0);
	m_heap->collect_garbage();

	new_stack_frame_function(*m_heap);

	m_heap->collect_garbage();

	ASSERT_EQ(g_counter, 5);
}