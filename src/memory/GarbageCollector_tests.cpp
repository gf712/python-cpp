#include "core.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>

import py.memory;
import py.runtime;
import std;

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

// Overwrites the stack region just vacated by a popped frame. Unoptimized
// builds give every temporary its own slot and the collector's call chain
// does not reliably overwrite all of them before the conservative scan runs,
// so a stale copy of a dead GC pointer can be picked up as a root. Zeroing
// the dead region makes collection of unreachable objects deterministic.
// The scrub stays within `buffer`, which cannot reach the last few words
// below this function's frame header (alignment padding and compiler-placed
// slots sit there); callers must run the allocating helper through
// call_in_padded_frame so its residue lands below that blind spot.
__attribute__((noinline)) void scrub_dead_stack()
{
	uint8_t buffer[16 * 1024];
	std::memset(buffer, 0, sizeof(buffer));
	// keep the memset from being eliminated as a dead store
	asm volatile("" ::"r"(buffer) : "memory");
}

// Runs fn with its stack frame pushed at least sizeof(pad) bytes deeper than
// the caller's, so every slot fn writes lies inside the span that
// scrub_dead_stack can zero without stepping past its buffer's bounds.
template<typename Fn> __attribute__((noinline)) void call_in_padded_frame(Fn &&fn)
{
	volatile uint8_t pad[256] = {};
	fn();
	// volatile read keeps pad live across the call, preventing a tail call
	// that would collapse this frame into fn's
	(void)pad[0];
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

	call_in_padded_frame([this] { new_stack_frame_function(*m_heap); });

	scrub_dead_stack();
	m_heap->collect_garbage();

	ASSERT_EQ(g_counter, 5);
}

namespace {

struct Cycle : Cell
{
	Cycle *other{ nullptr };
	int64_t &counter;
	explicit Cycle(int64_t &counter_) : counter(counter_) {}
	~Cycle() { counter++; }
	std::string to_string() const override { return "Cycle"; }
	void visit_graph(Visitor &visitor) override
	{
		visitor.visit(*this);
		if (other) { visitor.visit(*other); }
	}
};

// Allocates two mutually-referencing Cycle objects in a popped frame so
// the conservative stack scan can't keep them alive past return.
#if defined(__clang__)
__attribute__((noinline, optnone)) void allocate_cycle_in_popped_frame(Heap &heap, int64_t &counter)
#elif defined(__GNUC__)
__attribute__((noinline, optimize("-O0"))) void allocate_cycle_in_popped_frame(Heap &heap,
	int64_t &counter)
#endif
{
	auto *a = heap.allocate<Cycle>(counter);
	auto *b = heap.allocate<Cycle>(counter);
	a->other = b;
	b->other = a;
}

}// namespace

TEST_F(TestHeap, MutuallyReferencingObjectsAreCollected)
{
	// Two Cycle objects pointing at each other should be collected once
	// no external roots reach them — this is the canonical mark-sweep
	// reachability test that a refcount-only GC would fail.
	int64_t counter = 0;
	m_heap->garbage_collector().set_frequency(1);

	call_in_padded_frame([&] { allocate_cycle_in_popped_frame(*m_heap, counter); });

	scrub_dead_stack();
	m_heap->collect_garbage();

	ASSERT_EQ(counter, 2);
}
