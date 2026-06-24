#include "ast/ASTArena.hpp"

#include "gtest/gtest.h"

#include <cstddef>
#include <cstdint>
#include <string>

namespace {

struct Trivial
{
	int a;
	int b;
};

struct WithDestructor
{
	int *counter;
	explicit WithDestructor(int *c) : counter(c) {}
	~WithDestructor() { ++*counter; }
};

struct OverAligned
{
	alignas(64) std::int64_t value;
};

}// namespace

TEST(ASTArena, AllocatesTrivialType)
{
	ast::ASTArena arena;
	auto *obj = arena.create<Trivial>();
	ASSERT_NE(obj, nullptr);
	obj->a = 7;
	obj->b = 42;
	EXPECT_EQ(obj->a, 7);
	EXPECT_EQ(obj->b, 42);
}

TEST(ASTArena, ForwardsConstructorArgs)
{
	ast::ASTArena arena;
	auto *s = arena.create<std::string>("hello arena");
	ASSERT_NE(s, nullptr);
	EXPECT_EQ(*s, "hello arena");
}

TEST(ASTArena, CallsDestructorsOnArenaDestruction)
{
	int count = 0;
	{
		ast::ASTArena arena;
		arena.create<WithDestructor>(&count);
		arena.create<WithDestructor>(&count);
		arena.create<WithDestructor>(&count);
		EXPECT_EQ(count, 0);
	}
	EXPECT_EQ(count, 3);
}

TEST(ASTArena, DoesNotTrackTriviallyDestructible)
{
	// Trivially-destructible types should not consume destructor-list entries.
	// We verify this indirectly: allocate many trivial objects and confirm the
	// arena still works (no crash, sane byte count).
	ast::ASTArena arena;
	for (int i = 0; i < 10'000; ++i) { arena.create<Trivial>(); }
	EXPECT_GE(arena.bytes_allocated(), 10'000 * sizeof(Trivial));
}

TEST(ASTArena, RespectsAlignmentForOverAlignedTypes)
{
	ast::ASTArena arena;
	// Allocate a 1-byte hole first to force the next allocation to be aligned.
	(void)arena.create<char>();
	auto *obj = arena.create<OverAligned>();
	auto addr = reinterpret_cast<std::uintptr_t>(obj);
	EXPECT_EQ(addr % alignof(OverAligned), 0u);
}

TEST(ASTArena, GrowsAcrossManySlabs)
{
	// Force multiple slabs by allocating well past the initial slab size.
	ast::ASTArena arena;
	const std::size_t n = 200'000;
	for (std::size_t i = 0; i < n; ++i) { arena.create<Trivial>(); }
	EXPECT_GE(arena.bytes_allocated(), n * sizeof(Trivial));
}

TEST(ASTArena, ReturnsStablePointers)
{
	// Pointers must remain valid after later allocations trigger slab growth.
	ast::ASTArena arena;
	auto *first = arena.create<Trivial>();
	first->a = 1;
	first->b = 2;
	for (int i = 0; i < 100'000; ++i) { arena.create<Trivial>(); }
	EXPECT_EQ(first->a, 1);
	EXPECT_EQ(first->b, 2);
}
