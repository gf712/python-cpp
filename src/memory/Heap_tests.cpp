#include "Heap_test.hpp"
namespace {
static constexpr size_t chunk_size = 64;
}

TEST_F(TestHeap, AllocatesNewBlockWhenFull)
{
	struct Data : Cell
	{
		int64_t foo;
		Data(int64_t foo_) : foo(foo_) {}
		std::string to_string() const override { return "Data"; }
		void visit_graph(Visitor &) override {}
	};

	static_assert(sizeof(Data) + sizeof(GarbageCollected) > 16
				  && sizeof(Data) + sizeof(GarbageCollected) <= 32);

	[[maybe_unused]] auto scope = m_heap->scoped_gc_pause();

	const auto original_chunk_size = m_heap->slab().block_32()->chunks().size();

	const size_t n_chunks = original_chunk_size;

	for (size_t idx = 0; idx < n_chunks * chunk_size; ++idx) {
		m_heap->allocate<Data>(idx);
		ASSERT_EQ(original_chunk_size, m_heap->slab().block_32()->chunks().size());
	}

	auto *ptr = m_heap->allocate<Data>(n_chunks * chunk_size + 1);
	(void)ptr;

	ASSERT_LT(original_chunk_size, m_heap->slab().block_32()->chunks().size());
}

TEST_F(TestHeap, AllocatesInOldBlockWhenPossible)
{
	struct Data : Cell
	{
		int64_t foo;
		Data(int64_t foo_) : foo(foo_) {}
		std::string to_string() const override { return "Data"; }
		void visit_graph(Visitor &) override {}
	};

	static_assert(sizeof(Data) + sizeof(GarbageCollected) > 16
				  && sizeof(Data) + sizeof(GarbageCollected) <= 32);

	[[maybe_unused]] auto scope = m_heap->scoped_gc_pause();

	const auto original_chunk_size = m_heap->slab().block_32()->chunks().size();

	const size_t n_chunks = original_chunk_size;

	std::vector<Data *> data;
	data.reserve(n_chunks * chunk_size);
	for (size_t idx = 0; idx < n_chunks * chunk_size; ++idx) {
		data.push_back(m_heap->allocate<Data>(idx));
		ASSERT_EQ(original_chunk_size, m_heap->slab().block_32()->chunks().size());
	}

	static constexpr size_t index = 100;
	ASSERT(index < n_chunks * chunk_size);
	auto *old_ptr = data[index];
	m_heap->slab().block_32()->deallocate(bit_cast<uint8_t *>(old_ptr) - sizeof(GarbageCollected));

	auto *ptr = m_heap->allocate<Data>(n_chunks * chunk_size + 1);

	ASSERT_EQ(original_chunk_size, m_heap->slab().block_32()->chunks().size());
	ASSERT_EQ(ptr->foo, n_chunks * chunk_size + 1);
	ASSERT_EQ(ptr, old_ptr);
}

TEST_F(TestHeap, ResetCallsDestructorOfAllHeapAllocatecObjects)
{
	int64_t counter = 0;
	struct Data : Cell
	{
		int64_t foo;
		int64_t &m_counter;
		Data(int64_t foo_, int64_t &counter) : foo(foo_), m_counter(counter) {}
		~Data() { m_counter++; }
		std::string to_string() const override { return "Data"; }
		void visit_graph(Visitor &) override {}
	};

	static_assert(sizeof(Data) + sizeof(GarbageCollected) > 16
				  && sizeof(Data) + sizeof(GarbageCollected) <= 32);

	const auto original_chunk_size = m_heap->slab().block_32()->chunks().size();
	const size_t n_chunks = original_chunk_size;

	{
		[[maybe_unused]] auto scope = m_heap->scoped_gc_pause();
		std::vector<Data *> data;
		data.reserve(n_chunks * chunk_size);
		for (size_t idx = 0; idx < n_chunks * chunk_size; ++idx) {
			data.push_back(m_heap->allocate<Data>(idx, counter));
			ASSERT_EQ(original_chunk_size, m_heap->slab().block_32()->chunks().size());
		}
	}

	m_heap->reset();

	ASSERT_EQ(counter, n_chunks * chunk_size);
}