#pragma once

#include "GarbageCollector.hpp"
#include "runtime/PyObject.hpp"
#include "utilities.hpp"

#include <bitset>
#include <memory>
#include <optional>

static constexpr size_t KB = 1024;
static constexpr size_t MB = 1024 * KB;


class Block
{
	class Chunk : NonCopyable
	{
		template<size_t ChunkCount_ = 64> class ChunkView
		{
		  public:
			static constexpr size_t ChunkCount = ChunkCount_;

			bool has_free_chunk() const { return !m_occupied_chunks.all(); }

			std::optional<size_t> next_free_chunk() const
			{
				if (has_free_chunk()) {
					size_t i{ 0 };
					while (m_occupied_chunks[i]) { i++; }
					return i;
				} else {
					return {};
				}
			}

			std::optional<size_t> mark_next_free_chunk()
			{
				spdlog::trace(
					"mark_next_free_chunk -> chunk bit mask: {}", m_occupied_chunks.to_string());
				if (auto chunk_idx = next_free_chunk()) {
					ASSERT(!m_occupied_chunks[*chunk_idx])
					spdlog::trace("marking next free chunk -> old chunk bit mask: {}",
						m_occupied_chunks.to_string());
					m_occupied_chunks.flip(*chunk_idx);
					spdlog::trace("marking next free chunk -> new chunk bit mask: {} (index: {})",
						m_occupied_chunks.to_string(),
						*chunk_idx);
					return *chunk_idx;
				} else {
					return {};
				}
			}

			void mark_chunk_as_free(size_t idx)
			{
				ASSERT(m_occupied_chunks[idx])
				m_occupied_chunks.flip(idx);
			}

			void set() { m_occupied_chunks.set(); }

			void reset() { m_occupied_chunks.reset(); }

			std::bitset<ChunkCount> m_occupied_chunks;
		};

	  public:
		Chunk(uint8_t *memory, size_t object_size) : m_memory(memory), m_object_size(object_size) {}

		~Chunk();

		Chunk(Chunk &&other) noexcept
			: m_memory(other.m_memory), m_object_size(other.m_object_size),
			  m_chunk_view(other.m_chunk_view)
		{
			other.m_memory = nullptr;
			other.m_chunk_view.reset();
		}

		uint8_t *allocate()
		{
			if (auto chunk_idx = m_chunk_view.mark_next_free_chunk()) {
				spdlog::trace("Allocating memory at index {}, address {} (chunk base address @{})",
					*chunk_idx,
					(void *)(m_memory + *chunk_idx * m_object_size),
					(void *)m_memory);
				return m_memory + *chunk_idx * m_object_size;
			} else {
				return nullptr;
			}
		}

		bool has_address(uint8_t *memory) const;

		void deallocate(uint8_t *ptr)
		{
			uintptr_t start = bit_cast<uintptr_t>(m_memory);
			uintptr_t end =
				bit_cast<uintptr_t>(m_memory + (m_object_size + 1) * ChunkView<>::ChunkCount);
			ASSERT(bit_cast<uintptr_t>(ptr) >= start && bit_cast<uintptr_t>(ptr) < end)
			ASSERT((bit_cast<uintptr_t>(ptr) - start) % m_object_size == 0)

			size_t ptr_idx = (bit_cast<uintptr_t>(ptr) - start) / m_object_size;
			ASSERT(ptr_idx < ChunkView<>::ChunkCount)
			m_chunk_view.mark_chunk_as_free(ptr_idx);
			spdlog::debug("Marking memory at index {} as free, address {}",
				ptr_idx,
				(void *)(m_memory + ptr_idx * m_object_size));

			// TODO: this should only be needed for debug builds
			std::fill_n(ptr, m_object_size, 0xDD);
		}

		void set_all_in_mask(bool value)
		{
			if (value) {
				m_chunk_view.set();
			} else {
				m_chunk_view.reset();
			}
		}

		void reset();

		size_t object_size() const { return m_object_size; }

		void for_each_cell_alive(std::function<void(uint8_t *)> &&callback)
		{
			for (size_t i{ 0 }; i < m_chunk_view.m_occupied_chunks.size(); ++i) {
				const auto bit = m_chunk_view.m_occupied_chunks[i];
				if (bit) { callback(m_memory + i * m_object_size); }
			}
		}

		void for_each_cell(std::function<void(uint8_t *)> &&callback)
		{
			for (size_t i{ 0 }; i < m_chunk_view.m_occupied_chunks.size(); ++i) {
				callback(m_memory + i * m_object_size);
			}
		}

		uint8_t *m_memory;
		size_t m_object_size;

	  private:
		ChunkView<> m_chunk_view;
	};


  public:
	Block(size_t object_size, size_t capacity);

	void reset();

	uint8_t *allocate();

	void deallocate(uint8_t *ptr);

	std::vector<Chunk> &chunks() { return m_chunks; }

	size_t object_size() const { return m_chunks.back().object_size(); }

  private:
	std::vector<std::unique_ptr<uint8_t[]>> m_memory;
	std::vector<Chunk> m_chunks;
};

class Slab
{
  public:
	Slab()
	{
		block16 = std::make_unique<Block>(16, 1000);
		block32 = std::make_unique<Block>(32, 1000);
		block64 = std::make_unique<Block>(64, 1000);
		block128 = std::make_unique<Block>(128, 1000);
		block256 = std::make_unique<Block>(256, 1000);
		block512 = std::make_unique<Block>(512, 1000);
		block1024 = std::make_unique<Block>(1024, 1000);
		block2048 = std::make_unique<Block>(2048, 1000);
	}

	std::unique_ptr<Block> &block_16() { return block16; }
	std::unique_ptr<Block> &block_32() { return block32; }
	std::unique_ptr<Block> &block_64() { return block64; }
	std::unique_ptr<Block> &block_128() { return block128; }
	std::unique_ptr<Block> &block_256() { return block256; }
	std::unique_ptr<Block> &block_512() { return block512; }
	std::unique_ptr<Block> &block_1024() { return block1024; }
	std::unique_ptr<Block> &block_2048() { return block2048; }

	template<typename T> uint8_t *allocate() requires std::is_base_of_v<Cell, T>
	{
		spdlog::trace("Allocating Cell object memory for object of size {}", sizeof(T));
		if constexpr (sizeof(T) + sizeof(GarbageCollected) <= 16) { return block16->allocate(); }
		if constexpr (sizeof(T) + sizeof(GarbageCollected) <= 32) { return block32->allocate(); }
		if constexpr (sizeof(T) + sizeof(GarbageCollected) <= 64) { return block64->allocate(); }
		if constexpr (sizeof(T) + sizeof(GarbageCollected) <= 128) { return block128->allocate(); }
		if constexpr (sizeof(T) + sizeof(GarbageCollected) <= 256) { return block256->allocate(); }
		if constexpr (sizeof(T) + sizeof(GarbageCollected) <= 512) { return block512->allocate(); }
		if constexpr (sizeof(T) + sizeof(GarbageCollected) <= 1024) {
			return block1024->allocate();
		}
		if constexpr (sizeof(T) + sizeof(GarbageCollected) <= 2048) {
			return block2048->allocate();
		} else {
			[]<bool flag = false>()
			{
				static_assert(flag, "only object sizes <= 2048 bytes are currently supported");
			}
			();
		}
	}

	template<typename T> uint8_t *allocate()
	{
		[]<bool flag = false>()
		{
			static_assert(flag, "only GC collected objects are currently supported");
		}
		();
		return nullptr;

		// uint8_t *ptr{ nullptr };
		// spdlog::debug("Allocating memory for object of size {}", sizeof(T));
		// if constexpr (sizeof(T) <= 16) { ptr = block16->allocate(); }
		// if constexpr (sizeof(T) <= 32) { ptr = block32->allocate(); }
		// if constexpr (sizeof(T) <= 64) { ptr = block64->allocate(); }
		// if constexpr (sizeof(T) <= 128) { ptr = block128->allocate(); }
		// if constexpr (sizeof(T) <= 256) { ptr = block256->allocate(); }
		// if constexpr (sizeof(T) <= 512) {
		// 	return block512->allocate();
		// } else {
		// 	[]<bool flag = false>()
		// 	{
		// 		static_assert(flag, "only object sizes <= 512 bytes are currently supported");
		// 	}
		// 	();
		// }
		// new (ptr + sizeof(GarbageCollected)) T(std::forward<Args>(args)...);
		// return ptr;
	}

	bool has_address(uint8_t *addr) const;

	void reset()
	{
		block16->reset();
		block32->reset();
		block64->reset();
		block128->reset();
		block256->reset();
		block512->reset();
		block1024->reset();
		block2048->reset();
	}

  private:
	std::unique_ptr<Block> block16{ nullptr };
	std::unique_ptr<Block> block32{ nullptr };
	std::unique_ptr<Block> block64{ nullptr };
	std::unique_ptr<Block> block128{ nullptr };
	std::unique_ptr<Block> block256{ nullptr };
	std::unique_ptr<Block> block512{ nullptr };
	std::unique_ptr<Block> block1024{ nullptr };
	std::unique_ptr<Block> block2048{ nullptr };
};

class Heap
	: NonCopyable
	, NonMoveable
{
	friend class VirtualMachine;
	friend struct TestHeap;

	std::unique_ptr<uint8_t[]> m_static_memory;
	size_t m_static_memory_size{ 1 * MB };
	size_t m_static_offset{ 8 };
	Slab m_slab;
	std::unique_ptr<GarbageCollector> m_gc;
	uintptr_t *m_bottom_stack_pointer;
	bool m_allocate_in_static{ false };

	struct ScopedGCPause
	{
		GarbageCollector &gc_;
		bool m_needs_to_be_resumed{ false };
		ScopedGCPause(GarbageCollector &gc) : gc_(gc)
		{
			if (gc_.is_active()) {
				m_needs_to_be_resumed = true;
				gc_.pause();
			}
		}
		~ScopedGCPause()
		{
			if (m_needs_to_be_resumed) { gc_.resume(); }
		}
	};

	struct ScopedStaticAllocation
	{
		Heap &heap;
		ScopedStaticAllocation(Heap &heap_) : heap(heap_) { heap.m_allocate_in_static = true; }
		~ScopedStaticAllocation() { heap.m_allocate_in_static = false; }
	};

	friend ScopedStaticAllocation;

	static std::unique_ptr<Heap> create() { return std::unique_ptr<Heap>(new Heap); }

  public:
	void reset()
	{
		collect_garbage();
		m_slab.reset();
	}

	void set_start_stack_pointer(uintptr_t *address) { m_bottom_stack_pointer = address; }

	uintptr_t *start_stack_pointer() const { return m_bottom_stack_pointer; }

	template<typename T, typename... Args> T * __attribute__ ((noinline)) allocate(Args &&...args)
	{
		if (m_allocate_in_static) { return allocate_static<T>(std::forward<Args>(args)...).get(); }
		collect_garbage();
		auto *ptr = m_slab.allocate<T>();

		uint8_t *obj_ptr = allocate_gc(ptr);
		T *obj = new (obj_ptr) T(std::forward<Args>(args)...);

		if constexpr (std::is_base_of_v<py::PyObject, T>)
			log_allocation(static_cast<py::PyObject *>(obj));
		return obj;
	}

	void collect_garbage();

	GarbageCollector &garbage_collector()
	{
		ASSERT(m_gc);
		return *m_gc;
	}

	template<typename T, typename... Args> std::shared_ptr<T> allocate_static(Args &&...args)
	{
		if (m_static_offset + sizeof(T) >= m_static_memory_size) { TODO(); }
		T *ptr = new (m_static_memory.get() + m_static_offset) T(std::forward<Args>(args)...);
		m_static_offset += sizeof(T);
		return std::shared_ptr<T>(ptr, [](T *) { return; });
	}

	const uint8_t *static_memory() const { return m_static_memory.get(); }
	size_t static_memory_size() const { return m_static_memory_size; }

	Slab &slab() { return m_slab; }
	const Slab &slab() const { return m_slab; }

	[[nodiscard]] ScopedGCPause scoped_gc_pause() { return ScopedGCPause(*m_gc); }

	[[nodiscard]] ScopedStaticAllocation scoped_static_allocation()
	{
		return ScopedStaticAllocation(*this);
	}

  private:
	uint8_t *allocate_gc(uint8_t *ptr) const;
	void log_allocation(py::PyObject *obj) const;

	Heap();
};