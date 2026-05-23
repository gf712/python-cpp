#pragma once

#include "GarbageCollector.hpp"
#include "runtime/forward.hpp"
#include "utilities.hpp"

#include <algorithm>
#include <array>
#include <bitset>
#include <cstdint>
#include <memory>
#include <optional>
#include <unordered_map>

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
					ASSERT(!m_occupied_chunks[*chunk_idx]);
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
				ASSERT(m_occupied_chunks[idx]);
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
			uintptr_t end = bit_cast<uintptr_t>(m_memory + m_object_size * ChunkView<>::ChunkCount);
			ASSERT(bit_cast<uintptr_t>(ptr) >= start && bit_cast<uintptr_t>(ptr) < end);
			ASSERT((bit_cast<uintptr_t>(ptr) - start) % m_object_size == 0);

			size_t ptr_idx = (bit_cast<uintptr_t>(ptr) - start) / m_object_size;
			ASSERT(ptr_idx < ChunkView<>::ChunkCount);
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
	const std::vector<Chunk> &chunks() const { return m_chunks; }

	size_t object_size() const { return m_chunks.back().object_size(); }

  private:
	std::vector<std::unique_ptr<uint8_t[]>> m_memory;
	std::vector<Chunk> m_chunks;
};

class Slab
{
  public:
	static constexpr std::array<size_t, 8> kBlockSizes{ 16, 32, 64, 128, 256, 512, 1024, 2048 };
	static constexpr size_t kBlockCount = kBlockSizes.size();

	Slab()
	{
		for (size_t i = 0; i < kBlockCount; ++i) {
			m_blocks[i] = std::make_unique<Block>(kBlockSizes[i], 1000);
		}
	}

	std::unique_ptr<Block> &block_16() { return m_blocks[0]; }
	std::unique_ptr<Block> &block_32() { return m_blocks[1]; }
	std::unique_ptr<Block> &block_64() { return m_blocks[2]; }
	std::unique_ptr<Block> &block_128() { return m_blocks[3]; }
	std::unique_ptr<Block> &block_256() { return m_blocks[4]; }
	std::unique_ptr<Block> &block_512() { return m_blocks[5]; }
	std::unique_ptr<Block> &block_1024() { return m_blocks[6]; }
	std::unique_ptr<Block> &block_2048() { return m_blocks[7]; }

	template<typename F> void for_each_block(F &&f)
	{
		for (auto &b : m_blocks) { f(*b); }
	}
	template<typename F> void for_each_block(F &&f) const
	{
		for (const auto &b : m_blocks) { f(*b); }
	}

	template<typename T>
	uint8_t *allocate()
		requires std::is_base_of_v<Cell, T>
	{
		spdlog::trace("Allocating Cell object memory for object of size {}", sizeof(T));
		constexpr size_t needed = sizeof(T) + sizeof(GarbageCollected);
		static_assert(needed <= kBlockSizes.back(),
			"only object sizes <= 2048 bytes are currently supported");
		if constexpr (needed <= 16) {
			return m_blocks[0]->allocate();
		} else if constexpr (needed <= 32) {
			return m_blocks[1]->allocate();
		} else if constexpr (needed <= 64) {
			return m_blocks[2]->allocate();
		} else if constexpr (needed <= 128) {
			return m_blocks[3]->allocate();
		} else if constexpr (needed <= 256) {
			return m_blocks[4]->allocate();
		} else if constexpr (needed <= 512) {
			return m_blocks[5]->allocate();
		} else if constexpr (needed <= 1024) {
			return m_blocks[6]->allocate();
		} else {
			return m_blocks[7]->allocate();
		}
	}

	template<typename T>
	uint8_t *allocate(size_t extra_bytes)
		requires std::is_base_of_v<Cell, T>
	{
		spdlog::trace("Allocating Cell object memory for object of size {}", sizeof(T));
		const size_t needed = sizeof(T) + extra_bytes + sizeof(GarbageCollected);
		for (size_t i = 0; i < kBlockCount; ++i) {
			if (needed <= kBlockSizes[i]) { return m_blocks[i]->allocate(); }
		}
		TODO();
	}

	template<typename T> uint8_t *allocate()
	{
		[]<bool flag = false>() {
			static_assert(flag, "only GC collected objects are currently supported");
		}();
		return nullptr;
	}

	bool has_address(uint8_t *addr) const;

	void reset()
	{
		for (auto &b : m_blocks) { b->reset(); }
	}

  private:
	std::array<std::unique_ptr<Block>, kBlockCount> m_blocks;
};

class Heap
	: NonCopyable
	, NonMoveable
{
	friend class VirtualMachine;
	friend GarbageCollector;
	friend struct TestHeap;

	std::unique_ptr<uint8_t[]> m_static_memory;
	size_t m_static_memory_size{ 1 * MB };
	size_t m_static_offset{ 8 };
	Slab m_slab;
	std::unique_ptr<GarbageCollector> m_gc;
	std::unordered_map<uint8_t *, std::vector<py::PyObject *>> m_weakrefs;
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

	uintptr_t *start_sp() const { return m_bottom_stack_pointer; }

	template<typename T, typename... Args> T *__attribute__((noinline)) allocate(Args &&...args)
	{
		if (m_allocate_in_static) { return allocate_static<T>(std::forward<Args>(args)...); }
		collect_garbage();
		auto *ptr = m_slab.allocate<T>();

		uint8_t *obj_ptr = allocate_gc(ptr);
		T *obj = new (obj_ptr) T(std::forward<Args>(args)...);

		return obj;
	}

	template<typename T, typename... Args>
	T *__attribute__((noinline)) allocate_with_extra_bytes(size_t bytes, Args &&...args)
	{
		if (bytes == 0) { return allocate<T>(std::forward<Args>(args)...); }
		if (m_allocate_in_static) { TODO(); }
		collect_garbage();
		auto *ptr = m_slab.allocate<T>(bytes);

		uint8_t *obj_ptr = allocate_gc(ptr);
		T *obj = new (obj_ptr) T(std::forward<Args>(args)...);
		memset(obj_ptr + sizeof(T), 0, bytes);

		return obj;
	}

	template<typename T, typename TargetT, typename... Args>
	T *__attribute__((noinline)) allocate_weakref(TargetT &&target, Args &&...args)
	{
		T *obj = allocate<T>(std::forward<TargetT>(target), std::forward<Args>(args)...);
		if (obj) { m_weakrefs[bit_cast<uint8_t *>(target)].push_back(obj); }
		return obj;
	}

	void collect_garbage();

	GarbageCollector &garbage_collector()
	{
		ASSERT(m_gc);
		return *m_gc;
	}

	template<typename T, typename... Args> T *allocate_static(Args &&...args)
	{
		if (m_static_offset + sizeof(T) >= m_static_memory_size) { TODO(); }
		T *ptr = new (m_static_memory.get() + m_static_offset) T(std::forward<Args>(args)...);
		m_static_offset += sizeof(T);
		return ptr;
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

	bool has_weakref_object(uint8_t *obj) const { return m_weakrefs.contains(obj); }
	size_t weakref_count(uint8_t *obj) const
	{
		if (auto it = m_weakrefs.find(obj); it != m_weakrefs.end()) { return it->second.size(); }
		return 0;
	}
	std::vector<py::PyObject *> get_weakrefs(uint8_t *obj) const
	{
		if (auto it = m_weakrefs.find(obj); it != m_weakrefs.end()) { return it->second; }
		return {};
	}

	// Called from a weakref wrapper's destructor when the wrapper itself
	// (not the target) is being collected. Without this, m_weakrefs[target]
	// would retain a dangling pointer that getweakrefs/getweakrefcount
	// would later hand back to user code.
	void unregister_weakref(uint8_t *target, py::PyObject *wrapper)
	{
		auto it = m_weakrefs.find(target);
		if (it == m_weakrefs.end()) { return; }
		auto &wrappers = it->second;
		wrappers.erase(std::remove(wrappers.begin(), wrappers.end(), wrapper), wrappers.end());
		if (wrappers.empty()) { m_weakrefs.erase(it); }
	}

  private:
	uint8_t *allocate_gc(uint8_t *ptr) const;

	Heap();
};
