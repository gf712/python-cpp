#pragma once

#include "utilities.hpp"

#include <memory>
#include <bitset>

static constexpr size_t KB = 1024;
static constexpr size_t MB = 1024 * KB;


class Block
{
	class Chunk
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
				spdlog::debug("chunk bit mask: {}", m_occupied_chunks.to_string());
				if (auto chunk_idx = next_free_chunk()) {
					ASSERT(!m_occupied_chunks[*chunk_idx])
					spdlog::debug("marking next free chunk -> old chunk bit mask: {}",
						m_occupied_chunks.to_string());
					m_occupied_chunks.flip(*chunk_idx);
					spdlog::debug("marking next free chunk -> new chunk bit mask: {}",
						m_occupied_chunks.to_string());
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

		uint8_t *allocate()
		{
			if (auto chunk_idx = m_chunk_view.mark_next_free_chunk()) {
				spdlog::debug("Allocating memory at index {}, address {}",
					*chunk_idx,
					(void *)(m_memory + *chunk_idx * m_object_size));
				return m_memory + *chunk_idx * m_object_size;
			} else {
				return nullptr;
			}
		}

		void deallocate(uint8_t *ptr)
		{
			uintptr_t start = reinterpret_cast<uintptr_t>(m_memory);
			uintptr_t end =
				reinterpret_cast<uintptr_t>(m_memory + m_object_size * ChunkView<>::ChunkCount);
			ASSERT(reinterpret_cast<uintptr_t>(ptr) >= start
				   && reinterpret_cast<uintptr_t>(ptr) < end);
			size_t ptr_idx = (reinterpret_cast<uintptr_t>(ptr) - start) / m_object_size;
			ASSERT(ptr_idx < ChunkView<>::ChunkCount)
			m_chunk_view.mark_chunk_as_free(ptr_idx);
		}

		void set_all_in_mask(bool value)
		{
			if (value) {
				m_chunk_view.set();
			} else {
				m_chunk_view.reset();
			}
		}

		void reset() { m_chunk_view.reset(); }

		size_t object_size() const { return m_object_size; }

	  private:
		uint8_t *m_memory;
		size_t m_object_size;
		ChunkView<> m_chunk_view;
	};


  public:
	Block(size_t object_size, size_t capacity);

	void reset();

	uint8_t *allocate();

	void deallocate(uint8_t *ptr);

  private:
	std::vector<Chunk> m_chunks;
	std::vector<std::unique_ptr<uint8_t>> m_memory;
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
	}

	template<typename T> uint8_t *allocate()
	{
		spdlog::debug("Allocating memory for object of size {}", sizeof(T));
		if constexpr (sizeof(T) <= 16) { return block16->allocate(); }
		if constexpr (sizeof(T) <= 32) { return block32->allocate(); }
		if constexpr (sizeof(T) <= 64) { return block64->allocate(); }
		if constexpr (sizeof(T) <= 128) { return block128->allocate(); }
		if constexpr (sizeof(T) <= 256) { return block256->allocate(); }
		if constexpr (sizeof(T) <= 512) {
			return block512->allocate();
		} else {
			[]<bool flag = false>()
			{
				static_assert(flag, "only object sizes <= 512 bytes are currently supported");
			}
			();
		}
		return nullptr;
	}

	template<typename T> void deallocate(uint8_t *ptr)
	{
		spdlog::debug("Deallocating memory for object of size {}", sizeof(T));
		if constexpr (sizeof(T) <= 16) { return block16->deallocate(ptr); }
		if constexpr (sizeof(T) <= 32) { return block32->deallocate(ptr); }
		if constexpr (sizeof(T) <= 64) { return block64->deallocate(ptr); }
		if constexpr (sizeof(T) <= 128) { return block128->deallocate(ptr); }
		if constexpr (sizeof(T) <= 256) { return block256->deallocate(ptr); }
		if constexpr (sizeof(T) <= 512) {
			return block512->deallocate(ptr);
		} else {
			[]<bool flag = false>()
			{
				static_assert(flag, "only object sizes <= 512 bytes are currently supported");
			}
			();
		}
	}

	void reset()
	{
		block16->reset();
		block32->reset();
		block64->reset();
		block128->reset();
		block256->reset();
		block512->reset();
	}

  private:
	std::unique_ptr<Block> block16{ nullptr };
	std::unique_ptr<Block> block32{ nullptr };
	std::unique_ptr<Block> block64{ nullptr };
	std::unique_ptr<Block> block128{ nullptr };
	std::unique_ptr<Block> block256{ nullptr };
	std::unique_ptr<Block> block512{ nullptr };
};

class Heap
	: NonCopyable
	, NonMoveable
{
	uint8_t *m_static_memory;
	size_t m_static_memory_size{ 4 * KB };
	size_t m_static_offset{ 0 };
	Slab m_slab;

  public:
	static Heap &the()
	{
		static auto heap = Heap();
		return heap;
	}

	~Heap()
	{
		free(m_static_memory);
		m_static_memory = nullptr;
	}

	void reset() { m_slab.reset(); }

	template<typename T, typename... Args> std::shared_ptr<T> allocate(Args &&... args)
	{
		uint8_t *memory = m_slab.allocate<T>();
		T *ptr = new (memory) T(std::forward<Args>(args)...);
		return std::shared_ptr<T>(ptr, [](T *ptr) {
			(void)ptr;
			// this->m_slab.deallocate<T>(static_cast<uint8_t *>(static_cast<void *>(ptr)));
		});
	}

	template<typename T, typename... Args> std::shared_ptr<T> allocate_static(Args &&... args)
	{
		if (m_static_offset + sizeof(T) >= m_static_memory_size) { TODO(); }
		T *ptr = new (m_static_memory + m_static_offset) T(std::forward<Args>(args)...);
		m_static_offset += sizeof(T);
		return std::shared_ptr<T>(ptr, [](T *) { return; });
	}

  private:
	Heap() { m_static_memory = static_cast<uint8_t *>(malloc(m_static_memory_size)); }
};