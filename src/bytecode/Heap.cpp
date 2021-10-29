#include "Heap.hpp"


Block::Block(size_t object_size, size_t capacity)
{
	size_t chunks_needed = std::round(capacity / 64);
	spdlog::debug(
		"Initialising a block with {} chunks, each managing memory for 64 objects of size {}",
		chunks_needed,
		object_size);

	auto &mem = m_memory.emplace_back(new uint8_t[chunks_needed * 64 * object_size]);
	spdlog::debug(
		"Allocated {} bytes at address {}", chunks_needed * 64 * object_size, (void *)mem.get());

	size_t idx{ 0 };
	while (chunks_needed--) {
		m_chunks.emplace_back(mem.get() + idx * (object_size * 64), object_size);
		idx++;
	}

	// TODO: this should only be needed for debug builds
	memset(mem.get(), 0xCD, m_chunks.size() * object_size * 64);
}


void Block::reset()
{
	for (auto &chunk : m_chunks) { chunk.reset(); }
}

uint8_t *Block::allocate()
{
	for (auto &chunk : m_chunks) {
		if (auto *ptr = chunk.allocate()) { return ptr; }
	}

	spdlog::debug("Need to allocate more chunks");

	// add more chunks -> new chunk count is old count multiplied by golden ration (1.618)
	size_t old_chunk_count = m_chunks.size();
	size_t new_chunk_count = std::round(static_cast<int32_t>(m_chunks.size() * 1.618));
	size_t new_chunks_to_allocate = new_chunk_count - old_chunk_count;
	size_t idx{ 0 };
	const size_t object_size = m_chunks.back().object_size();
	const size_t new_memory_size = new_chunk_count * 64 * object_size;

	auto &new_memory = m_memory.emplace_back(new uint8_t[new_memory_size]);

	while (new_chunks_to_allocate--) {
		m_chunks.emplace_back(new_memory.get() + idx * (object_size * 64), object_size);
		idx++;
	}

	if (auto *ptr = m_chunks[old_chunk_count].allocate()) {
		return ptr;
	} else {
		spdlog::warn("Failed to allocate in new chunk {}/{}", old_chunk_count, m_chunks.size());
		// TODO: handle this more gracefully
		std::abort();
	}
}

void Block::deallocate(uint8_t *ptr)
{
	for (auto &mem : m_memory) {
		const size_t object_size = m_chunks.back().object_size();
		const size_t current_size = m_chunks.size() * 64 * object_size;
		uintptr_t start = reinterpret_cast<uintptr_t>(mem.get());
		uintptr_t end = reinterpret_cast<uintptr_t>(mem.get() + current_size);
		// if ptr not in this piece of memory move to next one
		if (reinterpret_cast<uintptr_t>(ptr) < start || reinterpret_cast<uintptr_t>(ptr) >= end) {
			continue;
		}

		const size_t chunk_idx = (reinterpret_cast<uintptr_t>(ptr) - start) / (64 * object_size);
		ASSERT(chunk_idx < m_chunks.size())

		m_chunks[chunk_idx].deallocate(ptr);

		// TODO: this should only be needed for debug builds
		memset(ptr, 0xDD, object_size);
		return;
	}
	spdlog::error("Failed to find memory piece of ptr {}", (void *)ptr);
	std::abort();
}

void Heap::collect_garbage()
{
	// collect_roots();

	// mark_live_objects();

	// sweep_dead_objects();
}
