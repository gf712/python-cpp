#include "Heap.hpp"
#include "GarbageCollector.hpp"

bool Block::Chunk::has_address(uint8_t *memory) const
{
	auto address = bit_cast<uintptr_t>(memory);
	uintptr_t start = bit_cast<uintptr_t>(m_memory);
	uintptr_t end = bit_cast<uintptr_t>(m_memory + (m_object_size + 1) * ChunkView<>::ChunkCount);

	if (address < start || address > end) { return false; }

	if ((address - start) % m_object_size == 0) {
		return m_chunk_view.m_occupied_chunks[(address - start) / m_object_size];
	} else {
		return false;
	}
}

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
		uintptr_t start = bit_cast<uintptr_t>(mem.get());
		uintptr_t end = bit_cast<uintptr_t>(mem.get() + current_size);
		// if ptr not in this piece of memory move to next one
		if (bit_cast<uintptr_t>(ptr) < start || bit_cast<uintptr_t>(ptr) >= end) { continue; }

		const size_t chunk_idx = (bit_cast<uintptr_t>(ptr) - start) / (64 * object_size);
		ASSERT(chunk_idx < m_chunks.size())

		m_chunks[chunk_idx].deallocate(ptr);

		// TODO: this should only be needed for debug builds
		memset(ptr, 0xDD, object_size);
		return;
	}
	spdlog::error("Failed to find memory piece of ptr {}", (void *)ptr);
	std::abort();
}


bool Slab::has_address(uint8_t *address) const
{
	{
		auto &block = block512;
		for (auto &chunk : block->chunks()) {
			if (chunk.has_address(address)) { return true; }
		}
	}
	{
		auto &block = block1024;
		for (auto &chunk : block->chunks()) {
			if (chunk.has_address(address)) { return true; }
		}
	}
	return false;
}

Heap::Heap()
{
	m_static_memory = static_cast<uint8_t *>(malloc(m_static_memory_size));
	m_gc = std::make_unique<MarkSweepGC>();
}

void Heap::collect_garbage()
{
	if (m_gc) m_gc->run(*this);
}


uint8_t *Heap::allocate_gc(uint8_t *ptr) const
{
	new (ptr) GarbageCollected();
	return ptr + sizeof(GarbageCollected);
}
