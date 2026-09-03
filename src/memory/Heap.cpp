module;
#include "core.hpp"
#include "memory/allocate.hpp"
#include <cstring>

module py.memory;
import py.runtime;
import std;

using namespace py;

Block::Chunk::~Chunk()
{
	for_each_cell_alive([](std::uint8_t *memory) {
		auto *cell = std::bit_cast<Cell *>(memory + sizeof(GarbageCollected));
		::detail::log_debug(
			std::format("Calling destructor of object at {}", (void *)cell).c_str());
		if (cell->is_pyobject()) {
			auto *obj = static_cast<PyObject *>(cell);
			::detail::log_debug(
				std::format("Deallocating {}@{}", obj->type()->name(), (void *)obj).c_str());
		}
		cell->~Cell();
	});
}

void Block::Chunk::reset()
{
	// before resetting make sure we are calling all the destructors
	for_each_cell_alive([](std::uint8_t *memory) {
		auto *cell = std::bit_cast<Cell *>(memory + sizeof(GarbageCollected));
		::detail::log_trace(
			std::format("Calling destructor of object at {}", (void *)cell).c_str());
		if (cell->is_pyobject()) {
			auto *obj = static_cast<PyObject *>(cell);
			::detail::log_trace(
				std::format("Deallocating {}@{}", obj->type()->name(), (void *)obj).c_str());
		}
		cell->~Cell();
	});
	m_chunk_view.reset();
}

bool Block::Chunk::has_address(std::uint8_t *memory) const
{
	auto address = std::bit_cast<std::uintptr_t>(memory);
	std::uintptr_t start = std::bit_cast<std::uintptr_t>(m_memory);
	std::uintptr_t end =
		std::bit_cast<std::uintptr_t>(m_memory + m_object_size * ChunkView<>::ChunkCount);

	if (address < start || address >= end) { return false; }

	if ((address - start) % m_object_size == 0) {
		return m_chunk_view.m_occupied_chunks[(address - start) / m_object_size];
	} else {
		return false;
	}
}

Block::Block(std::size_t object_size, std::size_t capacity)
{
	std::size_t chunks_needed = capacity / 64;
	::detail::log_debug(std::format(
		"Initialising a block with {} chunks, each managing memory for 64 objects of size {}",
		chunks_needed,
		object_size)
			.c_str());

	auto &mem =
		m_memory.emplace_back(std::make_unique<std::uint8_t[]>(chunks_needed * 64 * object_size));
	::detail::log_debug(std::format(
		"Allocated {} bytes at address {}", chunks_needed * 64 * object_size, (void *)mem.get())
			.c_str());

	std::size_t idx{ 0 };
	while (chunks_needed--) {
		m_chunks.emplace_back(mem.get() + idx * (object_size * 64), object_size);
		idx++;
	}

	// TODO: this should only be needed for debug builds
	std::memset(mem.get(), 0xCD, m_chunks.size() * object_size * 64);
}


void Block::reset()
{
	for (auto &chunk : m_chunks) { chunk.reset(); }
}

std::uint8_t *Block::allocate()
{
	for (std::size_t idx = 0; auto &chunk : m_chunks) {
		if (auto *ptr = chunk.allocate()) {
			::detail::log_trace(
				std::format("Allocated pointer in chunk {} (block size={})", idx, object_size())
					.c_str());
			return ptr;
		}
		++idx;
	}

	::detail::log_debug(std::format("Need to allocate more chunks").c_str());

	// add more chunks -> new chunk count is old count multiplied by golden ration (1.618)
	std::size_t old_chunk_count = m_chunks.size();
	std::size_t new_chunk_count =
		static_cast<std::size_t>(std::round(static_cast<float>(m_chunks.size()) * 1.618f));
	std::size_t new_chunks_to_allocate = new_chunk_count - old_chunk_count;
	std::size_t idx{ 0 };
	const std::size_t object_size = m_chunks.back().object_size();
	const std::size_t new_memory_size = new_chunk_count * 64 * object_size;

	auto &new_memory = m_memory.emplace_back(new std::uint8_t[new_memory_size]);

	while (new_chunks_to_allocate--) {
		m_chunks.emplace_back(new_memory.get() + idx * (object_size * 64), object_size);
		idx++;
	}

	if (auto *ptr = m_chunks[old_chunk_count].allocate()) {
		return ptr;
	} else {
		::detail::log_error(
			std::format("Failed to allocate in new chunk {}/{}", old_chunk_count, m_chunks.size())
				.c_str());
		// TODO: handle this more gracefully
		std::abort();
	}
}

void Block::deallocate(std::uint8_t *ptr)
{
	for (auto &mem : m_memory) {
		const std::size_t object_size = m_chunks.back().object_size();
		const std::size_t current_size = m_chunks.size() * 64 * object_size;
		std::uintptr_t start = std::bit_cast<std::uintptr_t>(mem.get());
		std::uintptr_t end = std::bit_cast<std::uintptr_t>(mem.get() + current_size);
		// if ptr not in this piece of memory move to next one
		if (std::bit_cast<std::uintptr_t>(ptr) < start
			|| std::bit_cast<std::uintptr_t>(ptr) >= end) {
			continue;
		}

		const std::size_t chunk_idx =
			(std::bit_cast<std::uintptr_t>(ptr) - start) / (64 * object_size);
		ASSERT(chunk_idx < m_chunks.size());

		m_chunks[chunk_idx].deallocate(ptr);

		return;
	}
	::detail::log_error(std::format("Failed to find memory piece of ptr {}", (void *)ptr).c_str());
	std::abort();
}


bool Slab::has_address(std::uint8_t *address) const
{
	bool found = false;
	for_each_block([&](const Block &block) {
		if (found) { return; }
		for (const auto &chunk : block.chunks()) {
			if (chunk.has_address(address)) {
				found = true;
				return;
			}
		}
	});
	return found;
}

Heap::Heap()
{
	m_static_memory = std::make_unique<std::uint8_t[]>(m_static_memory_size);
	m_gc = std::make_unique<MarkSweepGC>();
}

void Heap::collect_garbage()
{
	if (m_gc) m_gc->run(*this);
}

std::uint8_t *Heap::allocate_gc(std::uint8_t *ptr) const
{
	new (ptr) GarbageCollected();
	return ptr + sizeof(GarbageCollected);
}
