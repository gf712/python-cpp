#include "GarbageCollector.hpp"
#include "interpreter/Interpreter.hpp"
#include "interpreter/InterpreterSession.hpp"
#include "memory/Heap.hpp"
#include "runtime/PyCode.hpp"
#include "runtime/PyDict.hpp"
#include "runtime/PyFrame.hpp"
#include "runtime/PyModule.hpp"
#include "runtime/PyObject.hpp"
#include "runtime/PyType.hpp"
#include "vm/VM.hpp"

#include <csetjmp>

using namespace py;

MarkSweepGC::MarkSweepGC() : GarbageCollector() { set_frequency(10'000); }

namespace {
template<class To, class From>
__attribute__((no_sanitize_address)) typename std::enable_if_t<
	sizeof(To) == sizeof(From)
		&& std::is_trivially_copyable_v<From> && std::is_trivially_copyable_v<To>,
	To>
	bit_cast_without_sanitizer(const From &src) noexcept
{
	static_assert(std::is_trivially_constructible_v<To>,
		"This implementation additionally requires destination type to be trivially constructible");

	To dst;
	std::memcpy(&dst, &src, sizeof(To));
	return dst;
}

__attribute__((no_sanitize_address)) std::unordered_set<Cell *>
	collect_roots_on_the_stack(Heap &heap, uint8_t *stack_bottom)
{
	std::unordered_set<Cell *> roots;

	// push register values onto the stack
	std::jmp_buf jump_buffer;
	setjmp(jump_buffer);

	// traverse the stack from the stack pointer to the stack origin/bottom
	// uintptr_t *rsp_;
	// asm volatile("movq %%rsp, %0" : "=r"(rsp_));

	// uint8_t *rsp = bit_cast<uint8_t *>(rsp_);
	uint8_t *rsp = bit_cast_without_sanitizer<uint8_t*>(__builtin_frame_address(0));
	for (; rsp < stack_bottom; rsp += sizeof(uintptr_t)) {
		uint8_t *address =
			bit_cast_without_sanitizer<uint8_t *>(*bit_cast_without_sanitizer<uintptr_t *>(rsp))
			- sizeof(GarbageCollected);
		spdlog::trace("checking address {}, pointer address={}", (void *)address, (void *)rsp);
		if (heap.slab().has_address(address)) {
			spdlog::trace("valid address {}", (void *)address);
			auto *cell = bit_cast<Cell *>(address + sizeof(GarbageCollected));
			if (cell->is_pyobject()) {
				auto *obj = static_cast<PyObject *>(cell);
				spdlog::debug("adding root {}@{}", obj->type()->name(), (void *)obj);
			}
			roots.insert(cell);
		}
	}
	return roots;
}

}// namespace

std::unordered_set<Cell *> MarkSweepGC::collect_roots(Heap &heap) const
{
	if (!m_stack_bottom) { m_stack_bottom = bit_cast<uint8_t *>(heap.start_stack_pointer()); }

	auto roots = collect_roots_on_the_stack(heap, m_stack_bottom);

	spdlog::trace("adding objects in VM stack to roots");
	for (const auto &s : VirtualMachine::the().stack_objects()) {
		for (const auto &val : s) {
			if (std::holds_alternative<PyObject *>(*val)) {
				auto *obj = std::get<PyObject *>(*val);
				if (obj) {
					const auto [_, inserted] = roots.insert(obj);
					if (inserted) {
						spdlog::debug("adding root {}@{}", obj->type()->name(), (void *)obj);
					}
				}
			}
		}
	}

	if (VirtualMachine::the().has_interpreter()) {
		auto &interpreter = VirtualMachine::the().interpreter();
		struct AddRoot : Cell::Visitor
		{
			std::unordered_set<Cell *> &roots_;
			AddRoot(std::unordered_set<Cell *> &roots) : roots_(roots) {}
			void visit(Cell &cell)
			{
				const auto [_, inserted] = roots_.insert(&cell);
				if (inserted) { spdlog::debug("adding root {}", (void *)&cell); }
			}
		} visitor{ roots };
		interpreter.visit_graph(visitor);
	}
	return roots;
}

struct MarkGCVisitor : Cell::Visitor
{
	Heap &m_heap;
	std::unordered_set<Cell *> m_visited;

	MarkGCVisitor(Heap &heap) : m_heap(heap) {}

	void visit(Cell &cell)
	{
		if (m_visited.contains(&cell)) return;
		m_visited.insert(&cell);

		uint8_t *cell_start = bit_cast<uint8_t *>(&cell);

		const bool static_memory =
			bit_cast<uintptr_t>(cell_start) >= bit_cast<uintptr_t>(m_heap.static_memory())
			&& bit_cast<uintptr_t>(cell_start)
				   < bit_cast<uintptr_t>(m_heap.static_memory() + m_heap.static_memory_size());

		if (!static_memory) {
			auto *obj_header = bit_cast<GarbageCollected *>(cell_start - sizeof(GarbageCollected));

			if (obj_header->black()) {
				if (cell.is_pyobject()) {
					spdlog::trace("Already visited {}@{}, skipping",
						static_cast<PyObject *>(&cell)->type_prototype().__name__,
						(void *)&cell);
				}
				return;
			}
			if (cell.is_pyobject()) {
				auto *obj = static_cast<PyObject *>(&cell);
				spdlog::trace("Visiting {}@{}", obj->type_prototype().__name__, (void *)obj);
			}
			obj_header->mark(GarbageCollected::Color::BLACK);
		}

		cell.visit_graph(*this);
	}
};


void MarkSweepGC::mark_all_cell_unreachable(Heap &heap) const
{
	// TODO: once the ideal block sizes are fixed there should be an iterator
	//       returning a list of all blocks
	std::array blocks = {
		std::reference_wrapper{ heap.slab().block_16() },
		std::reference_wrapper{ heap.slab().block_32() },
		std::reference_wrapper{ heap.slab().block_64() },
		std::reference_wrapper{ heap.slab().block_128() },
		std::reference_wrapper{ heap.slab().block_256() },
		std::reference_wrapper{ heap.slab().block_512() },
		std::reference_wrapper{ heap.slab().block_1024() },
		std::reference_wrapper{ heap.slab().block_2048() },
	};
	for (const auto &block : blocks) {
		for (auto &chunk : block.get()->chunks()) {
			chunk.for_each_cell([](uint8_t *memory) {
				auto *header = static_cast<GarbageCollected *>(static_cast<void *>(memory));
				header->mark(GarbageCollected::Color::WHITE);
			});
		}
	}
}


void MarkSweepGC::mark_all_live_objects(Heap &heap, const std::unordered_set<Cell *> &roots) const
{
	auto mark_visitor = std::make_unique<MarkGCVisitor>(heap);

	// mark all live objects
	for (auto *root : roots) {
		spdlog::trace("Visiting root {}", (void *)root);
		root->visit_graph(*mark_visitor);
	}
}


void MarkSweepGC::sweep(Heap &heap) const
{
	spdlog::trace("MarkSweepGC::sweep start");

	// TODO: once the ideal block sizes are fixed there should be an iterator
	//       returning a list of all blocks
	std::array blocks = {
		std::reference_wrapper{ heap.slab().block_16() },
		std::reference_wrapper{ heap.slab().block_32() },
		std::reference_wrapper{ heap.slab().block_64() },
		std::reference_wrapper{ heap.slab().block_128() },
		std::reference_wrapper{ heap.slab().block_256() },
		std::reference_wrapper{ heap.slab().block_512() },
		std::reference_wrapper{ heap.slab().block_1024() },
		std::reference_wrapper{ heap.slab().block_2048() },
	};
	// sweep all the dead objects
	for (const auto &block : blocks) {
		for (auto &chunk : block.get()->chunks()) {
			chunk.for_each_cell_alive([&chunk](uint8_t *memory) {
				auto *header = bit_cast<GarbageCollected *>(memory);
				if (header->white()) {
					auto *cell = bit_cast<Cell *>(memory + sizeof(GarbageCollected));
					if (cell->is_pyobject()) {
						auto *obj = static_cast<PyObject *>(cell);
						spdlog::debug("Deallocating {}@{}", obj->type()->name(), (void *)obj);
					}
					spdlog::debug("Calling destructor of object at {}", (void *)cell);
					cell->~Cell();
					chunk.deallocate(memory);
				}
			});
		}
	}
	spdlog::trace("MarkSweepGC::sweep done");
}


void MarkSweepGC::run(Heap &heap) const
{
	if (m_pause) { return; }
	if (++m_iterations_since_last_sweep < m_frequency) { return; }

	mark_all_cell_unreachable(heap);

	const auto roots = collect_roots(heap);

	mark_all_live_objects(heap, roots);

	sweep(heap);

	m_iterations_since_last_sweep = 0;
}

void MarkSweepGC::resume()
{
	ASSERT(m_pause)
	m_pause = false;
}

void MarkSweepGC::pause()
{
	ASSERT(!m_pause)
	m_pause = true;
}