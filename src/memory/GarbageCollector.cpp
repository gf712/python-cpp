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
#include <unordered_set>

using namespace py;

MarkSweepGC::MarkSweepGC() : GarbageCollector() { set_frequency(10'000); }

namespace {

template<class To, class From>
__attribute__((no_sanitize_address))
typename std::enable_if_t<sizeof(To) == sizeof(From) && std::is_trivially_copyable_v<From>
							  && std::is_trivially_copyable_v<To>,
	To>
	bit_cast_without_sanitizer(const From &src) noexcept
{
	static_assert(std::is_trivially_constructible_v<To>,
		"This implementation additionally requires destination type to be trivially constructible");

	To dst;
	std::memcpy(&dst, &src, sizeof(To));
	return dst;
}

void add_root(GarbageCollected *obj_header, std::stack<Cell *> &roots)
{
	auto *cell = bit_cast<Cell *>(bit_cast<uint8_t *>(obj_header) + sizeof(GarbageCollected));

	ASSERT(!obj_header->black());

	if (obj_header->grey()) { return; }

	ASSERT(obj_header->white());

	if (cell->is_pyobject()) {
		auto *obj = static_cast<PyObject *>(cell);
		spdlog::debug("adding root {}@{}", obj->type()->name(), (void *)obj);
	}

	obj_header->mark(GarbageCollected::Color::GREY);
	roots.push(cell);
}


__attribute__((no_sanitize_address)) std::stack<Cell *> collect_roots_on_the_stack(const Heap &heap,
	uint8_t *stack_bottom)
{
	std::stack<Cell *> roots;

	// Spill all callee-saved registers into `jump_buffer`. setjmp captures
	// them as part of saving the calling environment.
	std::jmp_buf jump_buffer;
	setjmp(jump_buffer);

	auto scan_range = [&](uint8_t *begin, uint8_t *end) {
		for (uint8_t *p = begin; p < end; p += sizeof(uintptr_t)) {
			uint8_t *address =
				bit_cast_without_sanitizer<uint8_t *>(*bit_cast_without_sanitizer<uintptr_t *>(p))
				- sizeof(GarbageCollected);
			spdlog::trace("checking address {}, pointer address={}", (void *)address, (void *)p);
			if (heap.slab().has_address(address)) {
				spdlog::trace("valid address {}", (void *)address);
				auto *obj_header = bit_cast<GarbageCollected *>(address);
				add_root(obj_header, roots);
			}
		}
	};

	// jump_buffer is a local variable, so it sits below the current frame
	// pointer on stacks that grow downward. The frame-pointer-to-stack-bottom
	// scan below therefore skips it, which would lose any GC root whose only
	// live reference is in a callee-saved register at this point. Scan the
	// buffer's bytes explicitly to recover those spilled values.
	auto *jb_begin = bit_cast_without_sanitizer<uint8_t *>(&jump_buffer);
	scan_range(jb_begin, jb_begin + sizeof(jump_buffer));

	// Traverse the stack from the current frame pointer up to the recorded
	// stack bottom; this covers everything in our callers' frames.
	uint8_t *rsp = bit_cast_without_sanitizer<uint8_t *>(__builtin_frame_address(0));
	scan_range(rsp, stack_bottom);

	spdlog::debug("Done collecting roots from the stack, found {} roots", roots.size());
	return roots;
}

bool is_static_memory(uint8_t *cell_start, const Heap &heap)
{
	return bit_cast<uintptr_t>(cell_start) >= bit_cast<uintptr_t>(heap.static_memory())
		   && bit_cast<uintptr_t>(cell_start)
				  < bit_cast<uintptr_t>(heap.static_memory() + heap.static_memory_size());
}

}// namespace

void GarbageCollector::remove_weakref(Heap &heap, uint8_t *obj) const
{
	heap.m_weakrefs.erase(obj);
}

std::stack<Cell *> MarkSweepGC::collect_roots(const Heap &heap) const
{
	if (!m_stack_bottom) { m_stack_bottom = bit_cast<uint8_t *>(heap.start_sp()); }

	auto roots = collect_roots_on_the_stack(heap, m_stack_bottom);

	spdlog::trace("adding objects in VM stack to roots");
	for (const auto &s : VirtualMachine::the().stack_objects()) {
		for (const auto &val : s) {
			if (std::holds_alternative<PyObject *>(*val)) {
				auto *obj = std::get<PyObject *>(*val);
				if (obj) {
					if (!is_static_memory(bit_cast<uint8_t *>(obj), heap)) {
						auto *obj_header = bit_cast<GarbageCollected *>(
							bit_cast<uint8_t *>(obj) - sizeof(GarbageCollected));
						add_root(obj_header, roots);
					}
				}
			}
		}
	}

	if (VirtualMachine::the().has_interpreter()) {
		auto &interpreter = VirtualMachine::the().interpreter();
		struct AddRoot : Cell::Visitor
		{
			const Heap &heap_;
			std::stack<Cell *> &roots_;
			AddRoot(const Heap &heap, std::stack<Cell *> &roots_) : heap_(heap), roots_(roots_) {}
			void visit(Cell &cell)
			{
				auto *obj = static_cast<PyObject *>(&cell);
				if (obj) {
					if (!is_static_memory(bit_cast<uint8_t *>(obj), heap_)) {
						auto *obj_header = bit_cast<GarbageCollected *>(
							bit_cast<uint8_t *>(obj) - sizeof(GarbageCollected));
						add_root(obj_header, roots_);
					}
				}
			}
		} visitor{ heap, roots };
		interpreter.visit_graph(visitor);
	}
	return roots;
}

struct MarkGCVisitor : Cell::Visitor
{
	Heap &m_heap;
	std::stack<Cell *> &m_to_visit;

	// Collects the 1-hop neighbours of a given cell. visit_graph(visitor) on
	// the root invokes Visitor::visit() for each thing the root references
	// (which, by convention, may include the root itself for leaf cells); the
	// `m_collecting` gate isolates those nested calls so only direct
	// neighbours land in the vector and we do not recurse further.
	struct NeighbourVisitor : Cell::Visitor
	{
		std::vector<Cell *> m_neighbours;
		bool m_collecting{ false };

		void collect_neighbours_of(Cell &root)
		{
			m_neighbours.clear();
			m_collecting = true;
			root.visit_graph(*this);
			m_collecting = false;
		}

		void visit(Cell &cell) override
		{
			if (m_collecting) { m_neighbours.push_back(&cell); }
		}
	};

	MarkGCVisitor(Heap &heap, std::stack<Cell *> &to_visit) : m_heap(heap), m_to_visit(to_visit) {}

	void visit(Cell &cell)
	{
		uint8_t *cell_start = bit_cast<uint8_t *>(&cell);

		if (!is_static_memory(cell_start, m_heap)) {
			NeighbourVisitor nv{};
			nv.collect_neighbours_of(cell);
			auto &neighbours = nv.m_neighbours;
			spdlog::trace("node: {}", static_cast<void *>(&cell));
			for (auto *neighbour : neighbours) {
				if (is_static_memory(bit_cast<uint8_t *>(neighbour), m_heap)) { continue; }
				auto *obj_header = bit_cast<GarbageCollected *>(
					bit_cast<uint8_t *>(neighbour) - sizeof(GarbageCollected));

				// already visited
				if (obj_header->black()) {
					spdlog::trace("already visited @{}", static_cast<void *>(neighbour));
					continue;
				}

				// already on the 'to be visited' stack
				if (obj_header->grey()) {
					spdlog::trace(
						"already added @{} to the visited stack", static_cast<void *>(neighbour));
					continue;
				}

				if (neighbour->is_pyobject()) {
					auto *obj = static_cast<PyObject *>(neighbour);
					spdlog::trace("Adding PyObject to be visited stack {}@{}",
						obj->type_prototype().__name__,
						(void *)obj);
				}

				obj_header->mark(GarbageCollected::Color::GREY);
				m_to_visit.push(neighbour);
			}
		}
	}
};


void MarkSweepGC::mark_all_cell_unreachable(Heap &heap) const
{
	// NOTE: this site intentionally does NOT use Slab::for_each_block. Doing
	// so produces a stack layout (lambda captures across inlining) that
	// keeps a Block pointer alive in a slot the next collect_roots scan
	// reads as a heap root, regressing
	// TestHeap.GarbageCollectorDeallocatesGCPointersWhenStackFrameIsPopped.
	// The conservative-GC fragility this exposes is the underlying issue;
	// until it is replaced by a precise RootSet, keep the explicit array
	// here so the test stays green. The other sweep/has_address sites are
	// fine because they run after the scan, not before.
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


void MarkSweepGC::mark_all_live_objects(Heap &heap, std::stack<Cell *> &&roots) const
{
	auto mark_visitor = std::make_unique<MarkGCVisitor>(heap, roots);

	// mark all live objects
	while (!roots.empty()) {
		Cell *root = roots.top();
		roots.pop();

		if (is_static_memory(bit_cast<uint8_t *>(root), heap)) { continue; }

		spdlog::trace("Visiting root {}", (void *)root);

		auto *obj_header =
			bit_cast<GarbageCollected *>(bit_cast<uint8_t *>(root) - sizeof(GarbageCollected));
		ASSERT(obj_header->grey());
		obj_header->mark(GarbageCollected::Color::BLACK);
		mark_visitor->visit(*root);
	}
	spdlog::debug("Done marking all live objects");
}


void MarkSweepGC::sweep(Heap &heap) const
{
	spdlog::trace("MarkSweepGC::sweep start");

	heap.slab().for_each_block([this, &heap](Block &block) {
		for (auto &chunk : block.chunks()) {
			chunk.for_each_cell_alive([this, &chunk, &heap](uint8_t *memory) {
				auto *header = bit_cast<GarbageCollected *>(memory);
				if (header->white()) {
					auto *cell = bit_cast<Cell *>(memory + sizeof(GarbageCollected));
					if (cell->is_pyobject()) {
						auto *obj = static_cast<PyObject *>(cell);
						spdlog::debug("Deallocating {}@{}", obj->type()->name(), (void *)obj);
					}
					spdlog::debug("Calling destructor of object at {}", (void *)cell);
					this->remove_weakref(heap, bit_cast<uint8_t *>(cell));
					cell->~Cell();
					chunk.deallocate(memory);
					new (header) GarbageCollected();
				}
			});
		}
	});
	spdlog::trace("MarkSweepGC::sweep done");
}


void MarkSweepGC::mark_and_sweep(Heap &heap)
{
	mark_all_cell_unreachable(heap);
	auto roots = collect_roots(heap);
	mark_all_live_objects(heap, std::move(roots));
	sweep(heap);
	m_iterations_since_last_sweep = 0;
}

void MarkSweepGC::run(Heap &heap)
{
	if (m_pause) { return; }
	if (++m_iterations_since_last_sweep < m_frequency) { return; }
	mark_and_sweep(heap);
}

void MarkSweepGC::force_run(Heap &heap) { mark_and_sweep(heap); }

void MarkSweepGC::resume()
{
	ASSERT(!is_active());
	m_pause = false;
}

void MarkSweepGC::pause()
{
	ASSERT(is_active());
	m_pause = true;
}

bool MarkSweepGC::is_active() const { return !m_pause; }
