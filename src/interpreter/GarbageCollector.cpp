#include "GarbageCollector.hpp"
#include "bytecode/Heap.hpp"

#include "bytecode/VM.hpp"
#include "Interpreter.hpp"
#include "runtime/PyDict.hpp"

std::vector<Cell *> MarkSweepGC::collect_roots() const
{
	// TODO();
	return std::vector<Cell *>{};
}

struct MarkGCVisitor : Cell::Visitor
{
	void visit(Cell &cell)
	{
		uint8_t *cell_start = static_cast<uint8_t *>(static_cast<void *>(&cell));
		auto *obj_header = static_cast<GarbageCollected<Cell> *>(
			static_cast<void *>(cell_start - sizeof(GarbageCollected<Cell>)));

		if (obj_header->black()) {
			spdlog::debug("Already visited {}, skipping", cell.to_string());
			return;
		}
		spdlog::debug("Visiting: {} ({})", cell.to_string(), (void *)&cell);

		obj_header->mark(GarbageCollected<Cell>::Color::BLACK);
		cell.visit_graph(*this);
	}

  private:
};

void MarkSweepGC::run(Heap &heap) const
{
	// collect roots
	const auto roots = collect_roots();

	auto mark_visitor = std::make_unique<MarkGCVisitor>();

	// mark all alive objects
	for (auto *root : roots) {
		spdlog::debug("Visiting root {}", root->to_string());
		root->visit_graph(*mark_visitor);
	}

	// sweep all the dead objects
	const auto &block = heap.slab().block();
	for (auto &chunk : block->chunks()) {
		chunk.for_each_cell_alive([&chunk](uint8_t *memory) {
			auto *header = static_cast<GarbageCollected<Cell> *>(static_cast<void *>(memory));
			if (header->white()) {
				// pretty sure this logging can throw foobar
				spdlog::debug("Deallocating {}",
					static_cast<Cell *>(
						static_cast<void *>(memory + sizeof(GarbageCollected<Cell>)))
						->to_string());
				chunk.deallocate(memory);
			}
		});
	}
}
