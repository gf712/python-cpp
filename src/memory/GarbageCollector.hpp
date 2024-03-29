#pragma once

#include "utilities.hpp"

#include <bitset>
#include <stack>

class GarbageCollected
{
  public:
	enum class Color {
		WHITE,
		GREY,
		BLACK,
	};

	bool black() const { return m_state.all(); }

	bool grey() const { return m_state[1] == 1; }

	bool white() const { return m_state.none(); }

	void mark(Color color)
	{
		if (color == Color::WHITE) {
			m_state.reset();
		} else if (color == Color::GREY) {
			m_state[0] = 0;
			m_state[1] = 1;
		} else {
			m_state.set();
		}
	}

  private:
	std::bitset<2> m_state{ 0b00 };
};

class Cell
	: NonCopyable
	, NonMoveable
{
  public:
	struct Visitor
	{
		virtual ~Visitor() = default;
		virtual void visit(Cell &) = 0;
	};

  public:
	Cell() = default;
	virtual ~Cell() = default;

	virtual std::string to_string() const = 0;
	virtual void visit_graph(Visitor &) = 0;

	virtual bool is_pyobject() const { return false; }
};


class Heap;

class GarbageCollector
{
  public:
	virtual ~GarbageCollector() = default;
	virtual void run(Heap &) const = 0;
	virtual void resume() = 0;
	virtual void pause() = 0;
	virtual bool is_active() const = 0;

	void remove_weakref(Heap &heap, uint8_t *obj) const;

	void set_frequency(size_t new_frequency) { m_frequency = new_frequency; }

  protected:
	size_t m_frequency;
};

class MarkSweepGC : public GarbageCollector
{
  public:
	MarkSweepGC();
	void run(Heap &) const override;
	void resume() override;
	void pause() override;
	bool is_active() const override;

	std::stack<Cell *> collect_roots(const Heap &) const;
	void mark_all_cell_unreachable(Heap &) const;
	void mark_all_live_objects(Heap &, std::stack<Cell *> &&) const;
	void sweep(Heap &heap) const;

  private:
	mutable uint8_t *m_stack_bottom{ nullptr };
	mutable size_t m_iterations_since_last_sweep{ 0 };
	bool m_pause{ false };
};
