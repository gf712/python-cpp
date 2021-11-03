#pragma once

#include <bitset>
#include "utilities.hpp"

#include <unordered_set>

class GarbageCollected
{
  public:
	enum class Color {
		WHITE,
		GREY,
		BLACK,
	};

	virtual ~GarbageCollected() = default;
	// void *operator new(size_t) = delete;
	// void *operator new[](size_t) = delete;

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
	: public GarbageCollected
	, NonCopyable
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
};

class MarkSweepGC : public GarbageCollector
{
  public:
	MarkSweepGC();
	void run(Heap &) const override;

	std::unordered_set<Cell *> collect_roots() const;

  private:
	mutable uint8_t *m_bottom_top{ nullptr };
	size_t m_frequency;
	mutable size_t m_iterations_since_last_sweep{ 0 };
};
