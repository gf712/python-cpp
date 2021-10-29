#pragma once

#include <bitset>
#include "utilities.hpp"

template<typename T> class GarbageCollected
{
  public:
	enum class Color {
		WHITE,
		GREY,
		BLACK,
	};

	// void *operator new(size_t) = delete;
	// void *operator new[](size_t) = delete;

	GarbageCollected(T &obj) : m_object(obj) {}

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
	T &m_object;

	GarbageCollected() = delete;
};

class Cell
	: public GarbageCollected<Cell>
	, NonCopyable
	, NonMoveable
{
  public:
	struct Visitor
	{
		virtual ~Visitor() {}
		virtual void visit(Cell &) = 0;
	};

  public:
	Cell() : GarbageCollected<Cell>(*this) {}

	virtual std::string to_string() const = 0;
	virtual void visit_graph(Visitor &) = 0;
};


class Heap;

class GarbageCollector
{
  public:
	virtual void run(Heap &) const = 0;
};

class MarkSweepGC : GarbageCollector
{
  public:
	void run(Heap &) const override;

	std::vector<Cell *> collect_roots() const;
};
