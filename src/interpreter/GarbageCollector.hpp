#pragma once

#include <bitset>

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

	GarbageCollected<T> *gc() { return this; }

	bool black() const { return m_state.all(); }

	bool grey() const { return m_state.count() == 1; }

	bool white() const { return m_state.none(); }

	void mark(Color color)
	{
		if (color == Color::WHITE) {
			m_state.reset();
		} else if (color == Color::GREY) {
			m_state.reset();
			m_state[1] = 1;
		} else {
			m_state.set();
		}
	}

  private:
	std::bitset<2> m_state;
	T &m_object;

	GarbageCollected() = delete;
};

class Cell
	: GarbageCollected<Cell>
	, NonCopyable
	, NonMoveable
{
  public:
	Cell() : GarbageCollected<Cell>(*this) {}

  protected:
	struct Visitor
	{
		virtual void visit(Cell &) = 0;
	};
	virtual void visit_graph(Visitor &) = 0;
};


class Heap;

class GarbageCollector
{
	virtual void run(Heap &) const = 0;
};

class MarkSweepGC : GarbageCollector
{
	void run(Heap &) const override;
};
