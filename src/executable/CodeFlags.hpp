#pragma once
#include <bitset>
#include <cstdint>


class CodeFlags
{
  public:
	enum class Flag {
		OPTIMIZED = 0,
		NEWLOCALS = 1,
		VARARGS = 2,
		VARKEYWORDS = 3,
		NESTED = 4,
		GENERATOR = 5,
		COROUTINE = 6,
		CLASS = 7,
	};

  private:
	std::bitset<8> m_flags;

	CodeFlags() = default;

  public:
	template<typename... Args>
	// requires std::conjunction_v<std::is_same<Flag, Args>...>
	static CodeFlags create(Args... args)
	{
		CodeFlags f;
		(f.m_flags.set(static_cast<std::uint8_t>(args)), ...);
		return f;
	}

	static CodeFlags from_byte(std::uint8_t b)
	{
		auto f = CodeFlags();
		f.m_flags = std::bitset<8>(b);
		return f;
	}

	void set(Flag f) { m_flags.set(static_cast<std::uint8_t>(f)); }
	void reset(Flag f) { m_flags.reset(static_cast<std::uint8_t>(f)); }
	bool is_set(Flag f) const { return m_flags[static_cast<std::uint8_t>(f)]; }
	std::bitset<8> bits() const { return m_flags; }
};
