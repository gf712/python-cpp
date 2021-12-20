#pragma once

#include "utilities.hpp"

#include <optional>
#include <string>

class Label
	: NonCopyable
	, NonMoveable
{
	std::string m_label_name;
	size_t m_function_id;
	mutable std::optional<int64_t> m_position;
	bool m_immutable{ false };

  public:
	Label(std::string name, int64_t function_id)
		: m_label_name(std::move(name)), m_function_id(function_id)
	{}

	void set_position(int64_t position) const
	{
		// a hack to make sure multiple instructions cannot relocate a
		// shared label
		if (!m_immutable) { m_position = position; }
	}

	int64_t position() const
	{
		ASSERT(m_position.has_value());
		return *m_position;
	}

	void immutable() { m_immutable = true; }

	size_t function_id() const { return m_function_id; }

	const std::string &name() const { return m_label_name; }

	size_t hash() const
	{
		size_t seed = std::hash<std::string>{}(m_label_name);
		seed ^= std::hash<size_t>{}(m_function_id) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
		return seed;
	}

	bool operator<(const Label &other) const { return hash() < other.hash(); }
};
