#pragma once
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

#include "core.hpp"


class Label
	: NonCopyable
	, NonMoveable
{
	std::string m_label_name;
	std::size_t m_function_id;
	mutable std::optional<std::int64_t> m_position;

  public:
	// Public rather than protected + friend: the two users (BytecodeGenerator,
	// PythonBytecodeEmitter) live in py.codegen, and naming them here would
	// declare them in the global module.
	void set_position(std::int64_t position) const
	{
		ASSERT(!m_position.has_value());
		m_position = position;
	}

  public:
	Label(std::string name, std::size_t function_id)
		: m_label_name(std::move(name)), m_function_id(function_id)
	{}

	Label(std::int64_t position) : m_position(position) {}

	std::int64_t position() const
	{
		ASSERT(m_position.has_value());
		return *m_position;
	}

	std::size_t function_id() const { return m_function_id; }

	const std::string &name() const { return m_label_name; }

	std::size_t hash() const
	{
		std::size_t seed = std::hash<std::string>{}(m_label_name);
		seed ^= std::hash<std::size_t>{}(m_function_id) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
		return seed;
	}

	bool operator<(const Label &other) const { return hash() < other.hash(); }
};
