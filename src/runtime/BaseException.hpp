#pragma once

#include "PyObject.hpp"

class BaseException : public PyObject
{
	std::string m_exception_name;
	std::string m_message;

  public:
	BaseException(std::string exception_name, std::string &&name)
		: PyObject(PyObjectType::PY_BASE_EXCEPTION), m_exception_name(std::move(exception_name)),
		  m_message(std::move(name))
	{}

	std::string what() const { return fmt::format("{}: {}", m_exception_name, m_message); }

	void set_message(std::string msg) { m_message = std::move(msg); }
};