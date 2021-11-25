#pragma once

#include "PyObject.hpp"

class BaseException : public PyBaseObject
{
	std::string m_exception_name;
	std::string m_message;

  public:
	BaseException(std::string exception_name, std::string &&name);

	std::string what() const { return fmt::format("{}: {}", m_exception_name, m_message); }

	void set_message(std::string msg) { m_message = std::move(msg); }

	static std::unique_ptr<TypePrototype> register_type();

	 PyType *type_() const;
};