#pragma once

#include "runtime/Value.hpp"
#include "runtime/ValueError.hpp"

#include <variant>

namespace py {

// bool is_initialized() const - required by check_initialized()
// bool is_detached() const    - optional; reported separately when the object was
//                               initialized once and then detached
template<typename Derived> class IOChecks
{
	const Derived &derived() const { return static_cast<const Derived &>(*this); }

  public:
	PyResult<std::monostate> check_initialized() const
	{
		if (derived().is_initialized()) { return Ok(std::monostate{}); }
		if constexpr (requires(const Derived &d) { d.is_detached(); }) {
			if (derived().is_detached()) {
				return Err(value_error("raw stream has been detached"));
			}
		}
		return Err(value_error("I/O operation on uninitialized object"));
	}

	// the guard pair an accessor wants: usable means constructed *and* still open.
	// Spelling it once keeps the two checks from being chained in the wrong order.
	PyResult<std::monostate> check_usable() const
	{
		return check_initialized().and_then([this](auto) { return derived().check_closed(); });
	}
};

}// namespace py
