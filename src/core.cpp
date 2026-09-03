#include "core.hpp"

#include "spdlog/spdlog.h"

#include <cstdlib>

namespace detail {
void assertion_failed(const char *what, const char *file, int line)
{
	spdlog::error("{} {}:{}", what, file, line);
	std::abort();
}

void log_debug(const char *message) { spdlog::debug("{}", message); }
void log_trace(const char *message) { spdlog::trace("{}", message); }
void log_error(const char *message) { spdlog::error("{}", message); }

// Lets callers skip formatting entirely when debug logging is off - the
// spdlog::debug("...", args) form did that for free.
bool log_debug_enabled() { return spdlog::should_log(spdlog::level::debug); }
}// namespace detail
