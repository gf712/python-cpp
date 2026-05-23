#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace py {

// Caches source files on disk so traceback rendering can show the line
// that raised, without re-reading the file once per frame. Lookups are
// 1-indexed (matching tb_lineno). Failed opens are negatively cached so
// repeated requests for an unreadable file don't keep hitting disk.
class SourceManager
{
  public:
	static SourceManager &the();

	// Returns the line at `lineno` (1-indexed) in `filename`, or an empty
	// view if the file can't be opened, the line is out of range, or
	// `lineno == 0`.
	std::string_view line(const std::string &filename, size_t lineno);

	// Returns `s` with leading spaces and tabs removed. Provided here so
	// traceback formatters (the main consumer of source lines) don't grow
	// their own copy.
	static std::string_view strip_leading_whitespace(std::string_view s);

  private:
	SourceManager() = default;

	const std::vector<std::string> &load(const std::string &filename);

	std::unordered_map<std::string, std::vector<std::string>> m_files;
};

}// namespace py
