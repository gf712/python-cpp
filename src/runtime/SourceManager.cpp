#include "SourceManager.hpp"

#include <fstream>

namespace py {

SourceManager &SourceManager::the()
{
	static SourceManager instance;
	return instance;
}

std::string_view SourceManager::line(const std::string &filename, size_t lineno)
{
	const auto &lines = load(filename);
	if (lineno == 0 || lineno > lines.size()) { return {}; }
	return lines[lineno - 1];
}

std::string_view SourceManager::strip_leading_whitespace(std::string_view s)
{
	const auto first = s.find_first_not_of(" \t");
	if (first == std::string_view::npos) { return {}; }
	return s.substr(first);
}

const std::vector<std::string> &SourceManager::load(const std::string &filename)
{
	if (auto it = m_files.find(filename); it != m_files.end()) { return it->second; }
	std::vector<std::string> lines;
	if (std::ifstream f{ filename }) {
		for (std::string l; std::getline(f, l);) { lines.push_back(std::move(l)); }
	}
	return m_files.emplace(filename, std::move(lines)).first->second;
}

}// namespace py
