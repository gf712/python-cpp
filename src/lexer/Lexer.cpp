#include "Lexer.hpp"

#include <filesystem>
#include <fstream>
#include <optional>
#include <string>

std::optional<std::string> read_file(const std::string &filename)
{
	std::filesystem::path path = filename;
	if (!std::filesystem::exists(path)) {
		std::cerr << fmt::format("File {} does not exist", path.c_str()) << std::endl;
		return {};
	}

	std::ifstream in(std::filesystem::absolute(path).c_str());
	if (!in.is_open()) {
		std::cerr << fmt::format("Failed to open {}", std::filesystem::absolute(path).c_str())
				  << std::endl;
		return {};
	}

	std::string program;

	in.seekg(0, std::ios::end);
	program.reserve(in.tellg());
	in.seekg(0, std::ios::beg);

	program.assign((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
	if (program.back() != '\n') { program.append("\n"); }

	spdlog::debug("Input program: \n----\n{}\n----\n", program.c_str());
	return program;
}