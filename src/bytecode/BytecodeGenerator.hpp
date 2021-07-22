#pragma once

#include "forward.hpp"
#include "utilities.hpp"

#include <vector>
#include <memory>
#include <set>

struct FunctionMetaData
{
	size_t offset;
	std::string function_name;
	size_t register_count;
};

class Bytecode
{
	std::vector<std::unique_ptr<Instruction>> m_instructions;
	std::vector<FunctionMetaData> m_functions;
	size_t m_main_local_register_count;

  public:
	Bytecode(std::vector<std::unique_ptr<Instruction>> &&ins,
		std::vector<FunctionMetaData> &&funcs,
		size_t main_local_register_count);

	size_t start_offset() const { return m_functions.back().offset; }
	size_t main_local_register_count() const { return m_main_local_register_count; }

	const std::vector<std::unique_ptr<Instruction>> &instructions() const { return m_instructions; }
	const std::vector<FunctionMetaData> &functions() const { return m_functions; }

	size_t function_offset(size_t function_id) const { return m_functions[function_id - 1].offset; }

	std::string to_string() const;
};


class Label
{
	std::string m_label_name;
	size_t m_function_id;
	mutable std::optional<size_t> m_position;

  public:
	Label(std::string name, size_t function_id)
		: m_label_name(std::move(name)), m_function_id(function_id)
	{}

	void set_position(size_t position) const { m_position = position; }

	size_t position() const
	{
		ASSERT(m_position.has_value());
		return *m_position;
	}

	size_t function_id() const { return m_function_id; }

	const std::string &name() const { return m_label_name; }

	bool operator<(const Label &other) const
	{
		return (m_label_name < other.name()) || (m_function_id < other.function_id());
	}
};

class BytecodeGenerator;

struct FunctionInfo
{
	size_t function_id;
	BytecodeGenerator *generator;

	FunctionInfo(size_t, BytecodeGenerator *);
	~FunctionInfo();
};


class BytecodeGenerator
{
  public:
	static constexpr size_t start_register = 1;

  private:
	std::vector<std::vector<std::unique_ptr<Instruction>>> m_functions;
	std::vector<size_t> m_function_register_count;

	std::set<Label> m_labels;
	std::vector<size_t> m_frame_register_count;

  public:
	BytecodeGenerator();
	~BytecodeGenerator();

	static std::shared_ptr<Bytecode> compile(std::shared_ptr<ast::ASTNode> node);

	template<typename OpType, typename... Args> void emit(size_t function_id, Args &&... args)
	{
		m_functions.at(function_id)
			.push_back(std::make_unique<OpType>(std::forward<Args>(args)...));
	}

	friend std::ostream &operator<<(std::ostream &os, BytecodeGenerator &generator);

	std::string to_string() const;

	const std::vector<std::unique_ptr<Instruction>> &function(size_t idx) const
	{
		return m_functions.at(idx);
	}

	Label make_label(const std::string &name, size_t function_id)
	{
		spdlog::debug("New label to be added: name={} function_id={}", name, function_id);
		if (auto result = m_labels.emplace(name, function_id); result.second) {
			return *result.first;
		}
		ASSERT_NOT_REACHED()
	}

	Label label(const Label &l) const
	{
		if (auto result = m_labels.find(l); result != m_labels.end()) { return *result; }
		ASSERT_NOT_REACHED()
	}

	void bind(Label &label)
	{
		if (auto result = m_labels.find(label); result != m_labels.end()) {
			const size_t current_instruction_position = function(result->function_id()).size();
			result->set_position(current_instruction_position);
			return;
		}
		ASSERT_NOT_REACHED()
	}

	size_t register_count() const { return m_frame_register_count.back(); }

	Register allocate_register() { return m_frame_register_count.back()++; }
	FunctionInfo allocate_function();

	void enter_function() { m_frame_register_count.emplace_back(start_register); }

	void exit_function()
	{
		m_function_register_count.push_back(register_count());
		m_frame_register_count.pop_back();
	}

  private:
	std::shared_ptr<Bytecode> generate_executable();
	void relocate_labels(const std::vector<std::unique_ptr<Instruction>> &executable,
		const std::vector<size_t> &offsets);
};
