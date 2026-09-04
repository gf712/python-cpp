export module py.runtime:parser;

import std;
import py.ast;
import py.lexer;
import :value;

export namespace parser {

class Parser
{
	std::shared_ptr<ast::Module> m_module;
	Lexer &m_lexer;
	std::size_t m_token_position{ 0 };

  public:
	struct CacheValue
	{
		// AST nodes are owned by the Module's arena; the cache holds non-owning
		// raw pointers safe to alias across backtracking attempts.
		using ValueType = std::variant<ast::ASTNode *, std::vector<Token>>;
		std::variant<bool, ValueType> value;
		std::size_t position;
	};

	using MemoSlot = std::optional<CacheValue>;

	MemoSlot *memo_find(std::size_t position, std::uint16_t rule)
	{
		if (position >= m_memo_index.size()) { return nullptr; }
		for (const auto &[id, slot] : m_memo_index[position]) {
			if (id == rule) { return &m_memo_pool[slot]; }
		}
		return nullptr;
	}

	MemoSlot &memo_insert(std::size_t position, std::uint16_t rule)
	{
		if (auto *existing = memo_find(position, rule)) { return *existing; }
		if (position >= m_memo_index.size()) { m_memo_index.resize(position + 1); }
		m_memo_pool.emplace_back();
		m_memo_index[position].emplace_back(
			rule, static_cast<std::uint32_t>(m_memo_pool.size() - 1));
		return m_memo_pool.back();
	}

  private:
	std::deque<MemoSlot> m_memo_pool;
	std::vector<std::vector<std::pair<std::uint16_t, std::uint32_t>>> m_memo_index;

  public:
	Parser(Lexer &l) : m_module(std::make_shared<ast::Module>(l.filename())), m_lexer(l)
	{
		m_lexer.ignore_nl_token() = true;
		m_lexer.ignore_comments() = true;
	}

	Lexer &lexer() { return m_lexer; }

	std::shared_ptr<ast::Module> module() { return m_module; }

	// Arena that owns every child node reachable from the parsed Module.
	ast::ASTArena &arena() { return m_module->arena(); }

	const std::size_t &token_position() const { return m_token_position; }
	std::size_t &token_position() { return m_token_position; }

	// parses a file
	void parse();

	// parses an expression used by the builtin `eval` function
	py::PyResult<std::shared_ptr<ast::Module>> parse_expression();

	// parses the rule inside the {} in an fstring
	py::PyResult<ast::ASTNode *> parse_fstring();
};

}// namespace parser
