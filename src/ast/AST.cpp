#include "AST.hpp"

#include "runtime/PyObject.hpp"
#include "runtime/Value.hpp"

namespace ast {

#define __AST_NODE_TYPE(x)                                                                        \
	template<> x *as(ASTNode *node)                                                               \
	{                                                                                             \
		if (node && node->node_type() == ASTNodeType::x) { return static_cast<x *>(node); }       \
		return nullptr;                                                                           \
	}                                                                                             \
	template<> const x *as(const ASTNode *node)                                                   \
	{                                                                                             \
		if (node && node->node_type() == ASTNodeType::x) { return static_cast<const x *>(node); } \
		return nullptr;                                                                           \
	}
AST_NODE_TYPES
#undef __AST_NODE_TYPE

#define __AST_NODE_TYPE(NodeType) \
	Value *NodeType::codegen(CodeGenerator *generator) const { return generator->visit(this); }
AST_NODE_TYPES
#undef __AST_NODE_TYPE

void NodeVisitor::dispatch(ASTNode *node)
{
#define __AST_NODE_TYPE(NodeType)             \
	case ASTNodeType::NodeType: {             \
		visit(static_cast<NodeType *>(node)); \
	} break;
	switch (node->node_type()) {
		AST_NODE_TYPES
	}
#undef __AST_NODE_TYPE
}

#define __AST_NODE_TYPE(NodeType) \
	void NodeVisitor::dispatch(NodeType *node) { visit(node); }
AST_NODE_TYPES
#undef __AST_NODE_TYPE

void NodeVisitor::visit(Constant *) {}

void NodeVisitor::visit(Expression *node) { dispatch(node->value()); }

void NodeVisitor::visit(List *node)
{
	for (auto &el : node->elements()) { dispatch(el); }
}

void NodeVisitor::visit(Tuple *node)
{
	for (auto &el : node->elements()) { dispatch(el); }
}

void NodeVisitor::visit(Dict *node)
{
	for (auto &el : node->keys()) { dispatch(el); }
	for (auto &el : node->values()) { dispatch(el); }
}

void NodeVisitor::visit(Set *node)
{
	for (auto &el : node->elements()) { dispatch(el); }
}

void NodeVisitor::visit(Name *) {}

void NodeVisitor::visit(Assign *node)
{
	for (const auto &target : node->targets()) { dispatch(target); }
	if (node->value()) dispatch(node->value());
}

void NodeVisitor::visit(BinaryExpr *node)
{
	dispatch(node->lhs());
	dispatch(node->rhs());
}

void NodeVisitor::visit(AugAssign *node)
{
	dispatch(node->target());
	dispatch(node->value());
}

void NodeVisitor::visit(Return *node) { dispatch(node->value()); }

void NodeVisitor::visit(Yield *node) { dispatch(node->value()); }

void NodeVisitor::visit(YieldFrom *node) { dispatch(node->value()); }

void NodeVisitor::visit(Argument *node)
{
	if (node->annotation()) dispatch(node->annotation());
}

void NodeVisitor::visit(Arguments *node)
{
	for (auto &el : node->posonlyargs()) { dispatch(el); }
	for (auto &el : node->args()) { dispatch(el); }
	if (node->vararg()) dispatch(node->vararg());
	for (auto &el : node->kwonlyargs()) { dispatch(el); }
	for (auto &el : node->kw_defaults()) { dispatch(el); }
	if (node->kwarg()) dispatch(node->kwarg());
	for (auto &el : node->defaults()) { dispatch(el); }
}

void NodeVisitor::visit(FunctionDefinition *node)
{
	dispatch(node->args());
	for (auto &el : node->body()) { dispatch(el); }
	for (auto &el : node->decorator_list()) { dispatch(el); }
	dispatch(node->returns());
}

void NodeVisitor::visit(AsyncFunctionDefinition *node)
{
	dispatch(node->args());
	for (auto &el : node->body()) { dispatch(el); }
	for (auto &el : node->decorator_list()) { dispatch(el); }
	dispatch(node->returns());
}

void NodeVisitor::visit(Await *node) { dispatch(node->value()); }

void NodeVisitor::visit(Lambda *node)
{
	dispatch(node->args());
	dispatch(node->body());
}

void NodeVisitor::visit(Keyword *node) { dispatch(node->value()); }

void NodeVisitor::visit(ClassDefinition *node)
{
	for (auto &el : node->bases()) { dispatch(el); };
	for (auto &el : node->keywords()) { dispatch(el); };
	for (auto &el : node->body()) { dispatch(el); };
	for (auto &el : node->decorator_list()) { dispatch(el); };
}

void NodeVisitor::visit(Call *node)
{
	dispatch(node->function());
	for (auto &el : node->args()) { dispatch(el); };
	for (auto &el : node->keywords()) { dispatch(el); };
}

void NodeVisitor::visit(Module *node)
{
	for (auto &el : node->body()) { dispatch(el); }
}

void NodeVisitor::visit(If *node)
{
	dispatch(node->test());
	for (auto &el : node->body()) { dispatch(el); }
	for (auto &el : node->orelse()) { dispatch(el); }
}

void NodeVisitor::visit(For *node)
{
	dispatch(node->target());
	dispatch(node->iter());
	for (auto &el : node->body()) { dispatch(el); }
	for (auto &el : node->orelse()) { dispatch(el); }
}

void NodeVisitor::visit(While *node)
{
	dispatch(node->test());
	for (auto &el : node->body()) { dispatch(el); }
	for (auto &el : node->orelse()) { dispatch(el); }
}

void NodeVisitor::visit(Compare *node)
{
	dispatch(node->lhs());
	for (auto &el : node->comparators()) { dispatch(el); }
}

void NodeVisitor::visit(Attribute *node) { dispatch(node->value()); }

void NodeVisitor::visit(Import *) {}

void NodeVisitor::visit(ImportFrom *) {}

void NodeVisitor::visit(Subscript *node)
{
	dispatch(node->value());
	std::visit(overloaded{ [this](const Subscript::Index &val) { dispatch(val.value); },
				   [this](const Subscript::Slice &val) {
					   if (val.lower) dispatch(val.lower);
					   if (val.upper) dispatch(val.upper);
					   if (val.step) dispatch(val.step);
				   },
				   [this](const Subscript::ExtSlice &val) {
					   for (auto &dim : val.dims) {
						   std::visit(
							   overloaded{
								   [this](const Subscript::Index &val) { dispatch(val.value); },
								   [this](const Subscript::Slice &val) {
									   if (val.lower) dispatch(val.lower);
									   if (val.upper) dispatch(val.upper);
									   if (val.step) dispatch(val.step);
								   },
							   },
							   dim);
					   }
				   } },
		node->slice());
}

void NodeVisitor::visit(Raise *node)
{
	if (node->exception()) { dispatch(node->exception()); }
	if (node->cause()) { dispatch(node->cause()); }
}

void NodeVisitor::visit(ExceptHandler *node)
{
	if (node->type()) { dispatch(node->type()); }
	for (auto &el : node->body()) { dispatch(el); }
}

void NodeVisitor::visit(Try *node)
{
	for (auto &el : node->body()) { dispatch(el); }
	for (auto &el : node->handlers()) { dispatch(el); }
	for (auto &el : node->orelse()) { dispatch(el); }
	for (auto &el : node->finalbody()) { dispatch(el); }
}

void NodeVisitor::visit(Assert *node)
{
	if (node->test()) { dispatch(node->test()); }
	if (node->msg()) { dispatch(node->msg()); }
}

void NodeVisitor::visit(UnaryExpr *node) { dispatch(node->operand()); }

void NodeVisitor::visit(BoolOp *node)
{
	for (auto &el : node->values()) { dispatch(el); }
}

void NodeVisitor::visit(Pass *) {}

void NodeVisitor::visit(Continue *) {}

void NodeVisitor::visit(Break *) {}

void NodeVisitor::visit(Global *) {}

void NodeVisitor::visit(NonLocal *) {}

void NodeVisitor::visit(Delete *node)
{
	for (auto &el : node->targets()) { dispatch(el); }
}

void NodeVisitor::visit(With *node)
{
	for (auto &el : node->items()) { dispatch(el); }
	for (auto &el : node->body()) { dispatch(el); }
}

void NodeVisitor::visit(WithItem *node)
{
	dispatch(node->context_expr());
	if (node->optional_vars()) dispatch(node->optional_vars());
}

void NodeVisitor::visit(IfExpr *node)
{
	dispatch(node->test());
	dispatch(node->body());
	dispatch(node->orelse());
}

void NodeVisitor::visit(Starred *node) { dispatch(node->value()); }

void NodeVisitor::visit(NamedExpr *node)
{
	dispatch(node->target());
	dispatch(node->value());
}

void NodeVisitor::visit(JoinedStr *node)
{
	for (auto &el : node->values()) { dispatch(el); }
}

void NodeVisitor::visit(FormattedValue *node)
{
	dispatch(node->value());
	dispatch(node->format_spec());
}


void NodeVisitor::visit(Comprehension *node)
{
	dispatch(node->target());
	dispatch(node->iter());
	for (auto &if_ : node->ifs()) { dispatch(if_); }
}

void NodeVisitor::visit(ListComp *node)
{
	dispatch(node->elt());
	for (auto &generator : node->generators()) { dispatch(generator); }
}

void NodeVisitor::visit(DictComp *node)
{
	dispatch(node->key());
	dispatch(node->value());
	for (auto &generator : node->generators()) { dispatch(generator); }
}

void NodeVisitor::visit(GeneratorExp *node)
{
	dispatch(node->elt());
	for (auto &generator : node->generators()) { dispatch(generator); }
}

void NodeVisitor::visit(SetComp *node)
{
	dispatch(node->elt());
	for (auto &generator : node->generators()) { dispatch(generator); }
}

// TODO: re-port to arena ownership and re-enable. Disabled during the
// shared_ptr -> arena migration of AST nodes; only ConstantFolding and
// its tests depend on this visitor, and they are excluded from the build.
#if 0
void NodeTransformVisitor::transform_single_node(ASTNode * node)
{
	m_can_return_multiple_nodes = false;
#define __AST_NODE_TYPE(NodeType)                                        \
	case ASTNodeType::NodeType: {                                        \
		auto new_node = visit(std::static_pointer_cast<NodeType>(node)); \
		if (new_node.empty()) break;                                     \
		ASSERT(new_node.size() == 1);                                    \
		node.swap(new_node[0]);                                          \
	} break;
	switch (node->node_type()) {
		AST_NODE_TYPES
	}
#undef __AST_NODE_TYPE
}

void NodeTransformVisitor::transform_multiple_nodes(std::vector<ASTNode *> &nodes)
{
	std::vector<ASTNode *> new_node_vector;
	for (auto &node : nodes) {
		m_can_return_multiple_nodes = true;
		auto new_nodes = [node, this]() -> std::vector<ASTNode *> {
#define __AST_NODE_TYPE(NodeType)                               \
	case ASTNodeType::NodeType: {                               \
		return visit(std::static_pointer_cast<NodeType>(node)); \
	}
			switch (node->node_type()) {
				AST_NODE_TYPES
#undef __AST_NODE_TYPE
			}
			ASSERT_NOT_REACHED();
		}();
		new_node_vector.insert(new_node_vector.end(), new_nodes.begin(), new_nodes.end());
	}
	nodes = std::move(new_node_vector);
}

std::vector<ASTNode *> NodeTransformVisitor::visit(Constant * node)
{
	return { node };
}

std::vector<ASTNode *> NodeTransformVisitor::visit(Expression * node)
{
	transform_single_node(node->value());
	return { node };
}

std::vector<ASTNode *> NodeTransformVisitor::visit(List * node)
{
	for (auto &el : node->elements()) { transform_single_node(el); }
	return { node };
}

std::vector<ASTNode *> NodeTransformVisitor::visit(Tuple * node)
{
	for (auto &el : node->elements()) { transform_single_node(el); }
	return { node };
}

std::vector<ASTNode *> NodeTransformVisitor::visit(Dict * node)
{
	for (auto &el : node->keys()) { transform_single_node(el); }
	for (auto &el : node->values()) { transform_single_node(el); }
	return { node };
}

std::vector<ASTNode *> NodeTransformVisitor::visit(Set * node)
{
	for (auto &el : node->elements()) { transform_single_node(el); }
	return { node };
}

std::vector<ASTNode *> NodeTransformVisitor::visit(Name * node)
{
	return { node };
}

std::vector<ASTNode *> NodeTransformVisitor::visit(Assign * node)
{
	for (const auto &target : node->targets()) { transform_single_node(target); }
	if (node->value()) transform_single_node(node->value());

	return { node };
}

std::vector<ASTNode *> NodeTransformVisitor::visit(BinaryExpr * node)
{
	transform_single_node(node->lhs());
	transform_single_node(node->rhs());

	return { node };
}

std::vector<ASTNode *> NodeTransformVisitor::visit(AugAssign * node)
{
	transform_single_node(node->target());
	transform_single_node(node->value());

	return { node };
}

std::vector<ASTNode *> NodeTransformVisitor::visit(Return * node)
{
	transform_single_node(node->value());

	return { node };
}

std::vector<ASTNode *> NodeTransformVisitor::visit(Yield * node)
{
	transform_single_node(node->value());

	return { node };
}

std::vector<ASTNode *> NodeTransformVisitor::visit(YieldFrom * node)
{
	transform_single_node(node->value());

	return { node };
}

std::vector<ASTNode *> NodeTransformVisitor::visit(Argument * node)
{
	if (node->annotation()) transform_single_node(node->annotation());
	return { node };
}

std::vector<ASTNode *> NodeTransformVisitor::visit(Arguments * node)
{
	for (auto &el : node->posonlyargs()) { transform_single_node(el); }
	for (auto &el : node->args()) { transform_single_node(el); }
	if (node->vararg()) transform_single_node(node->vararg());
	for (auto &el : node->kwonlyargs()) { transform_single_node(el); }
	for (auto &el : node->kw_defaults()) { transform_single_node(el); }
	if (node->kwarg()) transform_single_node(node->kwarg());
	for (auto &el : node->defaults()) { transform_single_node(el); }
	return { node };
}

std::vector<ASTNode *> NodeTransformVisitor::visit(
	FunctionDefinition * node)
{
	transform_single_node(node->args());
	transform_multiple_nodes(node->body());
	for (auto &el : node->decorator_list()) { transform_single_node(el); }
	transform_single_node(node->returns());
	return { node };
}

std::vector<ASTNode *> NodeTransformVisitor::visit(
	AsyncFunctionDefinition * node)
{
	transform_single_node(node->args());
	transform_multiple_nodes(node->body());
	for (auto &el : node->decorator_list()) { transform_single_node(el); }
	transform_single_node(node->returns());
	return { node };
}

std::vector<ASTNode *> NodeTransformVisitor::visit(Await * node)
{
	transform_single_node(node->value());
	return { node };
}

std::vector<ASTNode *> NodeTransformVisitor::visit(Lambda * node)
{
	transform_single_node(node->args());
	transform_single_node(node->body());
	return { node };
}

std::vector<ASTNode *> NodeTransformVisitor::visit(Keyword * node)
{
	transform_single_node(node->value());
	return { node };
}

std::vector<ASTNode *> NodeTransformVisitor::visit(
	ClassDefinition * node)
{
	for (auto &el : node->bases()) { transform_single_node(el); };
	for (auto &el : node->keywords()) { transform_single_node(el); };
	transform_multiple_nodes(node->body());
	for (auto &el : node->decorator_list()) { transform_single_node(el); };
	return { node };
}

std::vector<ASTNode *> NodeTransformVisitor::visit(Call * node)
{
	transform_single_node(node->function());
	for (auto &el : node->args()) { transform_single_node(el); };
	for (auto &el : node->keywords()) { transform_single_node(el); };
	return { node };
}

std::vector<ASTNode *> NodeTransformVisitor::visit(Module * node)
{
	transform_multiple_nodes(node->body());
	return { node };
}

std::vector<ASTNode *> NodeTransformVisitor::visit(If * node)
{
	transform_single_node(node->test());
	transform_multiple_nodes(node->body());
	transform_multiple_nodes(node->orelse());
	return { node };
}

std::vector<ASTNode *> NodeTransformVisitor::visit(For * node)
{
	transform_single_node(node->target());
	transform_single_node(node->iter());
	transform_multiple_nodes(node->body());
	transform_multiple_nodes(node->orelse());
	return { node };
}

std::vector<ASTNode *> NodeTransformVisitor::visit(While * node)
{
	transform_single_node(node->test());
	transform_multiple_nodes(node->body());
	transform_multiple_nodes(node->orelse());
	return { node };
}

std::vector<ASTNode *> NodeTransformVisitor::visit(Compare * node)
{
	transform_single_node(node->lhs());
	transform_multiple_nodes(node->comparators());
	return { node };
}

std::vector<ASTNode *> NodeTransformVisitor::visit(Attribute * node)
{
	transform_single_node(node->value());
	return { node };
}

std::vector<ASTNode *> NodeTransformVisitor::visit(Import * node)
{
	return { node };
}

std::vector<ASTNode *> NodeTransformVisitor::visit(ImportFrom * node)
{
	return { node };
}

std::vector<ASTNode *> NodeTransformVisitor::visit(Subscript * node)
{
	transform_single_node(node->value());
	std::visit(
		overloaded{ [this](const Subscript::Index &val) { transform_single_node(val.value); },
			[this](const Subscript::Slice &val) {
				if (val.lower) transform_single_node(val.lower);
				if (val.upper) transform_single_node(val.upper);
				if (val.step) transform_single_node(val.step);
			},
			[this](const Subscript::ExtSlice &val) {
				for (auto &dim : val.dims) {
					std::visit(overloaded{
								   [this](const Subscript::Index &val) {
									   transform_single_node(val.value);
								   },
								   [this](const Subscript::Slice &val) {
									   if (val.lower) transform_single_node(val.lower);
									   if (val.upper) transform_single_node(val.upper);
									   if (val.step) transform_single_node(val.step);
								   },
							   },
						dim);
				}
			} },
		node->slice());
	return { node };
}

std::vector<ASTNode *> NodeTransformVisitor::visit(Raise * node)
{
	if (node->exception()) { transform_single_node(node->exception()); }
	if (node->cause()) { transform_single_node(node->cause()); }
	return { node };
}

std::vector<ASTNode *> NodeTransformVisitor::visit(
	ExceptHandler * node)
{
	if (node->type()) { transform_single_node(node->type()); }
	transform_multiple_nodes(node->body());
	return { node };
}

std::vector<ASTNode *> NodeTransformVisitor::visit(Try * node)
{
	transform_multiple_nodes(node->body());
	for (auto &el : node->handlers()) { transform_single_node(el); }
	transform_multiple_nodes(node->orelse());
	transform_multiple_nodes(node->finalbody());
	return { node };
}

std::vector<ASTNode *> NodeTransformVisitor::visit(Assert * node)
{
	if (node->test()) { transform_single_node(node->test()); }
	if (node->msg()) { transform_single_node(node->msg()); }
	return { node };
}

std::vector<ASTNode *> NodeTransformVisitor::visit(UnaryExpr * node)
{
	transform_single_node(node->operand());
	return { node };
}

std::vector<ASTNode *> NodeTransformVisitor::visit(BoolOp * node)
{
	for (auto &el : node->values()) { transform_single_node(el); }
	return { node };
}

std::vector<ASTNode *> NodeTransformVisitor::visit(Pass * node)
{
	return { node };
}

std::vector<ASTNode *> NodeTransformVisitor::visit(Continue * node)
{
	return { node };
}

std::vector<ASTNode *> NodeTransformVisitor::visit(Break * node)
{
	return { node };
}

std::vector<ASTNode *> NodeTransformVisitor::visit(Global * node)
{
	return { node };
}

std::vector<ASTNode *> NodeTransformVisitor::visit(NonLocal * node)
{
	return { node };
}

std::vector<ASTNode *> NodeTransformVisitor::visit(Delete * node)
{
	for (auto &el : node->targets()) { transform_single_node(el); }
	return { node };
}

std::vector<ASTNode *> NodeTransformVisitor::visit(With * node)
{
	for (auto &el : node->items()) { transform_single_node(el); }
	transform_multiple_nodes(node->body());
	return { node };
}

std::vector<ASTNode *> NodeTransformVisitor::visit(WithItem * node)
{
	transform_single_node(node->context_expr());
	if (node->optional_vars()) transform_single_node(node->optional_vars());
	return { node };
}

std::vector<ASTNode *> NodeTransformVisitor::visit(IfExpr * node)
{
	transform_single_node(node->test());
	transform_single_node(node->body());
	transform_single_node(node->orelse());
	return { node };
}

std::vector<ASTNode *> NodeTransformVisitor::visit(Starred * node)
{
	transform_single_node(node->value());
	return { node };
}

std::vector<ASTNode *> NodeTransformVisitor::visit(NamedExpr * node)
{
	transform_single_node(node->target());
	transform_single_node(node->value());
	return { node };
}

std::vector<ASTNode *> NodeTransformVisitor::visit(JoinedStr * node)
{
	for (auto &el : node->values()) { transform_single_node(el); }
	return { node };
}

std::vector<ASTNode *> NodeTransformVisitor::visit(
	FormattedValue * node)
{
	transform_single_node(node->value());
	transform_single_node(node->format_spec());
	return { node };
}

std::vector<ASTNode *> NodeTransformVisitor::visit(
	Comprehension * node)
{
	transform_single_node(node->target());
	transform_single_node(node->iter());
	transform_multiple_nodes(node->ifs());
	return { node };
}

std::vector<ASTNode *> NodeTransformVisitor::visit(ListComp * node)
{
	transform_single_node(node->elt());
	TODO();
	// transform_multiple_nodes(node->generators());
	return { node };
}

std::vector<ASTNode *> NodeTransformVisitor::visit(DictComp * node)
{
	transform_single_node(node->key());
	transform_single_node(node->value());
	TODO();
	// transform_multiple_nodes(node->generators());
	return { node };
}

std::vector<ASTNode *> NodeTransformVisitor::visit(
	GeneratorExp * node)
{
	transform_single_node(node->elt());
	TODO();
	// transform_multiple_nodes(node->generators());
	return { node };
}

std::vector<ASTNode *> NodeTransformVisitor::visit(SetComp * node)
{
	transform_single_node(node->elt());
	TODO();
	// transform_multiple_nodes(node->generators());
	return { node };
}
#endif

Constant::Constant(double value, SourceLocation source_location)
	: ASTNode(ASTNodeType::Constant, source_location),
	  m_value(std::make_unique<py::Value>(py::Number{ value }))
{}

Constant::Constant(int64_t value, SourceLocation source_location)
	: ASTNode(ASTNodeType::Constant, source_location),
	  m_value(std::make_unique<py::Value>(py::Number{ value }))
{}

Constant::Constant(mpz_class value, SourceLocation source_location)
	: ASTNode(ASTNodeType::Constant, source_location),
	  m_value(std::make_unique<py::Value>(py::Number{ value }))
{}

Constant::Constant(bool value, SourceLocation source_location)
	: ASTNode(ASTNodeType::Constant, source_location),
	  m_value(std::make_unique<py::Value>(py::NameConstant{ value }))
{}

Constant::Constant(std::string value, SourceLocation source_location)
	: ASTNode(ASTNodeType::Constant, source_location),
	  m_value(std::make_unique<py::Value>(py::String{ std::move(value) }))
{}

Constant::Constant(const char *value, SourceLocation source_location)
	: ASTNode(ASTNodeType::Constant, source_location),
	  m_value(std::make_unique<py::Value>(py::String{ std::string(value) }))
{}

Constant::Constant(const py::Value &value, SourceLocation source_location)
	: ASTNode(ASTNodeType::Constant, source_location), m_value(std::make_unique<py::Value>(value))
{}

void Expression::print_this_node(const std::string &indent) const
{
	spdlog::debug("{}Expression", indent);
	std::string new_indent = indent + std::string(6, ' ');
	m_value->print_node(new_indent);
}

void Constant::print_this_node(const std::string &indent) const
{
	spdlog::debug("{}Constant [{}:{}-{}:{}]",
		indent,
		source_location().start.row + 1,
		source_location().start.column + 1,
		source_location().end.row + 1,
		source_location().end.column + 1);
	std::visit(overloaded{ [&indent](const py::String &value) {
							  spdlog::debug("{}  - value: \"{}\"", indent, value.to_string());
						  },
				   [&indent](const auto &value) {
					   spdlog::debug("{}  - value: {}", indent, value.to_string());
				   },
				   [&indent](py::PyObject *const value) {
					   spdlog::debug("{}  - value: {}", indent, value->to_string());
				   } },
		*m_value);
}

void List::print_this_node(const std::string &indent) const
{
	spdlog::debug("{}List [{}:{}-{}:{}]",
		indent,
		source_location().start.row + 1,
		source_location().start.column + 1,
		source_location().end.row + 1,
		source_location().end.column + 1);
	spdlog::debug("{}  context: {}", indent, static_cast<int>(m_ctx));
	spdlog::debug("{}  elements:", indent);
	std::string new_indent = indent + std::string(6, ' ');
	for (const auto &el : m_elements) { el->print_node(new_indent); }
}

void Tuple::print_this_node(const std::string &indent) const
{
	spdlog::debug("{}Tuple [{}:{}-{}:{}]",
		indent,
		source_location().start.row + 1,
		source_location().start.column + 1,
		source_location().end.row + 1,
		source_location().end.column + 1);
	spdlog::debug("{}  context: {}", indent, static_cast<int>(m_ctx));
	spdlog::debug("{}  elements:", indent);
	std::string new_indent = indent + std::string(6, ' ');
	for (const auto &el : m_elements) { el->print_node(new_indent); }
}

void Dict::print_this_node(const std::string &indent) const
{
	spdlog::debug("{}Dict [{}:{}-{}:{}]",
		indent,
		source_location().start.row + 1,
		source_location().start.column + 1,
		source_location().end.row + 1,
		source_location().end.column + 1);
	spdlog::debug("{}  keys:", indent);
	std::string new_indent = indent + std::string(6, ' ');
	for (const auto &el : m_keys) {
		if (!el) {
			spdlog::debug("{}None", new_indent);
		} else {
			el->print_node(new_indent);
		}
	}
	spdlog::debug("{}  values:", indent);
	for (const auto &el : m_values) { el->print_node(new_indent); }
}

void Set::print_this_node(const std::string &indent) const
{
	spdlog::debug("{}Set [{}:{}-{}:{}]",
		indent,
		source_location().start.row + 1,
		source_location().start.column + 1,
		source_location().end.row + 1,
		source_location().end.column + 1);
	spdlog::debug("{}  context: {}", indent, static_cast<int>(m_ctx));
	spdlog::debug("{}  elements:", indent);
	std::string new_indent = indent + std::string(6, ' ');
	for (const auto &el : m_elements) { el->print_node(new_indent); }
}

void Name::print_this_node(const std::string &indent) const
{
	spdlog::debug("{}Name [{}:{}-{}:{}]",
		indent,
		source_location().start.row + 1,
		source_location().start.column + 1,
		source_location().end.row + 1,
		source_location().end.column + 1);
	spdlog::debug("{}  - id: \"{}\"", indent, m_id[0]);
	spdlog::debug("{}  - context_type: {}", indent, static_cast<int>(m_ctx));
}

void Assign::print_this_node(const std::string &indent) const
{
	spdlog::debug("{}Assign [{}:{}-{}:{}]",
		indent,
		source_location().start.row + 1,
		source_location().start.column + 1,
		source_location().end.row + 1,
		source_location().end.column + 1);
	spdlog::debug("{}  - targets:", indent);
	std::string new_indent = indent + std::string(6, ' ');
	for (const auto &t : m_targets) { t->print_node(new_indent); }
	spdlog::debug("{}  - value:", indent);
	m_value->print_node(new_indent);
	spdlog::debug("{}  - comment type: {}", indent, m_type_comment);
}

void BinaryExpr::print_this_node(const std::string &indent) const
{
	spdlog::debug("{}BinaryOp [{}:{}-{}:{}]",
		indent,
		source_location().start.row + 1,
		source_location().start.column + 1,
		source_location().end.row + 1,
		source_location().end.column + 1);
	spdlog::debug("{}  - op_type: {}", indent, stringify_binary_op(m_op_type));
	spdlog::debug("{}  - lhs:", indent);
	std::string new_indent = indent + std::string(6, ' ');
	m_lhs->print_node(new_indent);
	spdlog::debug("{}  - rhs:", indent);
	m_rhs->print_node(new_indent);
}

void AugAssign::print_this_node(const std::string &indent) const
{
	spdlog::debug("{}AugAssign [{}:{}-{}:{}]",
		indent,
		source_location().start.row + 1,
		source_location().start.column + 1,
		source_location().end.row + 1,
		source_location().end.column + 1);
	std::string new_indent = indent + std::string(6, ' ');
	spdlog::debug("{}  - target:", indent);
	m_target->print_node(new_indent);
	spdlog::debug("{}  - op: {}", indent, stringify_binary_op(m_op));
	spdlog::debug("{}  - value:", indent);
	m_value->print_node(new_indent);
}

void Return::print_this_node(const std::string &indent) const
{
	spdlog::debug("{}Return [{}:{}-{}:{}]",
		indent,
		source_location().start.row + 1,
		source_location().start.column + 1,
		source_location().end.row + 1,
		source_location().end.column + 1);
	spdlog::debug("{}  - value: {}", indent, m_value ? "" : "null");
	std::string new_indent = indent + std::string(6, ' ');
	if (m_value) { m_value->print_node(new_indent); }
}

void Yield::print_this_node(const std::string &indent) const
{
	spdlog::debug("{}Yield [{}:{}-{}:{}]",
		indent,
		source_location().start.row + 1,
		source_location().start.column + 1,
		source_location().end.row + 1,
		source_location().end.column + 1);
	spdlog::debug("{}  - value:", indent);
	std::string new_indent = indent + std::string(6, ' ');
	m_value->print_node(new_indent);
}

void YieldFrom::print_this_node(const std::string &indent) const
{
	spdlog::debug("{}YieldFrom [{}:{}-{}:{}]",
		indent,
		source_location().start.row + 1,
		source_location().start.column + 1,
		source_location().end.row + 1,
		source_location().end.column + 1);
	spdlog::debug("{}  - value:", indent);
	std::string new_indent = indent + std::string(6, ' ');
	m_value->print_node(new_indent);
}

void Argument::print_this_node(const std::string &indent) const
{
	spdlog::debug("{}Argument [{}:{}-{}:{}]",
		indent,
		source_location().start.row + 1,
		source_location().start.column + 1,
		source_location().end.row + 1,
		source_location().end.column + 1);
	spdlog::debug("{}  - arg: {}", indent, m_arg);
	if (m_annotation) {
		spdlog::debug("{}  - annotation:", indent);
		std::string new_indent = indent + std::string(6, ' ');
		m_annotation->print_node(new_indent);
	} else {
		spdlog::debug("{}  - annotation: None", indent);
	}
	spdlog::debug("{}  - type_comment: {}", indent, m_type_comment);
}

std::vector<std::string> Arguments::argument_names() const
{
	std::vector<std::string> arg_names;
	for (const auto &arg : m_posonlyargs) { arg_names.push_back(arg->name()); }
	for (const auto &arg : m_args) { arg_names.push_back(arg->name()); }
	return arg_names;
}

std::vector<std::string> Arguments::kw_only_argument_names() const
{
	std::vector<std::string> arg_names;
	for (const auto &arg : m_kwonlyargs) { arg_names.push_back(arg->name()); }
	return arg_names;
}


void Arguments::print_this_node(const std::string &indent) const
{
	spdlog::debug("{}Arguments [{}:{}-{}:{}]",
		indent,
		source_location().start.row + 1,
		source_location().start.column + 1,
		source_location().end.row + 1,
		source_location().end.column + 1);
	std::string new_indent = indent + std::string(6, ' ');

	spdlog::debug("{}  - posonlyarg:", indent);
	for (const auto &arg : m_posonlyargs) { arg->print_node(new_indent); }
	spdlog::debug("{}  - args:", indent);
	for (const auto &arg : m_args) { arg->print_node(new_indent); }
	spdlog::debug("{}  - vararg:", indent);
	if (m_vararg) { m_vararg->print_node(new_indent); }
	spdlog::debug("{}  - kwonlyargs:", indent);
	for (const auto &kwarg : m_kwonlyargs) { kwarg->print_node(new_indent); }
	spdlog::debug("{}  - kw_defaults:", indent);
	for (const auto &arg : m_kw_defaults) {
		if (arg)
			arg->print_node(new_indent);
		else
			spdlog::debug("{}null", new_indent);
	}
	spdlog::debug("{}  - kwarg:", indent);
	if (m_kwarg) { m_kwarg->print_node(new_indent); }
	spdlog::debug("{}  - defaults:", indent);
	for (const auto &arg : m_defaults) {
		if (arg)
			arg->print_node(new_indent);
		else
			spdlog::debug("{}null", new_indent);
	}
}

void FunctionDefinition::print_this_node(const std::string &indent) const
{
	spdlog::debug("{}FunctionDefinition [{}:{}-{}:{}]",
		indent,
		source_location().start.row + 1,
		source_location().start.column + 1,
		source_location().end.row + 1,
		source_location().end.column + 1);
	spdlog::debug("{}  - function_name: {}", indent, m_function_name);
	std::string new_indent = indent + std::string(6, ' ');
	spdlog::debug("{}  - args:", indent);
	m_args->print_node(new_indent);
	spdlog::debug("{}  - body:", indent);
	for (const auto &statement : m_body) { statement->print_node(new_indent); }
	spdlog::debug("{}  - decorator_list:", indent);
	for (const auto &decorator : m_decorator_list) { decorator->print_node(new_indent); }
	spdlog::debug("{}  - returns:", indent);
	if (m_returns) m_returns->print_node(new_indent);
	spdlog::debug("{}  - type_comment:{}", indent, m_type_comment);
}

void AsyncFunctionDefinition::print_this_node(const std::string &indent) const
{
	spdlog::debug("{}AsyncFunctionDefinition [{}:{}-{}:{}]",
		indent,
		source_location().start.row + 1,
		source_location().start.column + 1,
		source_location().end.row + 1,
		source_location().end.column + 1);
	spdlog::debug("{}  - function_name: {}", indent, m_function_name);
	std::string new_indent = indent + std::string(6, ' ');
	spdlog::debug("{}  - args:", indent);
	m_args->print_node(new_indent);
	spdlog::debug("{}  - body:", indent);
	for (const auto &statement : m_body) { statement->print_node(new_indent); }
	spdlog::debug("{}  - decorator_list:", indent);
	for (const auto &decorator : m_decorator_list) { decorator->print_node(new_indent); }
	spdlog::debug("{}  - returns:", indent);
	if (m_returns) m_returns->print_node(new_indent);
	spdlog::debug("{}  - type_comment:{}", indent, m_type_comment);
}

void Await::print_this_node(const std::string &indent) const
{
	spdlog::debug("{}Await [{}:{}-{}:{}]",
		indent,
		source_location().start.row + 1,
		source_location().start.column + 1,
		source_location().end.row + 1,
		source_location().end.column + 1);
	spdlog::debug("{}  - value:", indent);
	std::string new_indent = indent + std::string(6, ' ');
	m_value->print_node(new_indent);
}

void Lambda::print_this_node(const std::string &indent) const
{
	spdlog::debug("{}Lambda [{}:{}-{}:{}]",
		indent,
		source_location().start.row + 1,
		source_location().start.column + 1,
		source_location().end.row + 1,
		source_location().end.column + 1);
	std::string new_indent = indent + std::string(6, ' ');
	spdlog::debug("{}  - args:", indent);
	m_args->print_node(new_indent);
	spdlog::debug("{}  - body:", indent);
	m_body->print_node(new_indent);
}

void Keyword::print_this_node(const std::string &indent) const
{
	spdlog::debug("{}Keyword [{}:{}-{}:{}]",
		indent,
		source_location().start.row + 1,
		source_location().start.column + 1,
		source_location().end.row + 1,
		source_location().end.column + 1);
	if (m_arg.has_value()) {
		spdlog::debug("{}  - arg: {}", indent, *m_arg);
	} else {
		spdlog::debug("{}  - arg: null", indent);
	}
	spdlog::debug("{}  - value:", indent);
	std::string new_indent = indent + std::string(6, ' ');
	m_value->print_node(new_indent);
}

void ClassDefinition::print_this_node(const std::string &indent) const
{
	spdlog::debug("{}ClassDefinition [{}:{}-{}:{}]",
		indent,
		source_location().start.row + 1,
		source_location().start.column + 1,
		source_location().end.row + 1,
		source_location().end.column + 1);
	spdlog::debug("{}  - function_name: {}", indent, m_class_name);
	std::string new_indent = indent + std::string(6, ' ');
	spdlog::debug("{}  - bases:", indent);
	for (const auto &base : m_bases) { base->print_node(new_indent); }
	spdlog::debug("{}  - keywords:", indent);
	for (const auto &keyword : m_keywords) { keyword->print_node(new_indent); }
	spdlog::debug("{}  - body:", indent);
	for (const auto &statement : m_body) { statement->print_node(new_indent); }
	spdlog::debug("{}  - decorator_list:", indent);
	for (const auto &decorator : m_decorator_list) { decorator->print_node(new_indent); }
}

void Call::print_this_node(const std::string &indent) const
{
	std::string new_indent = indent + std::string(6, ' ');
	spdlog::debug("{}Call [{}:{}-{}:{}]",
		indent,
		source_location().start.row + 1,
		source_location().start.column + 1,
		source_location().end.row + 1,
		source_location().end.column + 1);
	spdlog::debug("{}  - function:", indent);
	m_function->print_node(new_indent);
	spdlog::debug("{}  - args:", indent);
	for (const auto &arg : m_args) { arg->print_node(new_indent); }
	spdlog::debug("{}  - keywords:", indent);
	for (const auto &keyword : m_keywords) { keyword->print_node(new_indent); }
}

void Module::print_this_node(const std::string &indent) const
{
	spdlog::debug("{}Module", indent);
	spdlog::debug("{}  - body:", indent);
	std::string new_indent = indent + std::string(6, ' ');
	for (const auto &el : m_body) { el->print_node(new_indent); }
}

void If::print_this_node(const std::string &indent) const
{
	spdlog::debug("{}If [{}:{}-{}:{}]",
		indent,
		source_location().start.row + 1,
		source_location().start.column + 1,
		source_location().end.row + 1,
		source_location().end.column + 1);
	std::string new_indent = indent + std::string(6, ' ');
	spdlog::debug("{}  - test:", indent);
	m_test->print_node(new_indent);
	spdlog::debug("{}  - body:", indent);
	for (const auto &el : m_body) { el->print_node(new_indent); }
	spdlog::debug("{}  - orelse:", indent);
	for (const auto &el : m_orelse) { el->print_node(new_indent); }
}

void For::print_this_node(const std::string &indent) const
{
	spdlog::debug("{}For [{}:{}-{}:{}]",
		indent,
		source_location().start.row + 1,
		source_location().start.column + 1,
		source_location().end.row + 1,
		source_location().end.column + 1);
	std::string new_indent = indent + std::string(6, ' ');
	spdlog::debug("{}  - target:", indent);
	m_target->print_node(new_indent);
	spdlog::debug("{}  - iter:", indent);
	m_iter->print_node(new_indent);
	spdlog::debug("{}  - body:", indent);
	for (const auto &el : m_body) { el->print_node(new_indent); }
	spdlog::debug("{}  - orelse:", indent);
	for (const auto &el : m_orelse) { el->print_node(new_indent); }
	spdlog::debug("{}  - type_comment:", m_type_comment);
}

void While::print_this_node(const std::string &indent) const
{
	spdlog::debug("{}While [{}:{}-{}:{}]",
		indent,
		source_location().start.row + 1,
		source_location().start.column + 1,
		source_location().end.row + 1,
		source_location().end.column + 1);
	std::string new_indent = indent + std::string(6, ' ');
	spdlog::debug("{}  - target:", indent);
	m_test->print_node(new_indent);
	spdlog::debug("{}  - body:", indent);
	for (const auto &el : m_body) { el->print_node(new_indent); }
	spdlog::debug("{}  - orelse:", indent);
}

void Compare::print_this_node(const std::string &indent) const
{
	spdlog::debug("{}Compare [{}:{}-{}:{}]",
		indent,
		source_location().start.row + 1,
		source_location().start.column + 1,
		source_location().end.row + 1,
		source_location().end.column + 1);
	std::string new_indent = indent + std::string(6, ' ');
	spdlog::debug("{}  - lhs:", indent);
	m_lhs->print_node(new_indent);
	spdlog::debug("{}  - op:", indent);
	for (size_t i = 0; i < m_ops.size(); ++i) {
		const auto op = op_type_to_string(m_ops[i]);
		spdlog::debug("{}        - {}", indent, op);
	}
	spdlog::debug("{}  - comparators:", indent);
	for (size_t i = 0; i < m_comparators.size(); ++i) {
		const auto &comparator = m_comparators[i];
		comparator->print_node(new_indent);
	}
}

void Attribute::print_this_node(const std::string &indent) const
{
	spdlog::debug("{}Attribute [{}:{}-{}:{}]",
		indent,
		source_location().start.row + 1,
		source_location().start.column + 1,
		source_location().end.row + 1,
		source_location().end.column + 1);
	std::string new_indent = indent + std::string(6, ' ');
	spdlog::debug("{}  - value:", indent);
	m_value->print_node(new_indent);
	spdlog::debug("{}  - attr: \"{}\"", indent, m_attr);
	spdlog::debug("{}  - ctx: {}", indent, static_cast<int>(m_ctx));
}

void Import::print_this_node(const std::string &indent) const
{
	spdlog::debug("{}Import [{}:{}-{}:{}]",
		indent,
		source_location().start.row + 1,
		source_location().start.column + 1,
		source_location().end.row + 1,
		source_location().end.column + 1);
	for (const auto &name : m_names) {
		spdlog::debug("{}  - alias:", indent);
		spdlog::debug("{}        asname: {}", indent, name.asname);
		spdlog::debug("{}        name: {}", indent, name.name);
	}
}

void ImportFrom::print_this_node(const std::string &indent) const
{
	spdlog::debug("{}ImportFrom [{}:{}-{}:{}]",
		indent,
		source_location().start.row + 1,
		source_location().start.column + 1,
		source_location().end.row + 1,
		source_location().end.column + 1);
	spdlog::debug("{}  - level: {}", indent, m_level);
	spdlog::debug("{}  - module: {}", indent, m_module);
	for (const auto &name : m_names) {
		spdlog::debug("{}  - alias:", indent);
		spdlog::debug("{}        asname: {}", indent, name.asname);
		spdlog::debug("{}        name: {}", indent, name.name);
	}
}

void Subscript::Index::print(const std::string &indent) const
{
	spdlog::debug("{}Index", indent);
	std::string new_indent = indent + std::string(6, ' ');
	spdlog::debug("{}  - value:", indent);
	value->print_node(new_indent);
}

void Subscript::Slice::print(const std::string &indent) const
{
	spdlog::debug("{}Slice", indent);
	std::string new_indent = indent + std::string(6, ' ');

	if (lower) {
		spdlog::debug("{}  - lower:", indent);
		lower->print_node(new_indent);
	} else {
		spdlog::debug("{}  - lower: null", indent);
	}

	if (upper) {
		spdlog::debug("{}  - upper:", indent);
		upper->print_node(new_indent);
	} else {
		spdlog::debug("{}  - upper: null", indent);
	}

	if (step) {
		spdlog::debug("{}  - step:", indent);
		step->print_node(new_indent);
	} else {
		spdlog::debug("{}  - step: null", indent);
	}
}

void Subscript::ExtSlice::print(const std::string &indent) const
{
	spdlog::debug("{}ExtSlice", indent);
	std::string new_indent = indent + std::string(6, ' ');
	for (const auto &d : dims) {
		std::visit([&new_indent](const auto &val) { val.print(new_indent); }, d);
	}
}

void Subscript::print_this_node(const std::string &indent) const
{
	spdlog::debug("{}Subscript [{}:{}-{}:{}]",
		indent,
		source_location().start.row + 1,
		source_location().start.column + 1,
		source_location().end.row + 1,
		source_location().end.column + 1);
	std::string new_indent = indent + std::string(6, ' ');
	spdlog::debug("{}  - value:", indent);
	if (m_value) { m_value->print_node(new_indent); }
	spdlog::debug("{}  - slice:", indent);
	if (m_slice) {
		std::visit([&new_indent](const auto &val) { val.print(new_indent); }, *m_slice);
	}
	spdlog::debug("{}  - ctx: {}", indent, m_ctx);
}

void Raise::print_this_node(const std::string &indent) const
{
	spdlog::debug("{}Raise [{}:{}-{}:{}]",
		indent,
		source_location().start.row + 1,
		source_location().start.column + 1,
		source_location().end.row + 1,
		source_location().end.column + 1);
	std::string new_indent = indent + std::string(6, ' ');
	if (m_exception) {
		spdlog::debug("{}  - exception:", indent);
		m_exception->print_node(new_indent);
	} else {
		spdlog::debug("{}  - exception: null", indent);
	}
	if (m_cause) {
		spdlog::debug("{}  - cause:", indent);
		m_cause->print_node(new_indent);
	} else {
		spdlog::debug("{}  - cause: null", indent);
	}
}

void ExceptHandler::print_this_node(const std::string &indent) const
{
	spdlog::debug("{}ExceptHandler [{}:{}-{}:{}]",
		indent,
		source_location().start.row + 1,
		source_location().start.column + 1,
		source_location().end.row + 1,
		source_location().end.column + 1);
	std::string new_indent = indent + std::string(6, ' ');
	if (m_type) {
		spdlog::debug("{}  - type:", indent);
		m_type->print_node(new_indent);
	} else {
		spdlog::debug("{}  - type: null", indent);
	}
	spdlog::debug("{}  - name: {}", indent, m_name);
	spdlog::debug("{}  - body:", indent);
	for (const auto &node : m_body) { node->print_node(new_indent); }
}

void Try::print_this_node(const std::string &indent) const
{
	spdlog::debug("{}Try [{}:{}-{}:{}]",
		indent,
		source_location().start.row + 1,
		source_location().start.column + 1,
		source_location().end.row + 1,
		source_location().end.column + 1);
	std::string new_indent = indent + std::string(6, ' ');
	spdlog::debug("{}  - body:", indent);
	for (const auto &node : m_body) { node->print_node(new_indent); }
	spdlog::debug("{}  - handlers:", indent);
	for (const auto &node : m_handlers) { node->print_node(new_indent); }
	spdlog::debug("{}  - orelse:", indent);
	for (const auto &node : m_orelse) { node->print_node(new_indent); }
	spdlog::debug("{}  - finalbody:", indent);
	for (const auto &node : m_finalbody) { node->print_node(new_indent); }
}

void Assert::print_this_node(const std::string &indent) const
{
	spdlog::debug("{}Assert [{}:{}-{}:{}]",
		indent,
		source_location().start.row + 1,
		source_location().start.column + 1,
		source_location().end.row + 1,
		source_location().end.column + 1);
	std::string new_indent = indent + std::string(6, ' ');
	spdlog::debug("{}  - test:", indent);
	m_test->print_node(new_indent);
	if (m_msg) {
		spdlog::debug("{}  - message:", indent);
		m_msg->print_node(new_indent);
	} else {
		spdlog::debug("{}  - message: null", indent);
	}
}

void UnaryExpr::print_this_node(const std::string &indent) const
{
	spdlog::debug("{}UnaryExpr [{}:{}-{}:{}]",
		indent,
		source_location().start.row + 1,
		source_location().start.column + 1,
		source_location().end.row + 1,
		source_location().end.column + 1);
	std::string new_indent = indent + std::string(6, ' ');
	spdlog::debug("{}  - op_type: {}", indent, stringify_unary_op(m_op_type));
	m_operand->print_node(new_indent);
}

void BoolOp::print_this_node(const std::string &indent) const
{
	spdlog::debug("{}BoolOp [{}:{}-{}:{}]",
		indent,
		source_location().start.row + 1,
		source_location().start.column + 1,
		source_location().end.row + 1,
		source_location().end.column + 1);
	std::string new_indent = indent + std::string(6, ' ');
	spdlog::debug("{}  - op_type: {}", indent, op_type_to_string(m_op));
	spdlog::debug("{}Values:", indent);
	for (const auto &value : m_values) { value->print_node(new_indent); }
}

void Pass::print_this_node(const std::string &indent) const { spdlog::debug("{}Pass", indent); }

void Continue::print_this_node(const std::string &indent) const
{
	spdlog::debug("{}Continue", indent);
}

void Break::print_this_node(const std::string &indent) const { spdlog::debug("{}Break", indent); }

void Global::print_this_node(const std::string &indent) const
{
	spdlog::debug("{}Global [{}:{}-{}:{}]",
		indent,
		source_location().start.row + 1,
		source_location().start.column + 1,
		source_location().end.row + 1,
		source_location().end.column + 1);
	std::string new_indent = indent + std::string(6, ' ');
	for (const auto &name : m_names) { spdlog::debug("{} {}", new_indent, name); }
}

void NonLocal::print_this_node(const std::string &indent) const
{
	spdlog::debug("{}NonLocal [{}:{}-{}:{}]",
		indent,
		source_location().start.row + 1,
		source_location().start.column + 1,
		source_location().end.row + 1,
		source_location().end.column + 1);
	std::string new_indent = indent + std::string(6, ' ');
	for (const auto &name : m_names) { spdlog::debug("{} {}", new_indent, name); }
}

void Delete::print_this_node(const std::string &indent) const
{
	spdlog::debug("{}Delete [{}:{}-{}:{}]",
		indent,
		source_location().start.row + 1,
		source_location().start.column + 1,
		source_location().end.row + 1,
		source_location().end.column + 1);
	spdlog::debug("{}  - targets", indent);
	std::string new_indent = indent + std::string(6, ' ');
	for (const auto &target : m_targets) { target->print_node(new_indent); }
}

void With::print_this_node(const std::string &indent) const
{
	spdlog::debug("{}With [{}:{}-{}:{}]",
		indent,
		source_location().start.row + 1,
		source_location().start.column + 1,
		source_location().end.row + 1,
		source_location().end.column + 1);
	spdlog::debug("{}  - items", indent);
	std::string new_indent = indent + std::string(6, ' ');
	for (const auto &item : m_items) { item->print_node(new_indent); }
	spdlog::debug("{}  - body", indent);
	for (const auto &statement : m_body) { statement->print_node(new_indent); }
	spdlog::debug("{}  - type_comment: ", indent, m_type_comment);
}

void WithItem::print_this_node(const std::string &indent) const
{
	spdlog::debug("{}WithItem [{}:{}-{}:{}]",
		indent,
		source_location().start.row + 1,
		source_location().start.column + 1,
		source_location().end.row + 1,
		source_location().end.column + 1);
	std::string new_indent = indent + std::string(6, ' ');
	spdlog::debug("{}  - context_expr", indent);
	m_context_expr->print_node(new_indent);
	spdlog::debug("{}  - optional_vars: ", indent);
	if (m_optional_vars) { m_optional_vars->print_node(new_indent); }
}

void IfExpr::print_this_node(const std::string &indent) const
{
	spdlog::debug("{}IfExpr [{}:{}-{}:{}]",
		indent,
		source_location().start.row + 1,
		source_location().start.column + 1,
		source_location().end.row + 1,
		source_location().end.column + 1);
	std::string new_indent = indent + std::string(6, ' ');
	spdlog::debug("{}  - test", indent);
	m_test->print_node(new_indent);
	spdlog::debug("{}  - body", indent);
	m_body->print_node(new_indent);
	spdlog::debug("{}  - orelse", indent);
	m_orelse->print_node(new_indent);
}

void Starred::print_this_node(const std::string &indent) const
{
	spdlog::debug("{}Starred [{}:{}-{}:{}]",
		indent,
		source_location().start.row + 1,
		source_location().start.column + 1,
		source_location().end.row + 1,
		source_location().end.column + 1);
	std::string new_indent = indent + std::string(6, ' ');
	spdlog::debug("{}  - value", indent);
	m_value->print_node(new_indent);
	spdlog::debug("{}  - context: ", indent, m_ctx);
}

void NamedExpr::print_this_node(const std::string &indent) const
{
	spdlog::debug("{}NamedExpr [{}:{}-{}:{}]",
		indent,
		source_location().start.row + 1,
		source_location().start.column + 1,
		source_location().end.row + 1,
		source_location().end.column + 1);
	std::string new_indent = indent + std::string(6, ' ');
	spdlog::debug("{}  - target", indent);
	m_target->print_node(new_indent);
	spdlog::debug("{}  - value: ", indent);
	m_value->print_node(new_indent);
}

void JoinedStr::print_this_node(const std::string &indent) const
{
	spdlog::debug("{}JoinedStr", indent);
	std::string new_indent = indent + std::string(6, ' ');
	spdlog::debug("{}  - values: ", indent);
	for (const auto &v : m_values) { v->print_node(new_indent); }
}

void FormattedValue::print_this_node(const std::string &indent) const
{
	spdlog::debug("{}FormattedValue", indent);
	std::string new_indent = indent + std::string(6, ' ');
	spdlog::debug("{}  - value: ", indent);
	m_value->print_node(new_indent);
	spdlog::debug("{}  - conversion: ", indent, static_cast<int64_t>(m_conversion));
	spdlog::debug("{}  - format_spec: ", indent);
	if (m_format_spec) m_format_spec->print_node(new_indent);
}

void Comprehension::print_this_node(const std::string &indent) const
{
	spdlog::debug("{}Comprehension", indent);
	std::string new_indent = indent + std::string(6, ' ');
	spdlog::debug("{}  - target: ", indent);
	m_target->print_node(new_indent);
	spdlog::debug("{}  - iter: ", indent);
	m_iter->print_node(new_indent);
	spdlog::debug("{}  - ifs: ", indent);
	for (const auto &if_ : m_ifs) { if_->print_node(new_indent); }
	spdlog::debug("{}  - is_async: {}", indent, m_is_async);
}

void ListComp::print_this_node(const std::string &indent) const
{
	spdlog::debug("{}ListComp", indent);
	std::string new_indent = indent + std::string(6, ' ');
	spdlog::debug("{}  - elt: ", indent);
	m_elt->print_node(new_indent);
	spdlog::debug("{}  - generators: ", indent);
	for (const auto &generator : m_generators) { generator->print_node(new_indent); }
}

void DictComp::print_this_node(const std::string &indent) const
{
	spdlog::debug("{}DictComp", indent);
	std::string new_indent = indent + std::string(6, ' ');
	spdlog::debug("{}  - key: ", indent);
	m_key->print_node(new_indent);
	spdlog::debug("{}  - value: ", indent);
	m_value->print_node(new_indent);
	spdlog::debug("{}  - generators: ", indent);
	for (const auto &generator : m_generators) { generator->print_node(new_indent); }
}

void GeneratorExp::print_this_node(const std::string &indent) const
{
	spdlog::debug("{}GeneratorExp", indent);
	std::string new_indent = indent + std::string(6, ' ');
	spdlog::debug("{}  - elt: ", indent);
	m_elt->print_node(new_indent);
	spdlog::debug("{}  - generators: ", indent);
	for (const auto &generator : m_generators) { generator->print_node(new_indent); }
}

void SetComp::print_this_node(const std::string &indent) const
{
	spdlog::debug("{}SetComp", indent);
	std::string new_indent = indent + std::string(6, ' ');
	spdlog::debug("{}  - elt: ", indent);
	m_elt->print_node(new_indent);
	spdlog::debug("{}  - generators: ", indent);
	for (const auto &generator : m_generators) { generator->print_node(new_indent); }
}
}// namespace ast
