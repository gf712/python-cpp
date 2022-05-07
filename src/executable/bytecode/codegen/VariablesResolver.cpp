#include "VariablesResolver.hpp"
#include "executable/Mangler.hpp"

#include <filesystem>

namespace fs = std::filesystem;

using namespace ast;

namespace {
bool captured_by_closure(VariablesResolver::Visibility v)
{
	return v == VariablesResolver::Visibility::CELL || v == VariablesResolver::Visibility::FREE;
}
}// namespace


VariablesResolver::Scope *VariablesResolver::top_level_node(const std::string &name) const
{
	auto *node = &m_current_scope->get();
	Scope *top_node = nullptr;
	while (node->parent) {
		node = node->parent;
		if (!node->parent) break;
		if (node->visibility.contains(name)) {
			top_node = node;
			break;
		}
	}

	return top_node;
}

void VariablesResolver::annotate_free_and_cell_variables(const std::string &name)
{
	auto *top_node = top_level_node(name);
	auto *child = &m_current_scope->get();
	ASSERT(child);

	if (!top_node) {
		child->visibility[name] = Visibility::LOCAL;
		return;
	}
	auto *parent = child->parent;
	ASSERT(parent)

	while (child != top_node) {
		child->visibility[name] = Visibility::FREE;
		child->captures.insert(name);
		child = child->parent;
		ASSERT(child)
	}

	top_node->visibility[name] = Visibility::CELL;
}

void VariablesResolver::store(const std::string &name, SourceLocation source_location)
{
	const auto mangled_name = Mangler::default_mangler().function_mangle(
		m_current_scope->get().namespace_, name, source_location);
	(void)mangled_name;
	auto &current_scope_vars = m_current_scope->get().visibility;

	if (auto it = current_scope_vars.find(name); it != current_scope_vars.end()) { return; }
	if (m_current_scope->get().type == Scope::Type::MODULE) {
		current_scope_vars[name] = Visibility::NAME;
	} else if (m_current_scope->get().type == Scope::Type::FUNCTION) {
		current_scope_vars[name] = Visibility::LOCAL;
	} else if (m_current_scope->get().type == Scope::Type::CLOSURE) {
		// look around the parent functions to see if variable is defined there
		auto *parent = m_current_scope->get().parent;
		bool found = false;
		while (parent) {
			if (parent->type == Scope::Type::CLASS) {
				parent = parent->parent;
			} else {
				auto &visibility = parent->visibility;
				if (auto it = visibility.find(name); it != visibility.end()) {
					if (it->second == Visibility::GLOBAL) {
					} else {
						annotate_free_and_cell_variables(name);
					}
					found = true;
					break;
				}
				parent = parent->parent;
			}
		}
		if (!found) { current_scope_vars[name] = Visibility::LOCAL; }
	} else if (m_current_scope->get().type == Scope::Type::CLASS) {
		current_scope_vars[name] = Visibility::NAME;
	} else {
		TODO();
	}
}

void VariablesResolver::load(const std::string &name, SourceLocation source_location)
{
	const auto mangled_name = Mangler::default_mangler().function_mangle(
		m_current_scope->get().namespace_, name, source_location);
	(void)mangled_name;
	auto &current_scope_vars = m_current_scope->get().visibility;

	if (auto it = current_scope_vars.find(name); it != current_scope_vars.end()) { return; }

	if (m_current_scope->get().type == Scope::Type::MODULE) {
		current_scope_vars[name] = Visibility::NAME;
	} else if (m_current_scope->get().type == Scope::Type::FUNCTION) {
		current_scope_vars[name] = Visibility::GLOBAL;
	} else if (m_current_scope->get().type == Scope::Type::CLOSURE) {
		auto *parent = m_current_scope->get().parent;
		bool found = false;
		while (parent) {
			auto &visibility = parent->visibility;
			if (auto it = visibility.find(name); it != visibility.end()) {
				if (it->second == Visibility::GLOBAL) {
					current_scope_vars[name] = Visibility::GLOBAL;
				} else {
					annotate_free_and_cell_variables(name);
				}
				found = true;
				break;
			}
			parent = parent->parent;
		}
		if (!found) { current_scope_vars[name] = Visibility::GLOBAL; }
	} else if (m_current_scope->get().type == Scope::Type::CLASS) {
		current_scope_vars[name] = Visibility::NAME;
	} else {
		TODO();
	}
}


Value *VariablesResolver::visit(const Argument *node)
{
	m_current_scope->get().visibility[node->name()] = Visibility::LOCAL;
	return nullptr;
}

Value *VariablesResolver::visit(const Arguments *node)
{
	for (const auto &arg : node->args()) { arg->codegen(this); }
	for (const auto &arg : node->kwonlyargs()) { arg->codegen(this); }

	if (node->vararg()) { node->vararg()->codegen(this); }
	if (node->kwarg()) { node->kwarg()->codegen(this); }

	return nullptr;
}

Value *VariablesResolver::visit(const Attribute *node)
{
	node->value()->codegen(this);
	return nullptr;
}

Value *VariablesResolver::visit(const Assign *node)
{
	node->value()->codegen(this);
	for (const auto &target : node->targets()) { target->codegen(this); }
	return nullptr;
}

Value *VariablesResolver::visit(const Assert *node)
{
	node->test()->codegen(this);
	if (node->msg()) { node->msg()->codegen(this); }
	return nullptr;
}

Value *VariablesResolver::visit(const AugAssign *node)
{
	node->value()->codegen(this);
	node->value()->codegen(this);
	return nullptr;
}

Value *VariablesResolver::visit(const BinaryExpr *node)
{
	node->lhs()->codegen(this);
	node->rhs()->codegen(this);
	return nullptr;
}

Value *VariablesResolver::visit(const BoolOp *node)
{
	for (const auto &val : node->values()) { val->codegen(this); }
	return nullptr;
}

Value *VariablesResolver::visit(const Call *node)
{
	node->function()->codegen(this);
	for (const auto &arg : node->args()) { arg->codegen(this); }
	for (const auto &arg : node->keywords()) { arg->codegen(this); }
	return nullptr;
}

Value *VariablesResolver::visit(const ClassDefinition *node)
{
	auto caller = m_current_scope;

	for (const auto &decorator : node->decorator_list()) { decorator->codegen(this); }
	store(node->name(), node->source_location());
	for (const auto &base : node->bases()) { base->codegen(this); }
	for (const auto &kw : node->keywords()) { kw->codegen(this); }


	const std::string &class_name = Mangler::default_mangler().class_mangle(
		m_current_scope->get().namespace_, node->name(), node->source_location());

	const auto &ns = m_current_scope->get().namespace_ + "." + node->name();

	m_visibility[class_name] = std::unique_ptr<Scope>(new Scope{ .name = class_name,
		.namespace_ = ns,
		.type = Scope::Type::CLASS,
		.parent = &m_current_scope->get() });
	m_current_scope = std::ref(*m_visibility.at(class_name));

	for (const auto &statement : node->body()) {
		m_to_visit.emplace_back(*m_current_scope, statement);
	}

	m_current_scope = caller;

	return nullptr;
}

Value *VariablesResolver::visit(const Compare *node)
{
	node->lhs()->codegen(this);
	node->rhs()->codegen(this);
	return nullptr;
}

Value *VariablesResolver::visit(const Constant *) { return nullptr; }

Value *VariablesResolver::visit(const Delete *node)
{
	(void)node;
	return nullptr;
}

Value *VariablesResolver::visit(const Dict *node)
{
	ASSERT(node->keys().size() == node->values().size())
	for (size_t i = 0; i < node->keys().size(); ++i) {
		node->keys()[i]->codegen(this);
		node->values()[i]->codegen(this);
	}
	return nullptr;
}

Value *VariablesResolver::visit(const ExceptHandler *node)
{
	if (node->name().size() > 0) { load(node->name(), node->source_location()); }
	if (node->type()) { node->type()->codegen(this); }
	for (const auto &el : node->body()) { el->codegen(this); }
	return nullptr;
}

Value *VariablesResolver::visit(const For *node)
{
	node->iter()->codegen(this);
	node->target()->codegen(this);
	for (const auto &el : node->body()) { el->codegen(this); }
	for (const auto &el : node->orelse()) { el->codegen(this); }
	return nullptr;
}

Value *VariablesResolver::visit(const FunctionDefinition *node)
{
	auto caller = m_current_scope;

	for (const auto &decorator : node->decorator_list()) { decorator->codegen(this); }
	store(node->name(), node->source_location());
	if (node->returns()) { node->returns()->codegen(this); }

	const std::string &function_name = Mangler::default_mangler().function_mangle(
		m_current_scope->get().namespace_, node->name(), node->source_location());

	const auto &ns = m_current_scope->get().namespace_ + "." + node->name();

	for (const auto &default_ : node->args()->defaults()) {
		// load default values using the outer scope
		default_->codegen(this);
	}

	if (caller->get().type == Scope::Type::FUNCTION || caller->get().type == Scope::Type::CLOSURE) {
		m_visibility[function_name] = std::unique_ptr<Scope>(new Scope{ .name = function_name,
			.namespace_ = ns,
			.type = Scope::Type::CLOSURE,
			.parent = &caller->get() });
		m_current_scope = std::ref(*m_visibility.at(function_name));
		caller->get().visibility[node->name()] = Visibility::LOCAL;
	} else {
		m_visibility[function_name] = std::unique_ptr<Scope>(new Scope{ .name = function_name,
			.namespace_ = ns,
			.type = Scope::Type::FUNCTION,
			.parent = &caller->get() });
		m_current_scope = std::ref(*m_visibility.at(function_name));
		caller->get().visibility[node->name()] = Visibility::NAME;
	}

	caller->get().children.push_back(*m_current_scope);

	m_to_visit.emplace_back(*m_current_scope, node->args());

	for (const auto &statement : node->body()) {
		m_to_visit.emplace_back(*m_current_scope, statement);
	}

	m_current_scope = caller;

	return nullptr;
}

Value *VariablesResolver::visit(const Global *node)
{
	auto &visibility = m_current_scope->get().visibility;

	for (const auto &name : node->names()) {
		if (auto it = visibility.find(name); it != visibility.end()) {
			if (it->second == Visibility::NAME || it->second == Visibility::LOCAL) {
				// TODO: raise SyntaxError
				spdlog::error(
					"SyntaxError: name '{}' is assigned to before global declaration", name);
				std::abort();
			} else if (captured_by_closure(it->second)) {
				// TODO: raise SyntaxError
				spdlog::error("SyntaxError: name '{}' is nonlocal and global", name);
				std::abort();
			}
		}
		m_current_scope->get().visibility[name] = Visibility::GLOBAL;
	}
	return nullptr;
}

Value *VariablesResolver::visit(const If *node)
{
	node->test()->codegen(this);
	for (const auto &el : node->body()) { el->codegen(this); }
	for (const auto &el : node->orelse()) { el->codegen(this); }
	return nullptr;
}

Value *VariablesResolver::visit(const IfExpr *node)
{
	node->test()->codegen(this);
	node->body()->codegen(this);
	node->orelse()->codegen(this);
	return nullptr;
}

Value *VariablesResolver::visit(const Import *node)
{
	if (node->asname().has_value()) {
		store(*node->asname(), node->source_location());
	} else {
		store(node->names()[0], node->source_location());
	}
	return nullptr;
}

Value *VariablesResolver::visit(const Keyword *node)
{
	if (node->arg().has_value()) {
		load(*node->arg(), node->source_location());
	} else {
		auto name = as<Name>(node->value());
		ASSERT(name);
		ASSERT(name->ids().size() == 1);
		load(name->ids()[0], node->source_location());
	}
	// m_current_scope->get().visibility[*node->arg()] = Visibility::LOCAL;
	if (node->value()) { node->value()->codegen(this); }
	return nullptr;
}

Value *VariablesResolver::visit(const List *node)
{
	for (const auto &el : node->elements()) { el->codegen(this); }
	return nullptr;
}

Value *VariablesResolver::visit(const Module *node)
{
	const auto &module_name = fs::path(node->filename()).stem();
	const auto ns = module_name;

	m_visibility[module_name] = std::unique_ptr<Scope>(new Scope{
		.name = module_name, .namespace_ = ns, .type = Scope::Type::MODULE, .parent = nullptr });
	m_current_scope = std::ref(*m_visibility.at(module_name));

	for (const auto &statement : node->body()) {
		m_to_visit.emplace_back(*m_current_scope, statement);
	}
	while (!m_to_visit.empty()) {
		const auto &[scope, node] = m_to_visit.front();
		m_to_visit.pop_front();
		m_current_scope = scope;
		node->codegen(this);
	}
	return nullptr;
}

Value *VariablesResolver::visit(const NamedExpr *node)
{
	node->value()->codegen(this);
	node->target()->codegen(this);
	return nullptr;
}

Value *VariablesResolver::visit(const Name *node)
{
	if (node->context_type() == ContextType::LOAD) {
		for (const auto &name : node->ids()) { load(name, node->source_location()); }
	} else if (node->context_type() == ContextType::STORE) {
		for (const auto &name : node->ids()) { store(name, node->source_location()); }
	} else {
		TODO();
	}
	return nullptr;
}

Value *VariablesResolver::visit(const Pass *) { return nullptr; }

Value *VariablesResolver::visit(const Raise *node)
{
	if (node->exception()) { node->exception()->codegen(this); }
	if (node->cause()) { node->cause()->codegen(this); }
	return nullptr;
}

Value *VariablesResolver::visit(const Return *node)
{
	node->value()->codegen(this);
	return nullptr;
}

Value *VariablesResolver::visit(const Starred *node)
{
	node->value()->codegen(this);
	return nullptr;
}

Value *VariablesResolver::visit(const Subscript *node)
{
	std::visit(overloaded{ [this](const Subscript::Index &idx) { idx.value->codegen(this); },
				   [this](const Subscript::Slice &slice) {
					   if (slice.lower) slice.lower->codegen(this);
					   if (slice.upper) slice.upper->codegen(this);
					   if (slice.step) slice.step->codegen(this);
				   },
				   [this](const Subscript::ExtSlice &slice) {
					   for (const auto &s : slice.dims) {
						   if (std::holds_alternative<Subscript::Index>(s)) {
							   std::get<Subscript::Index>(s).value->codegen(this);
						   } else {
							   const auto &slice = std::get<Subscript::Slice>(s);
							   if (slice.lower) slice.lower->codegen(this);
							   if (slice.upper) slice.upper->codegen(this);
							   if (slice.step) slice.step->codegen(this);
						   }
					   }
				   } },
		node->slice());

	node->value()->codegen(this);

	return nullptr;
}

Value *VariablesResolver::visit(const Try *node)
{
	for (const auto &el : node->body()) { el->codegen(this); }
	for (const auto &el : node->handlers()) { el->codegen(this); }
	for (const auto &el : node->finalbody()) { el->codegen(this); }
	for (const auto &el : node->orelse()) { el->codegen(this); }
	return nullptr;
}

Value *VariablesResolver::visit(const Tuple *node)
{
	for (const auto &el : node->elements()) { el->codegen(this); }
	return nullptr;
}

Value *VariablesResolver::visit(const UnaryExpr *node)
{
	node->operand()->codegen(this);
	return nullptr;
}

Value *VariablesResolver::visit(const While *node)
{
	node->test()->codegen(this);
	for (const auto &el : node->body()) { el->codegen(this); }
	for (const auto &el : node->orelse()) { el->codegen(this); }
	return nullptr;
}

Value *VariablesResolver::visit(const With *node)
{
	for (const auto &item : node->items()) { item->codegen(this); }
	for (const auto &el : node->body()) { el->codegen(this); }
	return nullptr;
}

Value *VariablesResolver::visit(const WithItem *node)
{
	node->context_expr()->codegen(this);
	if (node->optional_vars()) { node->optional_vars()->codegen(this); }
	return nullptr;
}

Value *VariablesResolver::visit(const Expression *node)
{
	(void)node;
	return nullptr;
}

Value *VariablesResolver::visit(const FormattedValue *node)
{
	(void)node;
	return nullptr;
}

Value *VariablesResolver::visit(const JoinedStr *node)
{
	(void)node;
	return nullptr;
}

Value *VariablesResolver::visit(const Continue *) { return nullptr; }
