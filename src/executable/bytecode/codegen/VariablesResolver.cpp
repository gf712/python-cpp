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
		if (node->symbol_map.get_visible_symbol(name).has_value()) {
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
		child->symbol_map.add_symbol(Symbol{ .name = name, .visibility = Visibility::GLOBAL });
		return;
	}
	auto *parent = child->parent;
	ASSERT(parent)

	while (child != top_node) {
		child->symbol_map.add_symbol(Symbol{ .name = name, .visibility = Visibility::FREE });
		child->captures.insert(name);
		child = child->parent;
		ASSERT(child)
	}

	ASSERT(top_node->symbol_map.get_visible_symbol(name).has_value());
	if (auto s = top_node->symbol_map.get_visible_symbol(name);
		s->get().visibility != Visibility::FREE) {
		auto source_location = s->get().source_location;
		top_node->symbol_map.delete_symbol(s->get());
		top_node->symbol_map.add_symbol(Symbol{ .name = name,
			.visibility = Visibility::CELL,
			.source_location = std::move(source_location) });
	}
}

void VariablesResolver::store(const std::string &name,
	SourceLocation source_location,
	Scope::Type type)
{
	const auto mangled_name = Mangler::default_mangler().function_mangle(
		m_current_scope->get().namespace_, name, source_location);
	(void)mangled_name;
	auto &current_scope_vars = m_current_scope->get().symbol_map;

	if (current_scope_vars.contains(name)) { return; }
	if (type == Scope::Type::MODULE) {
		current_scope_vars.add_symbol(Symbol{ .name = name,
			.visibility = Visibility::NAME,
			.source_location = std::move(source_location) });
	} else if (type == Scope::Type::FUNCTION) {
		current_scope_vars.add_symbol(Symbol{ .name = name,
			.visibility = Visibility::LOCAL,
			.source_location = std::move(source_location) });
	} else if (type == Scope::Type::CLOSURE) {
		// look around the parent functions to see if variable is defined there
		auto *parent = m_current_scope->get().parent;
		bool found = false;
		while (parent) {
			auto &visibility = parent->symbol_map;
			if (auto it = visibility.get_visible_symbol(name); it.has_value()) {
				if (it->get().visibility == Visibility::CELL) {
					annotate_free_and_cell_variables(name);
					found = true;
				} else if (it->get().visibility == Visibility::GLOBAL
						   || it->get().visibility == Visibility::FREE) {
					found = true;
				}
				break;
			}
			parent = parent->parent;
		}
		if (!found) {
			current_scope_vars.add_symbol(Symbol{ .name = name,
				.visibility = Visibility::LOCAL,
				.source_location = std::move(source_location) });
		}
	} else if (type == Scope::Type::CLASS) {
		current_scope_vars.add_symbol(Symbol{ .name = name,
			.visibility = Visibility::HIDDEN,
			.source_location = std::move(source_location) });
	} else {
		TODO();
	}
}

VariablesResolver::Scope *VariablesResolver::find_outer_scope_of_type(Scope::Type type) const
{
	auto *parent = m_current_scope->get().parent;
	while (parent) {
		if (parent->type == type) { return parent; }
		parent = parent->parent;
	}
	return nullptr;
}

void VariablesResolver::load(const std::string &name,
	SourceLocation source_location,
	Scope::Type type)
{
	const auto mangled_name = Mangler::default_mangler().function_mangle(
		m_current_scope->get().namespace_, name, source_location);
	(void)mangled_name;
	auto &current_scope_vars = m_current_scope->get().symbol_map;

	if (current_scope_vars.contains(name)) { return; }

	if (auto *scope = find_outer_scope_of_type(Scope::Type::CLASS);
		scope && (name == "__class__" || name == "super")) {
		scope->requires_class_ref = true;
		// artificially lookup __class__
		if (name == "super") {
			load("__class__", source_location, type);
		} else {
			annotate_free_and_cell_variables("__class__");
			return;
		}
	}

	if (type == Scope::Type::MODULE) {
		current_scope_vars.add_symbol(Symbol{ .name = name,
			.visibility = Visibility::NAME,
			.source_location = std::move(source_location) });
	} else if (type == Scope::Type::FUNCTION) {
		current_scope_vars.add_symbol(Symbol{ .name = name,
			.visibility = Visibility::GLOBAL,
			.source_location = std::move(source_location) });
	} else if (type == Scope::Type::CLOSURE || type == Scope::Type::CLASS) {
		auto *parent = m_current_scope->get().parent;
		bool found = false;
		while (parent) {
			auto &visibility = parent->symbol_map;
			if (auto it = visibility.get_visible_symbol(name); it.has_value()) {
				if (it->get().visibility == Visibility::GLOBAL) {
					current_scope_vars.add_symbol(Symbol{ .name = name,
						.visibility = Visibility::GLOBAL,
						.source_location = std::move(source_location) });
					found = true;
					break;
				} else if (it->get().visibility == Visibility::CELL
						   || it->get().visibility == Visibility::LOCAL) {
					annotate_free_and_cell_variables(name);
					found = true;
					break;
				}
			}
			parent = parent->parent;
		}
		if (!found) {
			current_scope_vars.add_symbol(Symbol{ .name = name,
				.visibility = Visibility::GLOBAL,
				.source_location = std::move(source_location) });
		}
	} else {
		TODO();
	}
}


Value *VariablesResolver::visit(const Argument *node)
{
	ASSERT(!m_current_scope->get().symbol_map.get_visible_symbol(node->name()).has_value());
	m_current_scope->get().symbol_map.add_symbol(Symbol{ .name = node->name(),
		.visibility = Visibility::LOCAL,
		.source_location = node->source_location() });
	return nullptr;
}

Value *VariablesResolver::visit(const Arguments *node)
{
	for (const auto &arg : node->posonlyargs()) { arg->codegen(this); }
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
	node->target()->codegen(this);
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
	for (const auto &decorator : node->decorator_list()) { decorator->codegen(this); }
	store(node->name(), node->source_location(), m_current_scope->get().type);
	for (const auto &base : node->bases()) { base->codegen(this); }
	for (const auto &kw : node->keywords()) { kw->codegen(this); }


	const std::string &class_name = Mangler::default_mangler().class_mangle(
		m_current_scope->get().namespace_, node->name(), node->source_location());

	auto ns = m_current_scope->get().namespace_ + "." + node->name();

	ASSERT(!m_visibility.contains(class_name))

	m_visibility[class_name] = std::unique_ptr<Scope>(new Scope{ .name = class_name,
		.namespace_ = std::move(ns),
		.type = Scope::Type::CLASS,
		.parent = &m_current_scope->get() });
	auto caller = m_current_scope;
	m_current_scope = std::ref(*m_visibility.at(class_name));

	m_visibility[class_name]->symbol_map.add_symbol(Symbol{ .name = "__class__",
		.visibility = Visibility::CELL,
		.source_location = node->source_location() });

	for (const auto &statement : node->body()) {
		m_to_visit.emplace_back(*m_current_scope, statement);
	}

	m_current_scope = std::move(caller);

	return nullptr;
}

Value *VariablesResolver::visit(const Compare *node)
{
	node->lhs()->codegen(this);
	for (const auto &el : node->comparators()) { el->codegen(this); }
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
		if (auto key = node->keys()[i]) { key->codegen(this); }
		auto value = node->values()[i];
		ASSERT(value);
		value->codegen(this);
	}
	return nullptr;
}

Value *VariablesResolver::visit(const ExceptHandler *node)
{
	if (node->name().size() > 0) {
		load(node->name(), node->source_location(), m_current_scope->get().type);
	}
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
	visit_function(node);
	return nullptr;
}

Value *VariablesResolver::visit(const AsyncFunctionDefinition *node)
{
	visit_function(node);
	return nullptr;
}

template<typename FunctionType> void VariablesResolver::visit_function(FunctionType *node)
{
	auto caller = m_current_scope;

	const std::string &function_name = Mangler::default_mangler().function_mangle(
		m_current_scope->get().namespace_, node->name(), node->source_location());

	for (const auto &decorator : node->decorator_list()) { decorator->codegen(this); }
	store(node->name(), node->source_location(), caller->get().type);
	if (node->returns()) { node->returns()->codegen(this); }

	auto ns = m_current_scope->get().namespace_ + "." + node->name();

	for (const auto &default_ : node->args()->defaults()) {
		// load default values using the outer scope
		default_->codegen(this);
	}

	for (const auto &default_ : node->args()->kw_defaults()) {
		// load default values using the outer scope
		if (default_) { default_->codegen(this); }
	}

	ASSERT(!m_visibility.contains(function_name));

	if (caller->get().type == Scope::Type::FUNCTION || caller->get().type == Scope::Type::CLOSURE
		|| caller->get().type == Scope::Type::CLASS) {
		m_visibility[function_name] = std::unique_ptr<Scope>(new Scope{ .name = function_name,
			.namespace_ = std::move(ns),
			.type = Scope::Type::CLOSURE,
			.parent = &caller->get() });
		m_current_scope = std::ref(*m_visibility.at(function_name));
	} else {
		m_visibility[function_name] = std::unique_ptr<Scope>(new Scope{ .name = function_name,
			.namespace_ = std::move(ns),
			.type = Scope::Type::FUNCTION,
			.parent = &caller->get() });
		m_current_scope = std::ref(*m_visibility.at(function_name));
	}

	caller->get().children.push_back(*m_current_scope);

	m_to_visit.emplace_back(*m_current_scope, node->args());

	for (const auto &statement : node->body()) {
		m_to_visit.emplace_back(*m_current_scope, statement);
	}

	m_current_scope = caller;
}

Value *VariablesResolver::visit(const Lambda *node)
{
	auto caller = m_current_scope;

	const std::string &function_name = Mangler::default_mangler().function_mangle(
		m_current_scope->get().namespace_, "<lambda>", node->source_location());

	auto ns = m_current_scope->get().namespace_ + ".<lambda>";

	for (const auto &default_ : node->args()->defaults()) {
		// load default values using the outer scope
		default_->codegen(this);
	}

	for (const auto &default_ : node->args()->kw_defaults()) {
		// load default values using the outer scope
		if (default_) { default_->codegen(this); }
	}

	ASSERT(!m_visibility.contains(function_name))

	if (caller->get().type == Scope::Type::FUNCTION || caller->get().type == Scope::Type::CLOSURE) {
		m_visibility[function_name] = std::unique_ptr<Scope>(new Scope{ .name = function_name,
			.namespace_ = std::move(ns),
			.type = Scope::Type::CLOSURE,
			.parent = &caller->get() });
		m_current_scope = std::ref(*m_visibility.at(function_name));
	} else {
		m_visibility[function_name] = std::unique_ptr<Scope>(new Scope{ .name = function_name,
			.namespace_ = std::move(ns),
			.type = Scope::Type::FUNCTION,
			.parent = &caller->get() });
		m_current_scope = std::ref(*m_visibility.at(function_name));
	}

	caller->get().children.push_back(*m_current_scope);

	m_to_visit.emplace_back(*m_current_scope, node->args());

	m_to_visit.emplace_back(*m_current_scope, node->body());

	m_current_scope = caller;

	return nullptr;
}

Value *VariablesResolver::visit(const Global *node)
{
	auto &visibility = m_current_scope->get().symbol_map;

	for (const auto &name : node->names()) {
		if (auto it = visibility.get_visible_symbol(name); it.has_value()) {
			if (it->get().visibility == Visibility::NAME
				|| it->get().visibility == Visibility::LOCAL) {
				// TODO: raise SyntaxError
				spdlog::error(
					"SyntaxError: name '{}' is assigned to before global declaration", name);
				std::abort();
			} else if (captured_by_closure(it->get().visibility)) {
				// TODO: raise SyntaxError
				spdlog::error("SyntaxError: name '{}' is nonlocal and global", name);
				std::abort();
			}
		}
		m_current_scope->get().symbol_map.add_symbol(Symbol{ .name = name,
			.visibility = Visibility::GLOBAL,
			.source_location = node->source_location() });
	}
	return nullptr;
}

Value *VariablesResolver::visit(const NonLocal *node)
{
	auto &visibility = m_current_scope->get().symbol_map;

	for (const auto &name : node->names()) {
		if (auto it = visibility.get_visible_symbol(name); it.has_value()) {
			if (it->get().visibility == Visibility::NAME
				|| it->get().visibility == Visibility::LOCAL) {
				// TODO: raise SyntaxError
				spdlog::error(
					"SyntaxError: name '{}' is assigned to before nonlocal declaration", name);
				std::abort();
			} else if (captured_by_closure(it->get().visibility)) {
				// TODO: raise SyntaxError
				spdlog::error("SyntaxError: name '{}' is nonlocal and global", name);
				std::abort();
			}
		}
		annotate_free_and_cell_variables(name);
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
	for (const auto &name : node->names()) {
		if (!name.asname.empty()) {
			store(name.asname, node->source_location(), m_current_scope->get().type);
		} else {
			auto idx = name.name.find('.');
			if (idx != std::string::npos) {
				std::string imported_name = name.name.substr(0, idx);
				store(imported_name, node->source_location(), m_current_scope->get().type);
			} else {
				store(name.name, node->source_location(), m_current_scope->get().type);
			}
		}
	}
	return nullptr;
}

Value *VariablesResolver::visit(const ImportFrom *node)
{
	for (const auto &name : node->names()) {
		if (!name.asname.empty()) {
			store(name.asname, node->source_location(), m_current_scope->get().type);
		} else {
			auto idx = name.name.find('.');
			if (idx != std::string::npos) {
				std::string imported_name = name.name.substr(idx + 1);
				store(imported_name, node->source_location(), m_current_scope->get().type);
			} else {
				store(name.name, node->source_location(), m_current_scope->get().type);
			}
		}
	}
	return nullptr;
}

Value *VariablesResolver::visit(const Keyword *node)
{
	if (node->arg().has_value()) {
		load(*node->arg(), node->source_location(), m_current_scope->get().type);
	} else {
		if (auto name = as<Name>(node->value())) {
			ASSERT(name->ids().size() == 1);
			load(name->ids()[0], node->source_location(), m_current_scope->get().type);
		} else if (auto attr = as<Attribute>(node->value())) {
			attr->value()->codegen(this);
		} else {
			TODO();
		}
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

Value *VariablesResolver::visit(const Set *node)
{
	for (const auto &el : node->elements()) { el->codegen(this); }
	return nullptr;
}

Value *VariablesResolver::visit(const Module *node)
{
	const auto &module_name = fs::path(node->filename()).stem();
	auto ns = module_name;

	ASSERT(!m_visibility.contains(module_name))

	m_visibility[module_name] = std::unique_ptr<Scope>(new Scope{ .name = module_name,
		.namespace_ = std::move(ns),
		.type = Scope::Type::MODULE,
		.parent = nullptr });
	m_current_scope = std::ref(*m_visibility.at(module_name));

	for (const auto &statement : node->body()) {
		m_to_visit.emplace_back(*m_current_scope, statement);
	}
	while (!m_to_visit.empty()) {
		const auto [scope, node] = m_to_visit.front();
		m_to_visit.pop_front();
		m_current_scope = std::move(scope);
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
		for (const auto &name : node->ids()) {
			load(name, node->source_location(), m_current_scope->get().type);
		}
	} else if (node->context_type() == ContextType::STORE) {
		for (const auto &name : node->ids()) {
			store(name, node->source_location(), m_current_scope->get().type);
		}
	} else {
		TODO();
	}
	return nullptr;
}

Value *VariablesResolver::visit(const Pass *) { return nullptr; }

Value *VariablesResolver::visit(const Break *) { return nullptr; }

Value *VariablesResolver::visit(const Raise *node)
{
	if (node->exception()) { node->exception()->codegen(this); }
	if (node->cause()) { node->cause()->codegen(this); }
	return nullptr;
}

Value *VariablesResolver::visit(const Return *node)
{
	if (node->value()) { node->value()->codegen(this); }
	return nullptr;
}

Value *VariablesResolver::visit(const Yield *node)
{
	ASSERT(m_current_scope->get().type == Scope::Type::FUNCTION
		   || m_current_scope->get().type == Scope::Type::CLOSURE)
	m_current_scope->get().is_generator = true;
	node->value()->codegen(this);
	return nullptr;
}

Value *VariablesResolver::visit(const YieldFrom *node)
{
	ASSERT(m_current_scope->get().type == Scope::Type::FUNCTION
		   || m_current_scope->get().type == Scope::Type::CLOSURE)
	m_current_scope->get().is_generator = true;
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
	node->value()->codegen(this);
	if (node->format_spec()) { node->format_spec()->codegen(this); }
	return nullptr;
}

Value *VariablesResolver::visit(const JoinedStr *node)
{
	for (const auto &value : node->values()) { value->codegen(this); }
	return nullptr;
}

Value *VariablesResolver::visit(const Comprehension *node)
{
	if (node->target()->node_type() == ASTNodeType::Name) {
		auto name = std::static_pointer_cast<Name>(node->target());
		ASSERT(name->ids().size() == 1);
		m_current_scope->get().symbol_map.add_symbol(Symbol{ .name = name->ids()[0],
			.visibility = Visibility::LOCAL,
			.source_location = node->target()->source_location() });
	} else if (auto target = as<Tuple>(node->target())) {
		for (const auto &el : target->elements()) {
			ASSERT(el->node_type() == ASTNodeType::Name);
			ASSERT(as<Name>(el)->ids().size() == 1);
			m_current_scope->get().symbol_map.add_symbol(Symbol{ .name = as<Name>(el)->ids()[0],
				.visibility = Visibility::LOCAL,
				.source_location = as<Name>(el)->source_location() });
		}
	} else {
		TODO();
	}

	for (auto &if_ : node->ifs()) { if_->codegen(this); }

	return nullptr;
}

Value *VariablesResolver::visit(const ListComp *node)
{
	auto caller = m_current_scope;

	store("<listcomp>", node->source_location(), m_current_scope->get().type);
	for (const auto &generator : node->generators()) { generator->iter()->codegen(this); }

	const std::string &function_name = Mangler::default_mangler().function_mangle(
		m_current_scope->get().namespace_, "<listcomp>", node->source_location());

	auto ns = m_current_scope->get().namespace_ + "." + "<listcomp>";

	if (caller->get().type == Scope::Type::FUNCTION || caller->get().type == Scope::Type::CLOSURE) {
		m_visibility[function_name] = std::unique_ptr<Scope>(new Scope{ .name = function_name,
			.namespace_ = std::move(ns),
			.type = Scope::Type::CLOSURE,
			.parent = &caller->get() });
		m_current_scope = std::ref(*m_visibility.at(function_name));
	} else {
		m_visibility[function_name] = std::unique_ptr<Scope>(new Scope{ .name = function_name,
			.namespace_ = std::move(ns),
			.type = Scope::Type::FUNCTION,
			.parent = &caller->get() });
		m_current_scope = std::ref(*m_visibility.at(function_name));
	}

	caller->get().children.push_back(*m_current_scope);

	for (const auto &generator : node->generators()) {
		m_to_visit.emplace_back(*m_current_scope, generator);
	}
	m_to_visit.emplace_back(*m_current_scope, node->elt());

	m_current_scope = caller;

	return nullptr;
}

Value *VariablesResolver::visit(const DictComp *node)
{
	auto caller = m_current_scope;

	store("<dictcomp>", node->source_location(), m_current_scope->get().type);
	for (const auto &generator : node->generators()) { generator->iter()->codegen(this); }

	const std::string &function_name = Mangler::default_mangler().function_mangle(
		m_current_scope->get().namespace_, "<dictcomp>", node->source_location());

	auto ns = m_current_scope->get().namespace_ + "." + "<dictcomp>";

	if (caller->get().type == Scope::Type::FUNCTION || caller->get().type == Scope::Type::CLOSURE) {
		m_visibility[function_name] = std::unique_ptr<Scope>(new Scope{ .name = function_name,
			.namespace_ = std::move(ns),
			.type = Scope::Type::CLOSURE,
			.parent = &caller->get() });
		m_current_scope = std::ref(*m_visibility.at(function_name));
	} else {
		m_visibility[function_name] = std::unique_ptr<Scope>(new Scope{ .name = function_name,
			.namespace_ = std::move(ns),
			.type = Scope::Type::FUNCTION,
			.parent = &caller->get() });
		m_current_scope = std::ref(*m_visibility.at(function_name));
	}

	caller->get().children.push_back(*m_current_scope);

	for (const auto &generator : node->generators()) {
		m_to_visit.emplace_back(*m_current_scope, generator);
	}
	m_to_visit.emplace_back(*m_current_scope, node->key());
	m_to_visit.emplace_back(*m_current_scope, node->value());

	m_current_scope = caller;

	return nullptr;
}

Value *VariablesResolver::visit(const GeneratorExp *node)
{
	auto caller = m_current_scope;

	store("<genexpr>", node->source_location(), m_current_scope->get().type);
	for (const auto &generator : node->generators()) { generator->iter()->codegen(this); }

	const std::string &function_name = Mangler::default_mangler().function_mangle(
		m_current_scope->get().namespace_, "<genexpr>", node->source_location());

	auto ns = m_current_scope->get().namespace_ + "." + "<genexpr>";

	if (caller->get().type == Scope::Type::FUNCTION || caller->get().type == Scope::Type::CLOSURE) {
		m_visibility[function_name] = std::unique_ptr<Scope>(new Scope{ .name = function_name,
			.namespace_ = std::move(ns),
			.type = Scope::Type::CLOSURE,
			.parent = &caller->get() });
		m_current_scope = std::ref(*m_visibility.at(function_name));
	} else {
		m_visibility[function_name] = std::unique_ptr<Scope>(new Scope{ .name = function_name,
			.namespace_ = std::move(ns),
			.type = Scope::Type::FUNCTION,
			.parent = &caller->get() });
		m_current_scope = std::ref(*m_visibility.at(function_name));
	}

	caller->get().children.push_back(*m_current_scope);

	for (const auto &generator : node->generators()) {
		m_to_visit.emplace_back(*m_current_scope, generator);
	}
	m_to_visit.emplace_back(*m_current_scope, node->elt());

	m_current_scope = caller;

	return nullptr;
}

Value *VariablesResolver::visit(const Await *node) { return node->value()->codegen(this); }

Value *VariablesResolver::visit(const SetComp *node)
{
	auto caller = m_current_scope;

	store("<setcomp>", node->source_location(), m_current_scope->get().type);
	for (const auto &generator : node->generators()) { generator->iter()->codegen(this); }

	const std::string &function_name = Mangler::default_mangler().function_mangle(
		m_current_scope->get().namespace_, "<setcomp>", node->source_location());

	auto ns = m_current_scope->get().namespace_ + "." + "<setcomp>";

	if (caller->get().type == Scope::Type::FUNCTION || caller->get().type == Scope::Type::CLOSURE) {
		m_visibility[function_name] = std::unique_ptr<Scope>(new Scope{ .name = function_name,
			.namespace_ = std::move(ns),
			.type = Scope::Type::CLOSURE,
			.parent = &caller->get() });
		m_current_scope = std::ref(*m_visibility.at(function_name));
	} else {
		m_visibility[function_name] = std::unique_ptr<Scope>(new Scope{ .name = function_name,
			.namespace_ = std::move(ns),
			.type = Scope::Type::FUNCTION,
			.parent = &caller->get() });
		m_current_scope = std::ref(*m_visibility.at(function_name));
	}

	caller->get().children.push_back(*m_current_scope);

	for (const auto &generator : node->generators()) {
		m_to_visit.emplace_back(*m_current_scope, generator);
	}
	m_to_visit.emplace_back(*m_current_scope, node->elt());

	m_current_scope = caller;

	return nullptr;
}

Value *VariablesResolver::visit(const Continue *) { return nullptr; }
