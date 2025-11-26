#include "PyMemoryView.hpp"
#include "runtime/NotImplementedError.hpp"
#include "runtime/PyBool.hpp"
#include "runtime/PyBytes.hpp"
#include "runtime/PyDict.hpp"
#include "runtime/PyFloat.hpp"
#include "runtime/PyInteger.hpp"
#include "runtime/PyList.hpp"
#include "runtime/PyObject.hpp"
#include "runtime/PyString.hpp"
#include "runtime/PyTuple.hpp"
#include "runtime/PyType.hpp"
#include "runtime/TypeError.hpp"
#include "runtime/Value.hpp"
#include "runtime/ValueError.hpp"
#include "runtime/forward.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"
#include "utilities.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <memory>
#include <string_view>

namespace py {

template<> PyMemoryView *as(PyObject *obj)
{
	if (obj->type() == types::memoryview()) { return static_cast<PyMemoryView *>(obj); }
	return nullptr;
}

template<> const PyMemoryView *as(const PyObject *obj)
{
	if (obj->type() == types::memoryview()) { return static_cast<const PyMemoryView *>(obj); }
	return nullptr;
}

namespace {
	std::optional<std::tuple<size_t, std::string_view>> get_native_size_and_format(
		std::string_view format)
	{
		if (format.front() == '@') { format = format.substr(1); }
		if (format.size() != 1) { return {}; }

		size_t size = 0;

		switch (format.front()) {
		case 'c':
		case 'b':
		case 'B': {
			size = sizeof(char);
		} break;
		case 'h':
		case 'H': {
			size = sizeof(short);
		} break;
		case 'i':
		case 'I': {
			size = sizeof(int);
		} break;
		case 'l':
		case 'L': {
			size = sizeof(long);
		} break;
		case 'q':
		case 'Q': {
			size = sizeof(long long);
		} break;
		case 'n':
		case 'N': {
			size = sizeof(size_t);
		} break;
		case 'f': {
			size = sizeof(float);
		} break;
		case 'd': {
			size = sizeof(double);
		} break;
		case '?': {
			size = sizeof(bool);
		} break;
		case 'P': {
			size = sizeof(void *);
		} break;
		default: {
			return {};
		}
		}

		return { { size, format } };
	}

	PyResult<PyObject *> unpack_single(std::byte *ptr, std::string_view format)
	{
		ASSERT(format.size() == 1);

		switch (format.front()) {
		case 'c': {
			Bytes b;
			b.b.reserve(1);
			b.b.push_back(static_cast<std::byte>(*bit_cast<char *>(ptr)));
			return PyBytes::create(std::move(b));
		} break;
		case 'b': {
			return PyInteger::create(*bit_cast<char *>(ptr));
		} break;
		case 'B': {
			return PyInteger::create(*bit_cast<unsigned char *>(ptr));
		} break;
		case 'h': {
			return PyInteger::create(*bit_cast<short *>(ptr));
		} break;
		case 'H': {
			return PyInteger::create(*bit_cast<unsigned short *>(ptr));
		} break;
		case 'i': {
			return PyInteger::create(*bit_cast<int32_t *>(ptr));
		} break;
		case 'I': {
			return PyInteger::create(*bit_cast<uint32_t *>(ptr));
		} break;
		case 'l': {
			return PyInteger::create(*bit_cast<int32_t *>(ptr));
		} break;
		case 'L': {
			return PyInteger::create(*bit_cast<uint32_t *>(ptr));
		} break;
		case 'q': {
			return PyInteger::create(*bit_cast<int64_t *>(ptr));
		} break;
		case 'Q': {
			return PyInteger::create(*bit_cast<uint64_t *>(ptr));
		} break;
		case 'n': {
			return PyInteger::create(*bit_cast<ssize_t *>(ptr));
		} break;
		case 'N': {
			return PyInteger::create(*bit_cast<size_t *>(ptr));
		} break;
		case 'f': {
			return PyFloat::create(static_cast<double>(*bit_cast<float *>(ptr)));
		} break;
		case 'd': {
			return PyFloat::create(*bit_cast<double *>(ptr));
		} break;
		case '?': {
			if (*bit_cast<bool *>(ptr)) {
				return Ok(py_true());
			} else {
				return Ok(py_false());
			}
		} break;
		case 'P': {
			return PyInteger::create(*bit_cast<size_t *>(ptr));
		} break;
		}

		return Err(not_implemented_error("memoryview: format {} not supported", format));
	}
}// namespace

PyMemoryView::PyMemoryView(PyType *type) : PyBaseObject(type)
{
	ASSERT(type == types::memoryview());
}

PyMemoryView::PyMemoryView(PyBuffer buffer)
	: PyBaseObject(types::memoryview()), m_view(std::move(buffer))
{}

PyResult<PyBuffer> PyMemoryView::create_view(PyBuffer &main_view)
{
	PyBuffer buffer{
		.buf = main_view.buf->view(),
		.obj = main_view.obj,
		.len = main_view.len,
		.itemsize = main_view.itemsize,
		.readonly = main_view.readonly,
		.format = !main_view.format.empty() ? main_view.format : "B",
		.internal = main_view.internal,
	};

	// init shape
	if (main_view.ndim == 1) {
		if (!main_view.shape.empty()) {
			buffer.shape.push_back(main_view.shape.front());
		} else {
			buffer.shape.push_back(main_view.len / main_view.itemsize);
		}
	} else if (main_view.ndim > 1) {
		buffer.shape = main_view.shape;
		if (!main_view.strides.empty()) {
			buffer.strides = main_view.strides;
		} else {
			buffer.strides.reserve(buffer.itemsize);
			std::transform(main_view.strides.begin() + 1,
				main_view.strides.end(),
				main_view.shape.begin() + 1,
				std::back_inserter(buffer.strides),
				std::multiplies<>{});
		}
	}

	// init suboffset
	buffer.suboffsets = main_view.suboffsets;

	// init flags
	// TODO

	return Ok(std::move(buffer));
}


PyResult<PyObject *> PyMemoryView::create(PyObject *object)
{
	if (auto view = as<PyMemoryView>(object)) {
		auto other_view = create_view(view->m_view);

		auto obj =
			VirtualMachine::the().heap().allocate<PyMemoryView>(std::move(other_view).unwrap());
		if (!obj) { return Err(memory_error(sizeof(PyMemoryView))); }

		obj->m_managed_buffer = view->m_managed_buffer;

		return Ok(obj);
	} else if (auto buffer = object->as_buffer(); buffer.is_ok()) {
		auto managed_buffer = std::make_shared<ManagedBuffer>();
		auto result = buffer.unwrap().getbuffer(object, managed_buffer->m_main_view, 0);

		if (result.is_err()) { return Err(result.unwrap_err()); }
		auto view = create_view(managed_buffer->m_main_view);

		auto obj = VirtualMachine::the().heap().allocate<PyMemoryView>(std::move(view).unwrap());
		if (!obj) { return Err(memory_error(sizeof(PyMemoryView))); }

		obj->m_managed_buffer = std::move(managed_buffer);

		return Ok(obj);
	}

	return Err(
		type_error("memoryview: a bytes-like object is required, not {}", object->type()->name()));
}

PyResult<PyObject *> PyMemoryView::__new__(const PyType *, PyTuple *args, PyDict *kwargs)
{
	auto result = PyArgsParser<PyObject *>::unpack_tuple(args,
		kwargs,
		"memoryview",
		std::integral_constant<size_t, 1>{},
		std::integral_constant<size_t, 1>{});

	if (result.is_err()) { return Err(result.unwrap_err()); }

	auto [object] = result.unwrap();

	return create(object);
}

PyResult<size_t> PyMemoryView::__len__() const
{
	if (m_view.ndim == 0) {
		return Ok(size_t{ 1 });
	} else {
		return Ok(m_view.shape[0]);
	}
}

PyResult<PyObject *> PyMemoryView::cast(PyTuple *args, PyDict *kwargs)
{
	ASSERT(!kwargs || kwargs->map().empty());

	auto result = PyArgsParser<PyString *, PyObject *>::unpack_tuple(args,
		kwargs,
		"memoryview.cast",
		std::integral_constant<size_t, 1>{},
		std::integral_constant<size_t, 2>{},
		nullptr);

	if (result.is_err()) { return Err(result.unwrap_err()); }

	auto [format_, shape] = result.unwrap();
	if (shape) { TODO(); }

	auto original_format = format_->value();
	auto size_and_format = get_native_size_and_format(original_format);
	if (!size_and_format.has_value()) {
		return Err(
			value_error("memoryview: destination format must be a native single character format "
						"prefixed with an optional '@'"));
	}

	auto [itemsize, format] = *size_and_format;

	auto new_view_ = create_view(m_view);
	if (new_view_.is_err()) { return Err(new_view_.unwrap_err()); }
	auto new_view = std::move(new_view_).unwrap();

	if (new_view.len % itemsize != 0) {
		return Err(type_error("memoryview: length is not a multiple of itemsize"));
	}

	new_view.itemsize = 4;
	new_view.format = format;
	new_view.ndim = 1;
	new_view.shape = { new_view.len / new_view.itemsize };
	new_view.strides = { new_view.itemsize };

	auto obj = VirtualMachine::the().heap().allocate<PyMemoryView>(std::move(new_view));
	if (!obj) { return Err(memory_error(sizeof(PyMemoryView))); }

	obj->m_managed_buffer = m_managed_buffer;

	return Ok(obj);
}

PyResult<PyObject *> PyMemoryView::tolist()
{
	auto result_ = PyList::create();
	if (result_.is_err()) { return result_; }
	auto *result = result_.unwrap();

	auto *buffer = static_cast<std::byte *>(m_view.buf->get_buffer());
	for (int64_t i = 0; i < m_view.shape[0]; ++i, buffer += m_view.strides[0]) {
		auto el = unpack_single(buffer, m_view.format);
		if (el.is_err()) { return el; }
		result->elements().push_back(el.unwrap());
	}

	return Ok(result);
}

PyResult<PyObject *> PyMemoryView::__repr__() const { return PyString::create(to_string()); }

namespace {
	std::once_flag memoryview_flag;

	std::unique_ptr<TypePrototype> register_memoryview()
	{
		return std::move(klass<PyMemoryView>("memoryview")
							 .def("cast", &PyMemoryView::cast)
							 .def("tolist", &PyMemoryView::tolist)
							 .property_readonly("itemsize",
								 [](PyMemoryView *view) -> PyResult<PyObject *> {
									 return PyInteger::create(view->itemsize());
								 })
							 .type);
	}
}// namespace

std::function<std::unique_ptr<TypePrototype>()> PyMemoryView::type_factory()
{
	return []() {
		static std::unique_ptr<TypePrototype> type = nullptr;
		std::call_once(memoryview_flag, []() { type = register_memoryview(); });
		return std::move(type);
	};
}

PyType *PyMemoryView::static_type() const { return types::memoryview(); }

void PyMemoryView::visit_graph(Visitor &visitor)
{
	PyObject::visit_graph(visitor);
	if (m_view.obj && !m_view.readonly) { visitor.visit(*m_view.obj); }
	if (m_managed_buffer && m_managed_buffer->m_main_view.obj
		&& !m_managed_buffer->m_main_view.readonly) {
		visitor.visit(*m_managed_buffer->m_main_view.obj);
	}
}

std::string PyMemoryView::to_string() const
{
	return fmt::format("<memory at {}>", static_cast<const void *>(this));
}

}// namespace py
