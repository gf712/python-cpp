#pragma once


// Allocation entry point for module units that must not pull the VM in.
//
// PyObject.cppm allocates from inside TypePrototype::create<Type>, a template in
// the module interface. Including vm/VM.hpp there would put VirtualMachine and
// Heap into a py.runtime partition's global module fragment and close a
// dependency cycle. Only sizeof(T) is type-dependent, so the sizing and the GC
// bookkeeping live behind allocate_raw() in a .cpp that can see the VM, and the
// placement new stays here where T is known.
//
// Deliberately free of libstdc++: this header is included by module GMFs, and
// anything a GMF pulls in lands in the BMI, where it collides with the same
// headers a consumer #includes. Hence the compiler builtins instead of
// <cstddef>/<cstdint> and the hand-written forward instead of <utility>.
//
// The placement new is a static member of Allocator rather than a free function
// because the runtime types keep their constructors private and grant access
// with `friend class ::Heap;`. Allocator is befriended the same way.

namespace py::detail {

using size_type = __SIZE_TYPE__;

unsigned char *allocate_raw(size_type size, size_type extra_bytes);

class Allocator
{
  public:
	template<typename T, typename... Args> [[nodiscard]] static T *allocate(Args &&...args)
	{
		unsigned char *mem = allocate_raw(sizeof(T), 0);
		if (!mem) { return nullptr; }
		return new (mem) T(static_cast<Args &&>(args)...);
	}

	template<typename T, typename... Args>
	[[nodiscard]] static T *allocate_with_extra_bytes(size_type extra, Args &&...args)
	{
		unsigned char *mem = allocate_raw(sizeof(T), extra);
		if (!mem) { return nullptr; }
		// The extra bytes hold __slots__ storage, and both PyObject::visit_graph and the
		// member accessor installed by PyType test those entries against nullptr. Slab
		// memory is poisoned (0xCD when a block is created, 0xDD when a slot is freed), so
		// an unset slot left uninitialised reads as a non-null garbage pointer: the GC
		// dereferences it and the accessor hands it back instead of raising AttributeError.
		// __builtin_memset rather than std::memset - see the note above on keeping
		// libstdc++ out of this header.
		if (extra) { __builtin_memset(mem + sizeof(T), 0, extra); }
		return new (mem) T(static_cast<Args &&>(args)...);
	}
};

// Kept so existing call sites do not have to change.
template<typename T, typename... Args> [[nodiscard]] T *allocate(Args &&...args)
{
	return Allocator::allocate<T>(static_cast<Args &&>(args)...);
}

}// namespace py::detail
