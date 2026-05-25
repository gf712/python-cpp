#pragma once

#include "utilities.hpp"

#include <cstddef>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace ast {

// Bump-pointer allocator with destructor tracking, owned by the Module.
//
// AST nodes are constructed via create<T>(args...) and live for the lifetime
// of the arena. Children are stored as raw pointers; the arena holds the only
// ownership, so PEG-cache aliasing during parsing is safe.
class ASTArena
	: private NonCopyable
	, private NonMoveable
{
	struct Slab
	{
		std::unique_ptr<std::byte[]> data;
		std::size_t size;
		std::size_t used;
	};

	struct Destructor
	{
		void *object;
		void (*fn)(void *);
	};

	std::vector<Slab> m_slabs;
	std::vector<Destructor> m_destructors;
	std::size_t m_next_slab_size;

	static constexpr std::size_t kInitialSlabSize = 64 * 1024;

	void grow(std::size_t at_least);
	void *allocate(std::size_t size, std::size_t alignment);

  public:
	ASTArena();
	~ASTArena();

	template<typename T, typename... Args> T *create(Args &&...args)
	{
		void *mem = allocate(sizeof(T), alignof(T));
		T *obj = ::new (mem) T(std::forward<Args>(args)...);
		if constexpr (!std::is_trivially_destructible_v<T>) {
			m_destructors.push_back(Destructor{ obj, [](void *p) { static_cast<T *>(p)->~T(); } });
		}
		return obj;
	}

	std::size_t bytes_allocated() const;
};

}// namespace ast
