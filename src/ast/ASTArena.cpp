#include "ast/ASTArena.hpp"

#include <algorithm>
#include <cstdint>

namespace ast {

ASTArena::ASTArena() : m_next_slab_size(kInitialSlabSize) {}

ASTArena::~ASTArena()
{
	for (auto it = m_destructors.rbegin(); it != m_destructors.rend(); ++it) { it->fn(it->object); }
}

void ASTArena::grow(std::size_t at_least)
{
	std::size_t size = std::max(m_next_slab_size, at_least);
	m_slabs.push_back(Slab{ std::make_unique<std::byte[]>(size), size, 0 });
	m_next_slab_size = size * 2;
}

void *ASTArena::allocate(std::size_t size, std::size_t alignment)
{
	ASSERT(alignment > 0 && (alignment & (alignment - 1)) == 0);

	if (m_slabs.empty()) { grow(size + alignment); }

	for (;;) {
		Slab &slab = m_slabs.back();
		auto base = reinterpret_cast<std::uintptr_t>(slab.data.get()) + slab.used;
		const std::uintptr_t aligned = (base + alignment - 1) & ~(alignment - 1);
		const std::size_t pad = aligned - base;
		if (slab.used + pad + size <= slab.size) {
			slab.used += pad + size;
			return reinterpret_cast<void *>(aligned);
		}
		grow(size + alignment);
	}
}

std::size_t ASTArena::bytes_allocated() const
{
	std::size_t total = 0;
	for (const auto &slab : m_slabs) { total += slab.used; }
	return total;
}

}// namespace ast
