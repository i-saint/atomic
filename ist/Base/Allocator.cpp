#include "istPCH.h"
#include "ist/Base/New.h"
#include "ist/Base/Allocator.h"

namespace ist {

void BadAllocHandlerGeneric(const void* allocator_ptr)
{
    DebugBreak();
}

IAllocator* GetDefaultAllocator()
{
    static HeapAllocator s_alloc;
    return &s_alloc;
}



void* HeapAllocator::allocate(size_t size, size_t align)
{
    return istRawMalloc(size, align);
}

void HeapAllocator::deallocate(void* p)
{
    istRawFree(p);
}



StackAllocator::StackAllocator()
    : m_memory(NULL)
    , m_block_size(0)
    , m_position(0)
    , m_parent(NULL)
{}

StackAllocator::StackAllocator(size_t block_size, size_t alignment, IAllocator *parent)
{
    initialize(block_size, alignment, parent);
}

StackAllocator::~StackAllocator()
{
    if(m_parent) {
        m_parent->deallocate(m_memory);
    }
}

void StackAllocator::initialize(size_t block_size, size_t alignment, IAllocator *parent)
{
    m_block_size    = block_size;
    m_position      = 0;
    m_parent        = parent;
    m_memory        = parent->allocate(block_size, alignment);
}

void* StackAllocator::allocate(size_t size, size_t alignment)
{
    alignment &= ~7;
    size_t alignment_mask = alignment-1;
    size_t position = m_position;
    size_t begin_pos = (position + alignment_mask) & alignment_mask;
    size_t gap = begin_pos - position;
    size_t real_size = size+gap;

    if(position+size > m_block_size) {
        BadAllocHandler(this);
    }
    char* p = (char*)m_memory+position+gap;
    m_position += real_size;
    return p;
}

void StackAllocator::deallocate(void* p)
{
    // do nothing
}

void StackAllocator::clear()
{
    m_position = 0;
}



FixedAllocator::FixedAllocator()
    : m_memory(NULL)
    , m_unused(NULL)
    , m_used(0)
    , m_size_element(0)
    , m_max_element(0)
    , m_alignment(0)
    , m_parent(NULL)
{
}

FixedAllocator::FixedAllocator( size_t size_element, size_t num_element, size_t alignment, IAllocator *parent )
{
    initialize(size_element, num_element, alignment, parent);
}

FixedAllocator::~FixedAllocator()
{
    if(m_parent) {
        m_parent->deallocate(m_memory);
        m_parent->deallocate(m_unused);
    }
}

void FixedAllocator::initialize( size_t size_element, size_t num_element, size_t alignment, IAllocator *parent )
{
    m_memory = NULL;
    m_used = NULL;
    m_size_element = size_element;
    m_max_element = num_element;
    m_alignment = alignment;
    m_parent = parent;

    void** unused = (void**)parent->allocate(sizeof(void*)*num_element, DefaultAlignment);
    void* mem = parent->allocate(size_element*num_element, alignment);
    for(size_t i=0; i<num_element; ++i) {
        unused[i] = (char*)mem + (size_element*i);
    }
    m_unused = unused;
    m_memory = mem;
}

void* FixedAllocator::allocate()
{
    if(m_used==m_max_element) {
        BadAllocHandler(this);
    }
    return m_unused[m_used++];
}

void FixedAllocator::defrag()
{
    stl::stable_sort(m_unused+m_used, m_unused+m_max_element);
}

void* FixedAllocator::allocate(size_t size, size_t align)
{
    return allocate();
}

void FixedAllocator::deallocate(void* p)
{
    m_unused[--m_used] = p;
}

} // namespace ist
