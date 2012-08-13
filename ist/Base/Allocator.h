#include <EASTL/allocator.h>
#include <EASTL/algorithm.h>
#include <EASTL/sort.h>
#include "ist/Base/Decl.h"


void bad_alloc_hander_generic(const void* allocator_ptr);

template <typename Allocator>
inline void bad_alloc_hander(const Allocator* allocator)
{
    bad_alloc_hander_generic(allocator);
}


namespace ist {

template<class ParentAllocatorType=stl::allocator, bool AllowOverflow=false>
class istInterModule stack_allocator
{
public:
    typedef ParentAllocatorType parent_allocator;

private:
    void* m_memory;
    size_t m_block_size;
    size_t m_position;
    parent_allocator *m_parent;


    // non copyable
    stack_allocator(const stack_allocator&);
    stack_allocator& operator=(const stack_allocator&);

public:
    explicit stack_allocator(const char* name=NULL)
        : m_memory(NULL)
        , m_block_size(0)
        , m_position(0)
        , m_parent(NULL)
    {}

    stack_allocator(size_t block_size, size_t alignment=8, size_t offset=0, parent_allocator& parent=get_default_allocator<parent_allocator>(NULL))
    {
        initialize(block_size, alignment, offset, parent);
    }

    ~stack_allocator()
    {
        if(m_parent) {
            m_parent->deallocate(m_memory, m_block_size);
        }
    }

    void initialize(size_t block_size, size_t alignment=8, size_t offset=0, parent_allocator& parent=get_default_allocator<parent_allocator>(NULL))
    {
        m_block_size    = block_size;
        m_position      = 0;
        m_parent        = &parent;
        m_memory        = parent.allocate(block_size, alignment, offset);
    }

    void* allocate(size_t size)
    {
        if(m_position>m_block_size) {
            bad_alloc_hander(this);
        }
        void* ret = (char*)m_memory+m_position;
        m_position += size;
        return ret;
    }

    void* allocate(size_t size, size_t alignment, size_t offset, int flags = 0)
    {
        alignment &= ~7;
        size_t alignment_mask = alignment-1;
        size_t position = m_position;
        size_t begin_pos = (position + alignment_mask) & alignment_mask;
        size_t gap = begin_pos - position;
        size_t real_size = size+gap;

        if(position+size > m_block_size) {
            bad_alloc_hander(this);
        }
        char* p = (char*)m_memory+position+gap;
        m_position += real_size;
        return p+offset;
    }

    void deallocate(void* p, size_t)
    {
        // do nothing
    }

    void clear()
    {
        m_position = 0;
    }
};


template<class ParentAllocatorType=stl::allocator>
class istInterModule pool_allocator
{
public:
    typedef ParentAllocatorType parent_allocator;

private:
    void *m_memory;
    void **m_unused;
    size_t m_used;

    size_t m_size_element;
    size_t m_max_element;
    size_t m_alignment;
    parent_allocator* m_parent;

    // non copyable
    pool_allocator(const pool_allocator&);
    pool_allocator& operator=(const pool_allocator&);

public:
    explicit pool_allocator(const char* name=NULL)
        : m_memory(NULL)
        , m_unused(NULL)
        , m_used(0)
        , m_size_element(0)
        , m_max_element(0)
        , m_alignment(0)
        , m_parent(NULL)
    {
    }

    pool_allocator( size_t size_element, size_t num_element, size_t alignment, parent_allocator *parent )
    {
        initialize(size_element, num_element, alignment, parent);
    }

    void initialize( size_t size_element, size_t num_element, size_t alignment, parent_allocator *parent )
    {
        m_memory = NULL;
        m_used = NULL;
        m_size_element = size_element;
        m_max_element = num_element;
        m_alignment = alignment;
        m_parent = parent;

        void** unused = (void**)parent->allocate(sizeof(void*)*num_element);
        void* mem = parent->allocate(size_element*num_element, alignment, 0);
        for(size_t i=0; i<num_element; ++i) {
            unused[i] = (char*)mem + (size_element*i);
        }
        m_unused = unused;
        m_memory = mem;
    }

    ~pool_allocator()
    {
        if(m_parent) {
            m_parent->deallocate(m_memory, m_size_element*m_max_element);
            m_parent->deallocate(m_unused, sizeof(void*)*m_max_element);
        }
    }

    void* allocate()
    {
        if(m_used==m_max_element) {
            bad_alloc_hander(this);
        }
        return m_unused[m_used++];
    }

    // 互換性のため用意。引数は全部無視
    void* allocate(size_t size)
    {
        return allocate();
    }
    // 同上
    void* allocate(size_t size, size_t alignment, size_t offset, int flags = 0)
    {
        return allocate();
    }

    void deallocate(void* p, size_t)
    {
        m_unused[--m_used] = p;
    }


    // 非互換機能
    void defrag()
    {
        stl::stable_sort(m_unused+m_used, m_unused+m_max_element);
    }
};

} // namespace ist

