#ifndef __ist_Base_Allocator__
#define __ist_Base_Allocator__

#include "ist/Base/Decl.h"


namespace ist {

const size_t DefaultAlignment = 16;

void BadAllocHandlerGeneric(const void* allocator_ptr);
template <typename Allocator>
inline void BadAllocHandler(const Allocator* allocator) { BadAllocHandlerGeneric(allocator); }

class IAllocator;
istInterModule IAllocator* GetDefaultAllocator();


class istInterModule IAllocator
{
public:
    virtual ~IAllocator() {}
    virtual void* allocate(size_t size, size_t align)=0;
    virtual void deallocate(void* p)=0;
};


class istInterModule HeapAllocator : public IAllocator
{
public:
    virtual void* allocate(size_t size, size_t align);
    virtual void deallocate(void* p);
};


class istInterModule StackAllocator : public IAllocator
{
public:
    StackAllocator();
    StackAllocator(size_t block_size, size_t alignment=DefaultAlignment, IAllocator *parent=GetDefaultAllocator());
    ~StackAllocator();

    void initialize(size_t block_size, size_t alignment=DefaultAlignment, IAllocator *parent=GetDefaultAllocator());
    void clear();

    virtual void* allocate(size_t size, size_t alignment);
    virtual void deallocate(void* p);

private:
    void* m_memory;
    size_t m_block_size;
    size_t m_position;
    IAllocator *m_parent;

    // non copyable
    StackAllocator(const StackAllocator&);
    StackAllocator& operator=(const StackAllocator&);
};


class istInterModule FixedAllocator : public IAllocator
{
public:
    FixedAllocator();
    FixedAllocator( size_t size_element, size_t num_element, size_t alignment=DefaultAlignment, IAllocator *parent=GetDefaultAllocator() );
    ~FixedAllocator();

    void initialize( size_t size_element, size_t num_element, size_t alignment=DefaultAlignment, IAllocator *parent=GetDefaultAllocator() );
    void* allocate();
    void defrag();

    virtual void* allocate(size_t size, size_t aligm);
    virtual void deallocate(void* p);

private:
    void *m_memory;
    void **m_unused;
    size_t m_used;

    size_t m_size_element;
    size_t m_max_element;
    size_t m_alignment;
    IAllocator* m_parent;

    // non copyable
    FixedAllocator(const FixedAllocator&);
    FixedAllocator& operator=(const FixedAllocator&);
};

} // namespace ist
#endif // __ist_Base_Allocator__
