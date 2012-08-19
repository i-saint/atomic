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

    virtual void* allocate(size_t size, size_t align);
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


// leak check 用にアロケート時のコールスタックを stl::map で保存したいが、その map にデフォルトのアロケータが使われると無限再起してしまう。
// なので、malloc()/free() を呼ぶだけのアロケータを用意する。
class istInterModule DebugAllocator : public IAllocator
{
public:
    void* allocate(size_t size, size_t align) { return malloc(size); }
    void  deallocate(void* p) { free(p); }
};



// leak check 用にアロケート時のコールスタックを stl::map で保存したいが、その map にデフォルトのアロケータが使われると無限再起してしまう。
// なので、malloc()/free() を呼ぶだけのアロケータを用意する。

#ifdef __ist_with_EASTL__

template<class Alloc>
class _STLAllocatorAdapter
{
public:
    _STLAllocatorAdapter(const char* pName="") {}
    _STLAllocatorAdapter(const _STLAllocatorAdapter& x) {}
    _STLAllocatorAdapter(const _STLAllocatorAdapter& x, const char* pName) {}

    _STLAllocatorAdapter& operator=(const _STLAllocatorAdapter& x) { return *this=x; return *this; }

    void* allocate(size_t n, int flags = 0) { return malloc(n); }
    void* allocate(size_t n, size_t alignment, size_t offset, int flags = 0) { return malloc(n); }
    void  deallocate(void* p, size_t n) { free(p); }

    const char* get_name() const { return NULL; }
    void        set_name(const char* pName) {}
    Alloc& getActualAllocator() { return m_alloc; }

private:
    Alloc m_alloc;
};
template<class Alloc> bool operator==(const _STLAllocatorAdapter<Alloc>& a, const _STLAllocatorAdapter<Alloc>& b);
template<class Alloc> bool operator!=(const _STLAllocatorAdapter<Alloc>& a, const _STLAllocatorAdapter<Alloc>& b);

#define STLAllocatorAdapter(A, ...) _STLAllocatorAdapter<A>

#else // __ist_with_EASTL__

template<typename T, typename Alloc>
class _STLAllocatorAdapter {
public : 
    //    typedefs
    typedef T value_type;
    typedef value_type* pointer;
    typedef const value_type* const_pointer;
    typedef value_type& reference;
    typedef const value_type& const_reference;
    typedef std::size_t size_type;
    typedef std::ptrdiff_t difference_type;

public : 
    //    convert an allocator<T> to allocator<U>
    template<typename U>
    struct rebind {
        typedef _STLAllocatorAdapter<U, Alloc> other;
    };

public : 
    _STLAllocatorAdapter() {}
    _STLAllocatorAdapter(const _STLAllocatorAdapter&) {}
    template<typename U> _STLAllocatorAdapter(const _STLAllocatorAdapter<U, Alloc>&) {}
    ~_STLAllocatorAdapter() {}

    pointer address(reference r) { return &r; }
    const_pointer address(const_reference r) { return &r; }

    pointer allocate(size_type cnt, const void *p=NULL) {  return (pointer)m_alloc.allocate(cnt * sizeof(T)); }
    void deallocate(pointer p, size_type) {  m_alloc.deallocate(p); }

    size_type max_size() const { return std::numeric_limits<size_type>::max() / sizeof(T); }

    void construct(pointer p, const T& t) { new(p) T(t); }
    void destroy(pointer p) { p->~T(); }
    Alloc& getActualAllocator() { return m_alloc; }

    bool operator==(_STLAllocatorAdapter const&) { return true; }
    bool operator!=(_STLAllocatorAdapter const& a) { return !operator==(a); }

private:
    Alloc m_alloc;
};
template<class T, typename Alloc> inline bool operator==(const _STLAllocatorAdapter<T, Alloc>& l, const _STLAllocatorAdapter<T, Alloc>& r) { return (l.equals(r)); }
template<class T, typename Alloc> inline bool operator!=(const _STLAllocatorAdapter<T, Alloc>& l, const _STLAllocatorAdapter<T, Alloc>& r) { return (!(l == r)); }

#define STLAllocatorAdapter(A, V) _STLAllocatorAdapter<A, V>

#endif // __ist_with_EASTL__

} // namespace ist
#endif // __ist_Base_Allocator__
