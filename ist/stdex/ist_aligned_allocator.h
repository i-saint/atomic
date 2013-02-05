#ifndef ist_stdex_aligned_allocator_h
#define ist_stdex_aligned_allocator_h

#include "crtex.h"

namespace ist {

template<typename T, size_t Align=16>
class aligned_allocator {
public : 
    //    typedefs
    typedef T value_type;
    typedef value_type* pointer;
    typedef const value_type* const_pointer;
    typedef value_type& reference;
    typedef const value_type& const_reference;
    typedef std::size_t size_type;
    typedef std::ptrdiff_t difference_type;
    static const size_t align = Align;

public : 
    //    convert an allocator<T> to allocator<U>
    template<typename U>
    struct rebind {
        typedef aligned_allocator<U> other;
    };

public : 
    aligned_allocator() {}
    aligned_allocator(const aligned_allocator&) {}
    template<typename U> aligned_allocator(const aligned_allocator<U, align>&) {}
    ~aligned_allocator() {}

    pointer         address(reference r)        { return &r; }
    const_pointer   address(const_reference r)  { return &r; }

    pointer allocate(size_type s, const void *p=NULL)   { return (pointer)istAlignedMalloc(sizeof(value_type)*s, align); }
    void deallocate(pointer p, size_type)               {  istAlignedFree(p); }

    size_type max_size() const { return std::numeric_limits<size_type>::max() / sizeof(T); }

    void construct(pointer p, const T& t) { new(p) T(t); }
    void destroy(pointer p) { p->~T(); }

    bool operator==(aligned_allocator const&) { return true; }
    bool operator!=(aligned_allocator const& a) { return !operator==(a); }
};
template<class T, typename Alloc> inline bool operator==(const aligned_allocator<T>& l, const aligned_allocator<T>& r) { return (l.equals(r)); }
template<class T, typename Alloc> inline bool operator!=(const aligned_allocator<T>& l, const aligned_allocator<T>& r) { return (!(l == r)); }

} // namespace ist


#endif // ist_stdex_aligned_allocator_h
