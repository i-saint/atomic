#ifndef ist_stdex_sorted_vector_h
#define ist_stdex_sorted_vector_h

#include "ist/Base/Serialize.h"
#include "crtex.h"
#include "ist_vector.h"
#include "ist_aligned_allocator.h"

namespace ist {

template<class T, class Compare=std::less<T>, class Alloc=ist::aligned_allocator<T> >
class sorted_vector_set
{
public:
    typedef ist::vector<T, Alloc> container;
    typedef Compare compare;
    typedef typename container::value_type      value_type;
    typedef typename container::allocator_type  allocator_type;
    typedef typename container::reference       reference;
    typedef typename container::const_reference const_reference;
    typedef typename container::pointer         pointer;
    typedef typename container::const_pointer   const_pointer;
    typedef typename container::iterator        iterator;
    typedef typename container::const_iterator  const_iterator;
    typedef typename container::reverse_iterator        reverse_iterator;
    typedef typename container::const_reverse_iterator  const_reverse_iterator;
    typedef typename container::size_type       size_type;

    sorted_vector_set() {}
    sorted_vector_set(const sorted_vector_set &v) { m_cont=v.m_cont; }
    sorted_vector_set& operator=(const sorted_vector_set &v) { m_cont=v.m_cont; return *this; }

    size_t          size() const    { return m_cont.size(); }
    bool            empty() const   { return m_cont.empty(); }

    iterator                begin()         { return m_cont.begin(); }
    const_iterator          begin() const   { return m_cont.begin(); }
    iterator                end()           { return m_cont.end(); }
    const_iterator          end() const     { return m_cont.end(); }
    reverse_iterator        rbegin()        { return m_cont.rbegin(); }
    const_reverse_iterator  rbegin() const  { return m_cont.rbegin(); }
    reverse_iterator        rend()          { return m_cont.rend(); }
    const_reverse_iterator  rend() const    { return m_cont.rend(); }

    reference       operator[](size_type i)         { return m_cont[i]; }
    const_reference operator[](size_type i) const   { return m_cont[i]; }

    void reserve(size_t n)
    {
        m_cont.reserve(n);
    }

    iterator find(const_reference v)
    {
        iterator p = std::lower_bound(begin(), end(), v);
        return (p!=end() && *p==v) ? p : end();
    }

    const_iterator find(const_reference v) const
    {
        const_iterator p = std::lower_bound(begin(), end(), v);
        return (p!=end() && *p==v) ? p : end();
    }

    std::pair<iterator, bool> insert(const_reference v)
    {
        iterator p = std::lower_bound(begin(), end(), v);
        if(p!=end() && *p==v) {
            return std::make_pair(p, false);
        }
        else {
            iterator r = m_cont.insert(p, v);
            return std::make_pair(r, true);
        }
    }

    void erase(iterator p)
    {
        m_cont.erase(p);
    }

    void erase(iterator first, iterator last)
    {
        m_cont.erase(first, last);
    }

    size_t erase(const_reference v)
    {
        iterator p = find(v);
        if(p==end()) {
            return 0;
        }
        else {
            erase(p);
            return 1;
        }
    }

    void clear()
    {
        m_cont.clear();
    }

    void swap(const sorted_vector_set &v)
    {
        m_cont.swap(v);
    }

private:
    container m_cont;
};

} // namespace ist

#endif // ist_stdex_sorted_vector_h
