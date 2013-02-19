#ifndef ist_stdex_raw_vector_h
#define ist_stdex_raw_vector_h

#include "crtex.h"
#include "ist_vector.h"
#include "ist_aligned_allocator.h"


namespace ist {

template<class T, class A=aligned_allocator<T, 16> >
class raw_vector
{
public:
    typedef T value_type;
    typedef A allocator_type;
    typedef T& reference;
    typedef const T& const_reference;
    typedef T* pointer;
    typedef const T* const_pointer;
    typedef pointer iterator;
    typedef const_pointer const_iterator;
    typedef size_t size_type;

    raw_vector() : m_data(NULL), m_size(0), m_capacity(0) {}
    raw_vector(size_t n) : m_data(NULL), m_size(0), m_capacity(0) { resize(n); }
    raw_vector(const raw_vector &other) : m_data(NULL), m_size(0), m_capacity(0) { *this=other; }
    ~raw_vector() { deallocate(m_data, m_capacity); }

    size_type size() const      { return m_size; }
    size_type capacity() const  { return m_capacity; }
    bool empty() const          { return m_size==0; }
    void clear()                { m_size=0; }

    void reserve(size_type s)
    {
        if(s <= m_capacity) { return; }
        size_t old_capacity = m_capacity;
        size_t old_size     = m_size;
        pointer old_data    = m_data;
        m_data = allocate(s);
        m_capacity = s;
        istMemcpy(m_data, old_data, sizeof(value_type)*old_size);
        deallocate(old_data, old_capacity);
    }

    void resize(size_type s)
    {
        if(s > m_capacity) {
            reserve(std::max<size_type>(s, m_size*2));
        }
        m_size = s;
    }

    void resize(size_type s, const_reference v)
    {
        size_type before = m_size;
        resize(s);
        if(s > before) {
            std::fill(m_data+before, m_data+s, v);
        }
    }

    void push_back(const_reference v)
    {
        resize(m_size+1);
        back() = v;
    }

    void pop_back()
    {
        m_size = m_size==0 ? 0 : m_size-1;
    }

    void insert(iterator pos, const_reference v)
    {
        insert(pos, &v, &v+1);
    }

    void insert(iterator pos, const_iterator first, const_iterator last)
    {
        size_t pos_i = (m_data==NULL && pos==NULL) ? 0 : std::distance(m_data, pos);
        size_t num = std::distance(first, last);
        size_t gap = m_size-pos_i;
        resize(m_size+num);
        if(gap>0) {
            for(size_t i=gap-1; ; --i) {
                m_data[pos_i+num+i] = m_data[pos_i+i];
                if(i==0) { break; }
            }
        }
        istMemcpy(m_data+pos_i, first, sizeof(value_type)*num);
    }

    void erase(iterator first, iterator last)
    {
        size_t first_i = std::distance(m_data, first);
        size_t last_i = std::distance(m_data, last);
        size_t num = last_i-first_i;
        size_t remain = m_size-last_i;
        if(remain>0) {
            istMemcpy(first, last, sizeof(value_type)*remain);
        }
        m_size -= num;
    }

    reference       front()         { return m_data[0]; }
    const_reference front() const   { return m_data[0]; }
    reference       back()          { return m_data[m_size-1]; }
    const_reference back() const    { return m_data[m_size-1]; }
    iterator        begin()         { return m_data; }
    const_iterator  begin() const   { return m_data; }
    iterator        end()           { return m_data+m_size; }
    const_iterator  end() const     { return m_data+m_size; }
    reference       operator[](size_type i)         { return m_data[i]; }
    const_reference operator[](size_type i) const   { return m_data[i]; }

    raw_vector& operator=(const raw_vector &other)
    {
        resize(other.size());
        istMemcpy(m_data, other.m_data, sizeof(value_type)*m_size);
        return *this;
    }

protected:
    pointer allocate(size_type n)
    {
        return allocator_type().allocate(n);
    }

    void deallocate(pointer p, size_type n)
    {
        allocator_type().deallocate(p, n);
    }

private:
    value_type *m_data;
    size_type m_size;
    size_type m_capacity;
};

} // namespace ist

#endif // ist_stdex_raw_vector_h
