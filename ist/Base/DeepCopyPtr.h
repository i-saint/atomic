#ifndef ist_Base_DeepCopyPtr_h
#define ist_Base_DeepCopyPtr_h

#include "ist/Config.h"

namespace ist {

template<class T, bool AllowCopy=true>
class deep_copy_ptr
{
public:
    deep_copy_ptr() : m_ptr(istNew(T)()) {}
    deep_copy_ptr(T *p) : m_ptr(p) {}
    ~deep_copy_ptr() { istDelete(m_ptr); }

    deep_copy_ptr(const deep_copy_ptr &other) : m_ptr(istNew(T)(*other.m_ptr)) {}
    deep_copy_ptr& operator=(const deep_copy_ptr &other) { *m_ptr=*other.m_ptr; }

    T&       operator*()        { return *m_ptr; }
    const T& operator*() const  { return *m_ptr; }
    T*       operator->()       { return m_ptr; }
    const T* operator->() const { return m_ptr; }

private:
    T * istRestrict m_ptr;
};

template<class T>
class deep_copy_ptr<T, false>
{
public:
    deep_copy_ptr() : m_ptr(istNew(T)()) {}
    deep_copy_ptr(T *p) : m_ptr(p) {}
    ~deep_copy_ptr() { istDelete(m_ptr); }

    deep_copy_ptr(const deep_copy_ptr &other);
    deep_copy_ptr& operator=(const deep_copy_ptr &other);

    T&       operator*()        { return *m_ptr; }
    const T& operator*() const  { return *m_ptr; }
    T*       operator->()       { return m_ptr; }
    const T* operator->() const { return m_ptr; }

private:
    T * istRestrict m_ptr;
};

} // namespace ist

#endif // ist_Base_DeepCopyPtr_h
