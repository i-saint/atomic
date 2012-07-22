#ifndef __ist_Concurrency_SpinMutex_h__
#define __ist_Concurrency_SpinMutex_h__

#include "ist/Concurrency/Atomic.h"

namespace ist {

template<class T>
class scoped_lock
{
private:
    T &m_mutex;

public:
    scoped_lock(T &m) : m_mutex(m) { m_mutex.lock(); }
    ~scoped_lock() { m_mutex.unlock(); }
};

class spin_mutex
{
public:
    typedef scoped_lock<spin_mutex> scoped_lock;

    void lock()
    {
        while(m_value.compare_and_swap(1, 0) != 0) {}
    }

    bool try_lock()
    {
        return m_value.compare_and_swap(1, 0) == 0;
    }

    void unlock()
    {
        m_value.swap(0);
    }

private:
    atomic_int32 m_value;
};

} // namespace ist

#endif // __ist_Concurrency_SpinMutex_h__
