#ifndef __ist_Base_ThreadUtil_h__
#define __ist_Base_ThreadUtil_h__

#include "Types.h"
#ifdef _WIN32
#include <intrin.h>
#endif

namespace ist {

#if defined(_WIN32)
class atomic_int32
{
public:
    atomic_int32() : m_value(0) {}
    atomic_int32(int32 v) : m_value(v) {}

    int32 swap(int32 v) { InterlockedExchange(&m_value, v); }
    int32 compare_and_swap(int32 v, int32 comp) { InterlockedCompareExchange(&m_value, v, comp); }
    //int32 operator+=(int32 v) { return _InterlockedAdd(&m_value, v); } // Itanium only ‚ç‚µ‚¢
    //int32 operator-=(int32 v) { return _InterlockedAdd(&m_value,-v); }
    int32 operator&=(int32 v) { return _InterlockedAnd(&m_value, v); }
    int32 operator|=(int32 v) { return _InterlockedOr(&m_value, v); }
    int32 operator++() { return InterlockedIncrement(&m_value); }
    int32 operator--() { return InterlockedDecrement(&m_value); }
    int32 operator=(int32 v) { swap(v); return v; }
    operator int32() const { return m_value; }

private:
    volatile long m_value;
};

#elif defined(__GNUC__)

class atomic_int32
{
public:
    atomic_int32() : m_value(0) {}
    atomic_int32(int32 v) : m_value(v) {}

    int32 swap(int32 v) { return __sync_lock_test_and_set(&m_value, v); }
    int32 compare_and_swap(int32 v, int32 comp) { __sync_val_compare_and_swap(m_value, comp, v); }
    //int32 operator+=(int32 v) { return __sync_add_and_fetch(&m_value, v); }
    //int32 operator-=(int32 v) { return __sync_sub_and_fetch(&m_value, v); }
    int32 operator&=(int32 v) { return __sync_and_and_fetch(&m_value, v); }
    int32 operator|=(int32 v) { return __sync_or_and_fetch(&m_value, v); }
    int32 operator++() { return __sync_add_and_fetch(&m_value, 1); }
    int32 operator--() { return __sync_sub_and_fetch(&m_value, 1); }
    int32 operator=(int32 v) { swap(v); return v; }
    operator int32() const { return m_value; }

private:
    int32 m_value;
};

#endif // _WIN32


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

    void unlock()
    {
        m_value.swap(0);
    }

private:
    atomic_int32 m_value;
};


} // namespace ist

#endif // __ist_Base_ThreadUtil_h__
