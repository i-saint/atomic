#ifndef __ist_Concurrency_Atomic_h__
#define __ist_Concurrency_Atomic_h__

#include "ist/Base.h"

namespace ist {

#if defined(istWindows)
class atomic_int32
{
public:
    atomic_int32() : m_value(0) {}
    atomic_int32(int32 v) : m_value(v) {}

    int32 swap(int32 v) { return InterlockedExchange(&m_value, v); }
    int32 compare_and_swap(int32 v, int32 comp) { return InterlockedCompareExchange(&m_value, v, comp); }
    int32 operator+=(int32 v) { return InterlockedExchangeAdd(&m_value, v); }
    int32 operator-=(int32 v) { return InterlockedExchangeAdd(&m_value,-v); }
    int32 operator&=(int32 v) { return _InterlockedAnd(&m_value, v); }
    int32 operator|=(int32 v) { return _InterlockedOr(&m_value, v); }
    int32 operator++() { return InterlockedIncrement(&m_value); }
    int32 operator--() { return InterlockedDecrement(&m_value); }
    int32 operator=(int32 v) { swap(v); return v; }
    operator int32() const { return m_value; }

private:
    volatile LONG m_value;
};

class atomic_int64
{
public:
    atomic_int64() : m_value(0) {}
    atomic_int64(int64 v) : m_value(v) {}

    int64 swap(int64 v) { return InterlockedExchange64(&m_value, v); }
    int64 compare_and_swap(int64 v, int64 comp) { return InterlockedCompareExchange64(&m_value, v, comp); }
    int64 operator+=(int64 v) { return InterlockedExchangeAdd64(&m_value, v); }
    int64 operator-=(int64 v) { return InterlockedExchangeAdd64(&m_value,-v); }
    int64 operator&=(int64 v) { return InterlockedAnd64(&m_value, v); }
    int64 operator|=(int64 v) { return InterlockedOr64(&m_value, v); }
    int64 operator++() { return InterlockedIncrement64(&m_value); }
    int64 operator--() { return InterlockedDecrement64(&m_value); }
    int64 operator=(int64 v) { swap(v); return v; }
    operator int64() const { return m_value; }

private:
    volatile LONGLONG m_value;
};

#if defined(istWin64)
typedef atomic_int64 atomic_ptr;
#elif defined(istWin32)
typedef atomic_int32 atomic_ptr;
#endif

#else

class atomic_int32
{
public:
    atomic_int32() : m_value(0) {}
    atomic_int32(int32 v) : m_value(v) {}

    int32 swap(int32 v) { return __sync_lock_test_and_set(&m_value, v); }
    int32 compare_and_swap(int32 v, int32 comp) { return __sync_val_compare_and_swap(m_value, comp, v); }
    int32 operator+=(int32 v) { return __sync_add_and_fetch(&m_value, v); }
    int32 operator-=(int32 v) { return __sync_sub_and_fetch(&m_value, v); }
    int32 operator&=(int32 v) { return __sync_and_and_fetch(&m_value, v); }
    int32 operator|=(int32 v) { return __sync_or_and_fetch(&m_value, v); }
    int32 operator++() { return __sync_add_and_fetch(&m_value, 1); }
    int32 operator--() { return __sync_sub_and_fetch(&m_value, 1); }
    int32 operator=(int32 v) { swap(v); return v; }
    operator int32() const { return m_value; }

private:
    int32 m_value;
};

typedef atomic_int32 atomic_ptr;

#endif // istWindows

} // namespace ist

#endif // __ist_Concurrency_Atomic_h__
