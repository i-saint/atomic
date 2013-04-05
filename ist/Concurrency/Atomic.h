#ifndef ist_Concurrency_Atomic_h
#define ist_Concurrency_Atomic_h

#include "../Config.h"

#if defined(ist_env_Windows)
extern "C" {
// 64bit 版は Win64 でのみ提供。
// 戻り値は全て変更前の値

// add
int8_t  __fastcall atomic_add8( int8_t  *v, int8_t  a);
int16_t __fastcall atomic_add16(int16_t *v, int16_t a);
int32_t __fastcall atomic_add32(int32_t *v, int32_t a);
int64_t __fastcall atomic_add64(int64_t *v, int64_t a);

// swap
int8_t  __fastcall atomic_swap8( int8_t  *v, int8_t  a);
int16_t __fastcall atomic_swap16(int16_t *v, int16_t a);
int32_t __fastcall atomic_swap32(int32_t *v, int32_t a);
int64_t __fastcall atomic_swap64(int64_t *v, int64_t a);

// compare & swap
int8_t  __fastcall atomic_cas8( int8_t  *v, int8_t  v_old, int8_t  v_new);
int16_t __fastcall atomic_cas16(int16_t *v, int16_t v_old, int16_t v_new);
int32_t __fastcall atomic_cas32(int32_t *v, int32_t v_old, int32_t v_new);
int64_t __fastcall atomic_cas64(int64_t *v, int64_t v_old, int64_t v_new);
} // extern "C"
#endif // ist_env_Windows

namespace ist {

template<class T> struct atomic_traits;

#if defined(ist_env_Windows)
template<> struct atomic_traits<int8>
{
    static int8 add(int8 *a, int8 b)         { return atomic_add8(a, b);    }
    static int8 swap(int8 *a, int8 b)        { return atomic_swap8(a, b);   }
    static int8 cas(int8 *a, int8 b, int8 c) { return atomic_cas8(a, b, c); }
};
template<> struct atomic_traits<int16>
{
    static int16 add(int16 *a, int16 b)          { return atomic_add16(a, b);    }
    static int16 swap(int16 *a, int16 b)         { return atomic_swap16(a, b);   }
    static int16 cas(int16 *a, int16 b, int16 c) { return atomic_cas16(a, b, c); }
};
template<> struct atomic_traits<int32>
{
    static int32 add(int32 *a, int32 b)          { return atomic_add32(a, b);    }
    static int32 swap(int32 *a, int32 b)         { return atomic_swap32(a, b);   }
    static int32 cas(int32 *a, int32 b, int32 c) { return atomic_cas32(a, b, c); }
};
template<> struct atomic_traits<int64>
{
    static int64 add(int64 *a, int64 b)          { return atomic_add64(a, b);    }
    static int64 swap(int64 *a, int64 b)         { return atomic_swap64(a, b);   }
    static int64 cas(int64 *a, int64 b, int64 c) { return atomic_cas64(a, b, c); }
};
#else // ist_env_Windows
template<class T> struct atomic_traits
{
    static T add(T *a, T b)      { return __sync_fetch_and_add(a, b);    }
    static T swap(T *a, T b)     { return __sync_lock_test_and_set(a, b);   }
    static T cas(T *a, T b, T c) { return __sync_val_compare_and_swap(a, b, c); }
};
#endif // ist_env_Windows


template<class T>
class atomic_int
{
typedef atomic_traits<T> traits;
public:
    atomic_int() : m_value(0) {}
    atomic_int(T v) : m_value(v) {}
    T swap(T v)             { return traits::swap(&m_value, v); }
    T cas(T v_old, T v_new) { return traits::cas(&m_value, v_old, v_new); }
    T operator+=(T v)       { return traits::add(&m_value, v)+v; }
    T operator-=(T v)       { return traits::add(&m_value,-v)-v; }
    T operator++()          { return traits::add(&m_value, 1)+1; }
    T operator--()          { return traits::add(&m_value,-1)-1; }
    T operator++(int)       { return traits::add(&m_value, 1); }
    T operator--(int)       { return traits::add(&m_value,-1); }
    T operator=(T v)        { swap(v); return v; }
    operator T() const      { return m_value; }

private:
    T m_value;
};

typedef atomic_int<int8>  atomic_int8;
typedef atomic_int<int16> atomic_int16;
typedef atomic_int<int32> atomic_int32;
#if defined(ist_env_x64) || (ist_env_ARM64)
typedef atomic_int<int64> atomic_int64;
typedef atomic_int64 atomic_ptrint;
#else // ist_env_*64
typedef atomic_int32 atomic_ptrint;
#endif // ist_env_*64

} // namespace ist

#endif // ist_Concurrency_Atomic_h
