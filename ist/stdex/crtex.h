#ifndef ist_stdex_crtex_h
#define ist_stdex_crtex_h

#include "ist/Config.h"

#ifdef ist_env_Windows

#define istSNPrintf(buf, count, format, ...)    _snprintf(buf, count, format, __VA_ARGS__)
#define istSNWPrintf(buf, count, format, ...)   _snwprintf(buf, count, format, __VA_ARGS__)
#define istVSNPrintf(buf, count, format, va)    _vsnprintf(buf, count, format, va)
#define istVSNWPrintf(buf, count, format, va)   _vsnwprintf(buf, count, format, va)

#define istMalloc(size)                 malloc(size)
#define istFree(ptr)                    free(ptr)
#define istAlignedMalloc(size, align)   _aligned_malloc(size, align)
#define istAlignedFree(ptr)             _aligned_free(ptr)

#pragma intrinsic(memcpy)
#pragma intrinsic(memset)
#pragma intrinsic(memcmp)
#define istMemcpy(Dst, Src, Size)       memcpy(Dst, Src, Size)
#define istMemset(Dst, Value, Size)     memset(Dst, Value, Size)
#define istMemcmp(Ptr1, Ptr2, Size)     memcmp(Ptr1, Ptr2, Size)


#else // ist_env_Windows

#define istSNPrintf(buf, count, format, ...)    snprintf(buf, count, format, __VA_ARGS__)
#define istSNWPrintf(buf, count, format, ...)   snwprintf(buf, count, format, __VA_ARGS__)
#define istVSNPrintf(buf, count, format, va)    vsnprintf(buf, count, format, va)
#define istVSNWPrintf(buf, count, format, va)   vsnwprintf(buf, count, format, va)

#define istMalloc(size)                 malloc(size)
#define istFree(ptr)                    free(ptr)
#define istAlignedMalloc(size, align)   memalign(align, size)
#define istAlignedFree(ptr)             free(ptr)

#define istMemcpy(Dst, Src, Size)       __builtin_memcpy(Dst, Src, Size)
#define istMemset(Dst, Value, Size)     __builtin_memset(Dst, Value, Size)
#define istMemcmp(Ptr1, Ptr2, Size)     __builtin_memcmp(Ptr1, Ptr2, Size)

#endif // ist_env_Windows

template<size_t N>
inline int istSPrintf(char (&buf)[N], const char *format, ...)
{
    va_list vl;
    va_start(vl, format);
    int r = istVSNPrintf(buf, N, format, vl);
    va_end(vl);
    return r;
}

template<size_t N>
inline int istSPrintf(wchar_t (&buf)[N], const wchar_t *format, ...)
{
    va_list vl;
    va_start(vl, format);
    int r = istVSNWPrintf(buf, N, format, vl);
    va_end(vl);
    return r;
}

template<size_t N>
inline int istVSprintf(char (&buf)[N], const char *format, va_list vl)
{
    return istVSNPrintf(buf, N, format, vl);
}

template<size_t N>
inline int istVSprintf(wchar_t (&buf)[N], const wchar_t *format, va_list vl)
{
    return istVSNWPrintf(buf, N, format, vl);
}

#endif // ist_stdex_crtex_h
