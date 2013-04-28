#ifndef ist_stdex_crtex_h
#define ist_stdex_crtex_h

#include "ist/Config.h"


// printf
#ifdef ist_env_Windows

#define istSNPrintf(buf, count, format, ...)    _snprintf(buf, count, format, __VA_ARGS__)
#define istSNWPrintf(buf, count, format, ...)   _snwprintf(buf, count, format, __VA_ARGS__)
#define istVSNPrintf(buf, count, format, va)    _vsnprintf(buf, count, format, va)
#define istVSNWPrintf(buf, count, format, va)   _vsnwprintf(buf, count, format, va)

#else // ist_env_Windows

#define istSNPrintf(buf, count, format, ...)    snprintf(buf, count, format, __VA_ARGS__)
#define istSNWPrintf(buf, count, format, ...)   snwprintf(buf, count, format, __VA_ARGS__)
#define istVSNPrintf(buf, count, format, va)    vsnprintf(buf, count, format, va)
#define istVSNWPrintf(buf, count, format, va)   vsnwprintf(buf, count, format, va)

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




// memset
#ifdef ist_env_Windows

#pragma intrinsic(memcpy)
#pragma intrinsic(memset)
#pragma intrinsic(memcmp)
#define istMemcpy(Dst, Src, Size)       memcpy(Dst, Src, Size)
#define istMemset(Dst, Value, Size)     memset(Dst, Value, Size)
#define istMemcmp(Ptr1, Ptr2, Size)     memcmp(Ptr1, Ptr2, Size)

#else  // ist_env_Windows

#define istMemcpy(Dst, Src, Size)       __builtin_memcpy(Dst, Src, Size)
#define istMemset(Dst, Value, Size)     __builtin_memset(Dst, Value, Size)
#define istMemcmp(Ptr1, Ptr2, Size)     __builtin_memcmp(Ptr1, Ptr2, Size)

#endif // ist_env_Windows

istForceInline void istMemset128(ist::uint128 *dst, ist::uint128 pattern, size_t size_byte)
{
    // これ書いてる時点では以下のコードは std::fill_n (rep movsd) より速い結果が出ているが、
    // プラットフォームや CPU 次第では std::fill_n の方が速いコードになる可能性アリ。要実測。

    // 可能であれば 128 byte 一気に埋める 
    size_t n = size_byte >> 4;
    size_t blocks = n >> 3;
    for(size_t i=0; i<blocks; ++i) {
        ist::uint128 *d = dst+(i*8);
        d[0] = pattern;
        d[1] = pattern;
        d[2] = pattern;
        d[3] = pattern;
        d[4] = pattern;
        d[5] = pattern;
        d[6] = pattern;
        d[7] = pattern;
    }

    // 余った領域を埋める
    size_t remain = n & 7;
    ist::uint128 *d = dst + (n & ~7);
    for(size_t i=0; i<remain; ++i) {
        d[i] = pattern;
    }
}

istForceInline void istMemset64(ist::uint64 *dst, ist::uint64 pattern, size_t size_byte)
{
    // 可能な部分は memset128 で埋める
    ist::uint128 *dst128 = (ist::uint128*)(((size_t)dst + 0xf) & ~0xf);
    size_t gap = (size_t)dst128-(size_t)dst;
    istAlign(16) ist::uint64 pattern128[2] = {pattern, pattern};
    istMemset128(dst128, (ist::uint128&)pattern128, size_byte-gap);

    // 最初と最後の要素が埋まってない可能性があるので埋める
    dst[0] = pattern;
    dst[(size_byte>>3)-1] = pattern;
}

istForceInline void istMemset32(ist::uint32 *dst, ist::uint32 pattern, size_t size_byte)
{
    // std::fill_n は 32bit 以下の型だと rep stos に化けて、これは SSE レジスタ使う fill より速い
    std::fill_n(dst, size_byte/sizeof(ist::uint32), pattern);
}

istForceInline void istMemset16(ist::uint16 *dst, ist::uint16 pattern, size_t size_byte)
{
    // 同上
    std::fill_n(dst, size_byte/sizeof(ist::uint16), pattern);
}




// malloc
#ifdef ist_env_Windows

#define istAlloca(size)                 alloca(size)
#define istMalloc(size)                 malloc(size)
#define istRealloc(addr, size)          realloc(addr,size)
#define istFree(ptr)                    free(ptr)
#define istAlignedMalloc(size, align)   _aligned_malloc(size, align)
#define istAlignedFree(ptr)             _aligned_free(ptr)

enum istPageFlag {
    istPageNoAccess = PAGE_NOACCESS,
    istPageR        = PAGE_READONLY,
    istPageW        = PAGE_READWRITE,
    istPageRW       = PAGE_READWRITE,
    istPageRX       = PAGE_EXECUTE_READ,
    istPageRWX      = PAGE_EXECUTE_READWRITE,
};

#else // ist_env_Windows

#define istAlloca(size)                 alloca(size)
#define istMalloc(size)                 malloc(size)
#define istRealloc(addr, size)          realloc(addr,size)
#define istFree(ptr)                    free(ptr)
#define istAlignedMalloc(size, align)   memalign(align, size)
#define istAlignedFree(ptr)             free(ptr)

enum istPageFlag {
    istPageNoAccess = PROT_NONE,
    istPageR        = PROT_READ,
    istPageW        = PROT_WRITE,
    istPageRW       = PROT_READ|PROT_WRITE,
    istPageRX       = PROT_READ|PROT_EXEC,
    istPageRWX      = PROT_READ|PROT_WRITE|PROT_EXEC,
};

#endif // ist_env_Windows

inline void* istPageAlloc(void *addr, size_t size, istPageFlag protect)
{
#ifdef ist_env_Windows
    return ::VirtualAlloc(addr, size, MEM_COMMIT|MEM_RESERVE, protect);
#else // ist_env_Windows
    return ::mmap(addr, size, protect, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
#endif // ist_env_Windows
}

inline bool istPageProtect(void *addr, size_t size, istPageFlag protect)
{
#ifdef ist_env_Windows
    DWORD old;
    return ::VirtualProtect(addr, size, protect, &old)==TRUE;
#else // ist_env_Windows
    return ::mpotect(addr, size, protect)==0;
#endif // ist_env_Windows
}

#endif // ist_stdex_crtex_h
