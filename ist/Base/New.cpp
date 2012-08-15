#include "istPCH.h"
#include "New.h"

#ifdef istWindows

void* istRawMalloc(size_t size, size_t align)
{
    void *p = ::_aligned_malloc(size, align);
    return p;
}

void istRawFree(void* p)
{
    ::_aligned_free(p);
}

void* istRawAlloca( size_t size )
{
    return _alloca(size);
}

#elif // istWindows

void* istRawMalloc(size_t size, size_t align)
{
    void *p = memalign(align, size);
    return p;
}

void istRawFree(void* p)
{
    free(p);
}

void* istRawAlloca( size_t size )
{
    return alloca(size);
}

#endif // istWindows
