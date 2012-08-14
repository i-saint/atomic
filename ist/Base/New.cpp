#include "istPCH.h"
#include "New.h"


void* istRawMalloc(size_t size, size_t align)
{
    void *p = ::_aligned_malloc(size, align);
    return p;
}

void istRawFree(void* p)
{
    ::_aligned_free(p);
}
