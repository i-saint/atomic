#include "stdafx.h"
#include "New.h"

static const size_t minimum_alignment = 16;

void* istnew(size_t size)
{
    void *p = ::_aligned_malloc(size, minimum_alignment);
    return p;
}

void istdelete(void* p)
{
    ::_aligned_free(p);
}


void* operator new[](size_t size, const char* pName, int flags, unsigned debugFlags, const char* file, int line)
{
    void* p = ::_aligned_malloc(size, minimum_alignment);
    return p;
}

void* operator new[](size_t size, size_t alignment, size_t alignmentOffset, const char* pName, int flags, unsigned debugFlags, const char* file, int line)
{
    void* p = ::_aligned_offset_malloc(size, alignment, alignmentOffset);
    return p;
}
