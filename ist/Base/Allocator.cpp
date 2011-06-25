#include "stdafx.h"
#include "Allocator.h"

static const size_t minimum_alignment = 16;


//struct allocation_info
//{
//    void* aligned;
//    void* pointer;
//    size_t size;
//    int flag;
//    char* name;
//    int line;
//    char* file;
//};
//
//struct less_pointer
//{
//    bool operator()(const allocation_info& a, const allocation_info& b)
//    {
//        return a.aligned < b.aligned;
//    }
//};
//typedef std::map<allocation_info, less_pointer> allocation_list;
//static allocation_list g_allocations;


//OutputDebugString

void bad_alloc_hander_generic(const void* allocator_ptr)
{
    DebugBreak();
}


void* operator new[](size_t size)
{
    void *p = ::_aligned_malloc(size, minimum_alignment);
    return p;
}

void operator delete[](void* p)
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
