#include "istPCH.h"
#include "ist/Base.h"
#include "New.h"
#include "ist/Debug/Callstack.h"
#include "ist/Concurrency/Mutex.h"


#ifdef __ist_enable_memory_leak_check__
namespace ist {


struct AllocInfo
{
    void *stack[16];
    int32 depth;
};

class MemoryLeakChecker
{
public:
    void addAllocationInfo(void *p)
    {
        AllocInfo cs;
        cs.depth = GetCallstack(cs.stack, _countof(cs.stack));
        {
            Mutex::ScopedLock l(m_mutex);
            m_leakinfo[p] = cs;
        }
    }

    void eraseAllocationInfo(void *p)
    {
        Mutex::ScopedLock l(m_mutex);
        m_leakinfo.erase(p);
    }

    void printLeakInfo()
    {
        Mutex::ScopedLock l(m_mutex);
        for(DataTable::iterator i=m_leakinfo.begin(); i!=m_leakinfo.end(); ++i) {
            stl::string text = CallstackToSymbolNames(i->second.stack, i->second.depth);
            istPrint("memory leak: %p\n", i->first);
            istPrint(text.c_str());
            istPrint("\n");
        }
    }

private:
#ifdef __ist_with_EASTL__
    typedef stl::map<void*, AllocInfo, stl::less<void*>, DebugAllocator > DataTable;
#else // __ist_with_EASTL__
    typedef stl::map<void*, AllocInfo, stl::less<void*>, DebugAllocator< stl::pair<const void*, AllocInfo> > > DataTable;
#endif // __ist_with_EASTL__
    DataTable m_leakinfo;
    Mutex m_mutex;
};

MemoryLeakChecker *g_memory_leak_checker = NULL;

void InitializeMemoryLeakChecker()
{
    g_memory_leak_checker = new MemoryLeakChecker();
}

void FinalizeMemoryLeakChecker()
{
    delete g_memory_leak_checker;
    g_memory_leak_checker = NULL;
}

void PrintMemoryLeakInfo()
{
    g_memory_leak_checker->printLeakInfo();
}

} // namespace ist

#endif // __ist_enable_memory_leak_check__




void* istRawMalloc(size_t size, size_t align)
{
#ifdef istWindows
    void *p = ::_aligned_malloc(size, align);
#elif // istWindows
    void *p = memalign(align, size);
#endif // istWindows

#ifdef __ist_enable_memory_leak_check__
    if(ist::g_memory_leak_checker) {
        ist::g_memory_leak_checker->addAllocationInfo(p);
    }
#endif // __ist_enable_memory_leak_check__

    return p;
}

void istRawFree(void* p)
{
#ifdef __ist_enable_memory_leak_check__
    if(ist::g_memory_leak_checker) {
        ist::g_memory_leak_checker->eraseAllocationInfo(p);
    }
#endif // __ist_enable_memory_leak_check__

#ifdef istWindows
    ::_aligned_free(p);
#elif // istWindows
    free(p);
#endif // istWindows
}

void* istRawAlloca( size_t size )
{
#ifdef istWindows
    return _alloca(size);
#elif // istWindows
    return alloca(size);
#endif // istWindows
}

