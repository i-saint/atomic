#include "istPCH.h"
#include "ist/Base.h"
#include "New.h"
#include "ist/Debug/Callstack.h"
#include "ist/Concurrency/Mutex.h"


#ifdef __ist_enable_memory_leak_check__
namespace ist {


struct AllocInfo
{
    void *stack[ist_leak_check_max_callstack_size];
    int32 depth;
};

class MemoryLeakChecker
{
public:
    MemoryLeakChecker() : m_enabled(true)
    {
    }

    void enableLeakCheck(bool v) { m_enabled=v; }

    void addAllocationInfo(void *p)
    {
        if(!m_enabled) { return; }

        AllocInfo cs;
        cs.depth = GetCallstack(cs.stack, _countof(cs.stack), 3);
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
    typedef stl::map<void*, AllocInfo, stl::less<void*>, STLAllocatorAdapter(DebugAllocator, stl::pair<const void*, AllocInfo>) > DataTable;
    DataTable m_leakinfo;
    Mutex m_mutex;
    bool m_enabled;
};

} // namespace iat

ist::MemoryLeakChecker *g_memory_leak_checker = NULL;

void istMemoryLeakCheckerInitialize()
{
    g_memory_leak_checker = new (malloc(sizeof(ist::MemoryLeakChecker))) ist::MemoryLeakChecker();
}

void istMemoryLeakCheckerFinalize()
{
    g_memory_leak_checker->~MemoryLeakChecker();
    free(g_memory_leak_checker);
    g_memory_leak_checker = NULL;
}

void istMemoryLeakCheckerPrint()
{
    g_memory_leak_checker->printLeakInfo();
}

void istMemoryLeakCheckerEnable( bool v )
{
    g_memory_leak_checker->enableLeakCheck(v);
}

#endif // __ist_enable_memory_leak_check__




void* istRawMalloc(size_t size, size_t align)
{
#ifdef __ist_env_Windows__
    void *p = ::_aligned_malloc(size, align);
#elif // __ist_env_Windows__
    void *p = memalign(align, size);
#endif // __ist_env_Windows__

#ifdef __ist_enable_memory_leak_check__
    if(g_memory_leak_checker) {
        g_memory_leak_checker->addAllocationInfo(p);
    }
#endif // __ist_enable_memory_leak_check__

    return p;
}

void istRawFree(void* p)
{
#ifdef __ist_enable_memory_leak_check__
    if(g_memory_leak_checker) {
        g_memory_leak_checker->eraseAllocationInfo(p);
    }
#endif // __ist_enable_memory_leak_check__

#ifdef __ist_env_Windows__
    ::_aligned_free(p);
#elif // __ist_env_Windows__
    free(p);
#endif // __ist_env_Windows__
}

void* istRawAlloca( size_t size )
{
#ifdef __ist_env_Windows__
    return _alloca(size);
#elif // __ist_env_Windows__
    return alloca(size);
#endif // __ist_env_Windows__
}

