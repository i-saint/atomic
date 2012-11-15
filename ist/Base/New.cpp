#include "istPCH.h"
#include "ist/Base.h"
#include "New.h"
#include "ist/Debug/Callstack.h"
#include "ist/Concurrency/Mutex.h"


#ifdef ist_enable_memory_leak_check
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

#endif // ist_enable_memory_leak_check




void* istRawMalloc(size_t size, size_t align)
{
#ifdef ist_env_Windows
    void *p = ::_aligned_malloc(size, align);
#elif // ist_env_Windows
    void *p = memalign(align, size);
#endif // ist_env_Windows

#ifdef ist_enable_memory_leak_check
    if(g_memory_leak_checker) {
        g_memory_leak_checker->addAllocationInfo(p);
    }
#endif // ist_enable_memory_leak_check

    return p;
}

void istRawFree(void* p)
{
#ifdef ist_enable_memory_leak_check
    if(g_memory_leak_checker) {
        g_memory_leak_checker->eraseAllocationInfo(p);
    }
#endif // ist_enable_memory_leak_check

#ifdef ist_env_Windows
    ::_aligned_free(p);
#elif // ist_env_Windows
    free(p);
#endif // ist_env_Windows
}

void* istRawAlloca( size_t size )
{
#ifdef ist_env_Windows
    return _alloca(size);
#elif // ist_env_Windows
    return alloca(size);
#endif // ist_env_Windows
}

