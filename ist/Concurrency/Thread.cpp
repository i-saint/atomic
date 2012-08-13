#include "stdafx.h"
#include "ist/Concurrency/Thread.h"

#if defined(istWindows)
#include <process.h>

const DWORD MS_VC_EXCEPTION=0x406D1388;

#pragma pack(push,8)
typedef struct tagTHREADNAME_INFO
{
    DWORD dwType; // Must be 0x1000.
    LPCSTR szName; // Pointer to name (in user addr space).
    DWORD dwThreadID; // Thread ID (-1=caller thread).
    DWORD dwFlags; // Reserved for future use, must be zero.
} THREADNAME_INFO;
#pragma pack(pop)

#elif defined(istAndroid)

#include <cpu-features.h>

#endif // istWindows


namespace ist {

void Thread::setNameToCurrentThread( const char* name )
{
#ifdef istWindows
    THREADNAME_INFO info;
    info.dwType = 0x1000;
    info.szName = name;
    info.dwThreadID = ::GetThreadId(::GetCurrentThread());
    info.dwFlags = 0;
    __try
    {
        ::RaiseException( MS_VC_EXCEPTION, 0, sizeof(info)/sizeof(ULONG_PTR), (ULONG_PTR*)&info );
    }
    __except(EXCEPTION_EXECUTE_HANDLER)
    {
    }
#endif // istWindows
}

void Thread::setAffinityMaskToCurrentThread( size_t mask )
{
#if defined(istWindows)
    ::SetThreadAffinityMask(::GetCurrentThread(), mask);
#elif defined(istAndroid)
    int err, syscallres;
    syscallres = ::syscall(__NR_sched_setaffinity, ::gettid(), sizeof(mask), &mask);
    if (syscallres)
    {
        err = errno;
        LOGE("Error in the syscall setaffinity: mask=%d=0x%x err=%d=0x%x", mask, mask, err, err);
    }
#else
    cpu_set_t target_mask;
    CPU_ZERO( &target_mask );
    for(int i=0; i<sizeof(mask)*8; ++i) {
        if(mask & (1<<i) != 0) {
            CPU_SET( i, &target_mask );
        }
    }
    ::sched_setaffinity( ::gettid(), sizeof(cpu_set_t), &target_mask );
#endif
}

void Thread::setPriorityToCurrentThread( int priority )
{
#if defined(istWindows)
    ::SetThreadPriority(::GetCurrentThread(), priority);
#else
#endif
}

size_t Thread::getLogicalCpuCount()
{
#if defined(istWindows)
    SYSTEM_INFO info={{0}};
    ::GetSystemInfo(&info);
    return info.dwNumberOfProcessors;
#elif defined(istAndroid)
    return ::android_getCpuCount();
#else
    return ::get_nprocs();
#endif
}

Thread::Handle Thread::getCurrentThread()
{
#ifdef istWindows
    return ::GetCurrentThread();
#else // istWindows
    return ::pthread_self();
#endif // istWindows
}


#ifdef istWindows
unsigned int __stdcall _EntryPoint(void *arg)
{
    static_cast<Thread*>(arg)->setParams();
    static_cast<Thread*>(arg)->exec();
    return 0;
}
#else // istWindows
void* _EntryPoint(void *arg)
{
    static_cast<Thread*>(arg)->setParams();
    static_cast<Thread*>(arg)->exec();
    return NULL;
}
#endif // istWindows

Thread::Thread()
    : m_stacksize(0)
    , m_affinity(0)
    , m_priority(0)
{
    sprintf(m_name, "ist::Thread");
}

Thread::~Thread()
{
#ifdef istWindows
#else // istWindows
#endif // istWindows
}

void Thread::run()
{
#ifdef istWindows
    m_handle = (Handle)_beginthreadex(NULL, m_stacksize, _EntryPoint, this, 0, NULL);
#else // istWindows
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setstacksize(&attr, m_stacksize);
    pthread_create(&m_handle, &attr, _EntryPoint, this);
    pthread_attr_destroy(&attr);
#endif // istWindows
}

void Thread::join()
{
#ifdef istWindows
    ::WaitForSingleObject(m_handle, INFINITE);
#else // istWindows
    void *ret = NULL;
    pthread_join(m_handle, &ret);
#endif // istWindows
}

void Thread::setParams()
{
    setNameToCurrentThread(m_name);
    setAffinityMaskToCurrentThread(m_affinity);
    setPriorityToCurrentThread(m_priority);
}

} // namespace ist
