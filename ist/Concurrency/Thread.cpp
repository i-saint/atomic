#include "istPCH.h"
#include "ist/Concurrency/Thread.h"

#if defined(ist_env_Windows)
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

#elif defined(ist_env_Android)

#include <cpu-features.h>

#endif // ist_env_Windows


namespace ist {

void Thread::setNameToCurrentThread( const char* name )
{
#ifdef ist_env_Windows
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
#endif // ist_env_Windows
}

void Thread::setAffinityMaskToCurrentThread( size_t mask )
{
#if defined(ist_env_Windows)
    ::SetThreadAffinityMask(::GetCurrentThread(), mask);
#elif defined(ist_env_Android)
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
#if defined(ist_env_Windows)
    ::SetThreadPriority(::GetCurrentThread(), priority);
#else
#endif
}

size_t Thread::getLogicalCpuCount()
{
#if defined(ist_env_Windows)
    SYSTEM_INFO info={{0}};
    ::GetSystemInfo(&info);
    return info.dwNumberOfProcessors;
#elif defined(ist_env_Android)
    return ::android_getCpuCount();
#else
    return ::get_nprocs();
#endif
}

Thread::Handle Thread::getCurrentThread()
{
#ifdef ist_env_Windows
    return ::GetCurrentThread();
#else // ist_env_Windows
    return ::pthread_self();
#endif // ist_env_Windows
}


#ifdef ist_env_Windows
unsigned int __stdcall _EntryPoint(void *arg)
{
    static_cast<Thread*>(arg)->setParams();
    static_cast<Thread*>(arg)->exec();
    return 0;
}
#else // ist_env_Windows
void* _EntryPoint(void *arg)
{
    static_cast<Thread*>(arg)->setParams();
    static_cast<Thread*>(arg)->exec();
    return NULL;
}
#endif // ist_env_Windows

Thread::Thread()
    : m_stacksize(0)
    , m_affinity(0)
    , m_priority(0)
{
    sprintf(m_name, "ist::Thread");
    m_locale[0] = '\0';
}

Thread::~Thread()
{
#ifdef ist_env_Windows
#else // ist_env_Windows
#endif // ist_env_Windows
}

void Thread::run()
{
#ifdef ist_env_Windows
    m_handle = (Handle)_beginthreadex(NULL, m_stacksize, _EntryPoint, this, 0, NULL);
#else // ist_env_Windows
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setstacksize(&attr, m_stacksize);
    pthread_create(&m_handle, &attr, _EntryPoint, this);
    pthread_attr_destroy(&attr);
#endif // ist_env_Windows
}

void Thread::join()
{
#ifdef ist_env_Windows
    ::WaitForSingleObject(m_handle, INFINITE);
#else // ist_env_Windows
    void *ret = NULL;
    pthread_join(m_handle, &ret);
#endif // ist_env_Windows
}

void Thread::setParams()
{
    setNameToCurrentThread(m_name);
    setAffinityMaskToCurrentThread(m_affinity);
    setPriorityToCurrentThread(m_priority);
    ::setlocale(LC_ALL, m_locale);
}

} // namespace ist
