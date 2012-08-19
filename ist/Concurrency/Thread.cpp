#include "istPCH.h"
#include "ist/Concurrency/Thread.h"

#if defined(__ist_env_Windows__)
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

#elif defined(__ist_env_Android__)

#include <cpu-features.h>

#endif // __ist_env_Windows__


namespace ist {

void Thread::setNameToCurrentThread( const char* name )
{
#ifdef __ist_env_Windows__
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
#endif // __ist_env_Windows__
}

void Thread::setAffinityMaskToCurrentThread( size_t mask )
{
#if defined(__ist_env_Windows__)
    ::SetThreadAffinityMask(::GetCurrentThread(), mask);
#elif defined(__ist_env_Android__)
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
#if defined(__ist_env_Windows__)
    ::SetThreadPriority(::GetCurrentThread(), priority);
#else
#endif
}

size_t Thread::getLogicalCpuCount()
{
#if defined(__ist_env_Windows__)
    SYSTEM_INFO info={{0}};
    ::GetSystemInfo(&info);
    return info.dwNumberOfProcessors;
#elif defined(__ist_env_Android__)
    return ::android_getCpuCount();
#else
    return ::get_nprocs();
#endif
}

Thread::Handle Thread::getCurrentThread()
{
#ifdef __ist_env_Windows__
    return ::GetCurrentThread();
#else // __ist_env_Windows__
    return ::pthread_self();
#endif // __ist_env_Windows__
}


#ifdef __ist_env_Windows__
unsigned int __stdcall _EntryPoint(void *arg)
{
    static_cast<Thread*>(arg)->setParams();
    static_cast<Thread*>(arg)->exec();
    return 0;
}
#else // __ist_env_Windows__
void* _EntryPoint(void *arg)
{
    static_cast<Thread*>(arg)->setParams();
    static_cast<Thread*>(arg)->exec();
    return NULL;
}
#endif // __ist_env_Windows__

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
#ifdef __ist_env_Windows__
#else // __ist_env_Windows__
#endif // __ist_env_Windows__
}

void Thread::run()
{
#ifdef __ist_env_Windows__
    m_handle = (Handle)_beginthreadex(NULL, m_stacksize, _EntryPoint, this, 0, NULL);
#else // __ist_env_Windows__
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setstacksize(&attr, m_stacksize);
    pthread_create(&m_handle, &attr, _EntryPoint, this);
    pthread_attr_destroy(&attr);
#endif // __ist_env_Windows__
}

void Thread::join()
{
#ifdef __ist_env_Windows__
    ::WaitForSingleObject(m_handle, INFINITE);
#else // __ist_env_Windows__
    void *ret = NULL;
    pthread_join(m_handle, &ret);
#endif // __ist_env_Windows__
}

void Thread::setParams()
{
    setNameToCurrentThread(m_name);
    setAffinityMaskToCurrentThread(m_affinity);
    setPriorityToCurrentThread(m_priority);
    ::setlocale(LC_ALL, m_locale);
}

} // namespace ist
