#include "stdafx.h"
#include "ist/Concurrency/Mutex.h"

namespace ist {

#ifdef istWindows

    Mutex::Mutex()          { InitializeCriticalSection(&m_lockobj); }
    Mutex::~Mutex()         { DeleteCriticalSection(&m_lockobj); }
    void Mutex::lock()      { EnterCriticalSection(&m_lockobj); }
    bool Mutex::tryLock()   { return TryEnterCriticalSection(&m_lockobj)==TRUE; }
    void Mutex::unlock()    { LeaveCriticalSection(&m_lockobj); }

#else // istWindows

    Mutex::Mutex()
    {
        pthread_mutexattr_t attr;
        pthread_mutexattr_init(&attr);
        pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE); // recursive 設定
        pthread_mutex_init(&m_lockobj, &attr);
        pthread_mutexattr_destroy(&attr);
    }

    Mutex::~Mutex()         { pthread_mutex_destroy(&m_lockobj); }
    void Mutex::lock()      { pthread_mutex_lock(&m_lockobj); }
    bool Mutex::tryLock()   { return pthread_mutex_trylock.(&m_lockobj); }
    void Mutex::unlock()    { pthread_mutex_unlock(&m_lockobj); }

#endif // istWindows


} // namespace ist
