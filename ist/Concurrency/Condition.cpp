#include "istPCH.h"
#include "ist/Base/Assert.h"
#include "ist/Concurrency/Condition.h"

namespace ist {

#if defined(ist_env_Windows)

    Condition::Condition()
    {
        m_lockobj = ::CreateEventA(NULL, FALSE, FALSE, NULL);
    }

    Condition::~Condition()
    {
        ::CloseHandle(m_lockobj);
    }

    void Condition::wait()
    {
        ::WaitForSingleObject(m_lockobj, INFINITE);
    }

    void Condition::timedWait( uint32 millisec )
    {
        ::WaitForSingleObject(m_lockobj, millisec);
    }

    void Condition::signalOne()
    {
        BOOL r = ::SetEvent(m_lockobj);
        istAssert(r==TRUE);
    }

    void Condition::signalAll()
    {
        // http://msdn.microsoft.com/en-us/library/windows/desktop/ms684914.aspx
        // PulseEvent() は問題があるらしいが、ConditionVariable は vista 以降なので…
        ::PulseEvent(m_lockobj);
    }

#else

    Condition::Condition()
    {
        pthread_cond_init(m_lockobj, NULL);
    }

    Condition::~Condition()
    {
        pthread_cond_destroy(m_lockobj);
    }

    void Condition::wait()
    {
        // Windows の Event の仕組みをエミュレーション。signal 状態なら待たずに return
        if(m_signal.compare_and_swap(0, 1)==0) { return; }

        pthread_cond_wait(&m_lockobj, &m_mutex.getHandle());
        m_signal.swap(0);
    }

    void Condition::timedWait(uint32 millisec)
    {
        // Windows の Event の仕組みをエミュレーション。signal 状態なら待たずに return
        if(m_signal.compare_and_swap(0, 1)==0) { return; }

        timespec ts;
        ts.tv_sec = 0;
        ts.TV_INSERTSTRUCT = millisec * 1000;
        pthread_cond_timedwait(&m_lockobj, &m_mutex.getHandle(), &ts);
        m_signal.swap(0);
    }

    void Condition::signalOne()
    {
        m_signal.swap(1);
        pthread_cond_signal(&m_lockobj);
    }

    void Condition::signalAll()
    {
        pthread_cond_broadcast(&m_lockobj);
    }

#endif

} // namespace ist
