#ifndef __ist_Concurrency_Mutex_h__
#define __ist_Concurrency_Mutex_h__

#include "ist/Base/NonCopyable.h"
#include "ist/Concurrency/Atomic.h"
#include "ist/Concurrency/ThreadCommon.h"

namespace ist {


template<class T>
class ScopedLock
{
public:
    ScopedLock(T &m) : m_mutex(m) { m_mutex.lock(); }

    template<class F>
    ScopedLock(T &m, const F &f) : m_mutex(m)
    {
        while(!m_mutex.tryLock()) { f(); }
    }

    ~ScopedLock() { m_mutex.unlock(); }

private:
    T &m_mutex;
};


/// CAS (compare and swap) による spin lock。
/// atomic operation の busy loop で待つため、こいつで長時間待つとシステム全体に深刻なパフォーマンスの悪影響を及ぼします。
/// 使う場合、必ず light_mutex と比較して本当に速くなるか検証が必要です。
template<class AtomicIntT>
class TSpinMutex
{
istNonCopyable(TSpinMutex);
public:
    typedef ScopedLock<TSpinMutex> ScopedLock;

    TSpinMutex() {}

    void lock()
    {
        while(m_lockobj.cas(1, 0)!=0) { NanoSleep(10); }
    }

    template<class F>
    void lock(const F &f)
    {
        while(m_lockobj.cas(1, 0)!=0) { f(); }
    }

    bool tryLock()
    {
        return m_lockobj.cas(1, 0)==0;
    }

    void unlock()
    {
        m_lockobj.swap(0);
    }

private:
    AtomicIntT m_lockobj;
};
typedef TSpinMutex<atomic_int8>  SpinMutex8;
typedef TSpinMutex<atomic_int16> SpinMutex16;
typedef TSpinMutex<atomic_int32> SpinMutex32;
#ifdef ist_env_x64
typedef TSpinMutex<atomic_int64> SpinMutex64;
#endif // ist_env_x64
typedef SpinMutex32 SpinMutex;


/// mutex
/// Windows では CRITICAL_SECTION による実装。
/// recursive 対応
class istInterModule Mutex
{
istNonCopyable(Mutex);
public:
    typedef ScopedLock<Mutex> ScopedLock;
#ifdef ist_env_Windows
    typedef CRITICAL_SECTION Handle;
#else
    typedef pthread_mutex_t Handle;
#endif // ist_env_Windows

    Mutex();
    ~Mutex();
    void lock();
    bool tryLock();
    void unlock();

    Handle getHandle() const { return m_lockobj; }

private:
    Handle m_lockobj;
};


// 何もしないダミー実装
// policy base の実装でマルチスレッド対応/非対応 指定したい場合これが欲しくなる状況がある
class istInterModule DummyMutex
{
istNonCopyable(DummyMutex);
public:
    typedef ScopedLock<DummyMutex> ScopedLock;

    DummyMutex()    {}
    ~DummyMutex()   {}
    void lock()     {}
    bool tryLock()  { return true; }
    void unlock()   {}
};

} // namespace ist

#endif // __ist_Concurrency_Mutex_h__
