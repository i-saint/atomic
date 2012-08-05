#ifndef __ist_Concurrency_Mutex_h__
#define __ist_Concurrency_Mutex_h__

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
/// 特に Windows の場合 priority boost がかからないためか多くの場合普通の mutex より遅くなります。
/// android で有効に機能することがあるか検証して、だめなら消した方がいいと思われます。
class istInterModule SpinMutex
{
public:
    typedef ScopedLock<SpinMutex> ScopedLock;

    void lock()
    {
        while(m_lockobj.compare_and_swap(1, 0) != 0) {}
    }

    bool tryLock()
    {
        return m_lockobj.compare_and_swap(1, 0) == 0;
    }

    void unlock()
    {
        m_lockobj.swap(0);
    }

private:
    atomic_int32 m_lockobj;
};


/// mutex
/// Windows では CRITICAL_SECTION による実装。
/// recursive 対応
class istInterModule Mutex
{
public:
    typedef ScopedLock<Mutex> ScopedLock;
#ifdef istWindows
    typedef CRITICAL_SECTION Handle;
#else
    typedef pthread_mutex_t Handle;
#endif // istWindows

    Mutex();
    ~Mutex();
    void lock();
    bool tryLock();
    void unlock();

    Handle getHandle() const { return m_lockobj; }

private:
    Handle m_lockobj;
};


} // namespace ist

#endif // __ist_Concurrency_Mutex_h__
