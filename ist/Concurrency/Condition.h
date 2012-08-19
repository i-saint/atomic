#ifndef __ist_Concurrency_Condition_h__
#define __ist_Concurrency_Condition_h__

#include "ist/Concurrency/ThreadCommon.h"
#ifndef __ist_env_Windows__
#include "ist/Concurrency/Mutex.h"
#endif // __ist_env_Windows__

namespace ist {

class istInterModule Condition
{
public:
#ifdef __ist_env_Windows__
    typedef HANDLE Handle;
#else
    typedef pthread_cond_t Handle;
#endif // __ist_env_Windows__

    Condition();
    ~Condition();
    void wait();
    /// 誰も待っていない状態で signalOne() した場合、signal 状態が継続します。(Windows の Event 方式)
    /// * 非 Windows の場合、2 つのスレッドに signal が伝わってしまう可能性があります。
    ///   遺憾ですが、1 つのスレッドしか起きないという前提のコードは書かないでください。
    void signalOne();
    /// こちらは誰も待っていなくても signal 状態は継続しません
    void signalAll();

    Handle& getHandle() { return m_lockobj; }

private:
#ifdef __ist_env_Windows__
    Handle m_lockobj;
#else
    Handle m_lockobj;
    Mutex m_mutex;
    atomic_int32 m_signal;
#endif // __ist_env_Windows__
};

} // namespace ist

#endif // __ist_Concurrency_Condition_h__
