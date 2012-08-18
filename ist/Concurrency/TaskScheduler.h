#ifndef __ist_Concurrency_TashScheduler_h__
#define __ist_Concurrency_TashScheduler_h__

#include <vector>
#include <deque>
#include <boost/intrusive_ptr.hpp>
#include "ist/Base/Types.h"
#include "ist/Base/SharedObject.h"
#include "ist/Concurrency/Atomic.h"
#include "ist/Concurrency/Mutex.h"
#include "ist/Concurrency/Condition.h"
#include "ist/Concurrency/Thread.h"


namespace ist {

class istInterModule Task
{
friend class TaskScheduler;
public:
    enum Priority {
        Priority_Low    = 0,
        Priority_Normal = 1,
        Priority_High   = 2,

        Priority_Default= 1,
        Priority_Max    = 2,
    };
    enum State {
        State_Completed,
        State_Ready,
        State_Running,
    };

public:
    Task();
    virtual ~Task();

    Priority    getPriority() const             { return m_priority; }
    State       getState() const                { return m_state; }
    void        setPriority(Priority v)         { m_priority=v; }

    void wait();

    virtual void exec()=0;

protected:
    /// 継承先でオーバーライドする場合、Task::setState() を呼ぶタイミングに注意。
    /// setState(State_Completed) した瞬間別のスレッドから破棄される可能性があるため、
    /// 最初に setState() すると以降の処理で解放済みメモリへのアクセスエラーが発生する可能性がある。
    /// したがって多くの場合は最後に呼ばないといけない。
    virtual void setState(State v);

private:
    Priority m_priority;
    State m_state;
};


class TaskWorker;
class TaskStream;

class istInterModule TaskScheduler
{
istMakeDestructable;
friend class TaskWorker;
public:
    static bool initializeInstance(int32 uint32=-1); // -1: hardware_concurrency-1
    static bool finalizeInstance();
    static TaskScheduler* getInstance();

    void enqueue(Task *task);
    bool processOneTask();

    void waitForAll();

private:
    TaskScheduler( uint32 num_threads );
    ~TaskScheduler();
    Task* dequeue();
    void waitForNewTask();
    void advertiseNewTask();
    int32 getHungryWorkerCount() { return m_num_hungry_worker; }
    int32 incrementHungryWorker() { return ++m_num_hungry_worker; }
    int32 decrementHungryWorker() { return --m_num_hungry_worker; }

private:
    stl::vector< TaskStream* > m_taskstream;
    stl::vector< TaskWorker* > m_workers;
    Condition m_cond_new_task;
    atomic_int32 m_num_hungry_worker;
};

} // namespace ist

#endif // __ist_Concurrency_TashScheduler_h__
