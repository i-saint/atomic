#ifndef ist_Concurrency_TashScheduler_h
#define ist_Concurrency_TashScheduler_h

#include "ist/Base/Types.h"
#include "ist/Base/NonCopyable.h"
#include "ist/Base/SharedObject.h"
#include "ist/Concurrency/Atomic.h"
#include "ist/Concurrency/Mutex.h"
#include "ist/Concurrency/Condition.h"
#include "ist/Concurrency/Thread.h"

#ifdef ist_with_tbb
#include <tbb/tbb.h>

namespace ist {

typedef tbb::task Task;
typedef tbb::task_group TaskGroup;
typedef tbb::blocked_range<size_t> size_range;
using tbb::parallel_for;

template<class F>
class FunctorTask : public Task
{
public:
    FunctorTask(const F &f) : m_f(f) {}
    Task* execute()
    {
        m_f();
        return NULL;
    }

private:
    F m_f;
};

} // namespace ist

#define istNewRootTask(Type)        new(tbb::task::allocate_root()) Type
#define istEnqueueTask(...)     tbb::task::spawn(__VA_ARGS__)
#define istWaitTasks()          tbb::task::wait_for_all()

#define istTaskSchedulerInitialize(...) 
#define istTaskSchedulerFinalize()      

#else // ist_with_tbb


namespace ist {

class Task;
class TaskWorker;
class TaskStream;
class TaskScheduler;


class istAPI Task
{
friend class TaskScheduler;
friend class TaskWorker;
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



class istAPI TaskScheduler
{
istNonCopyable(TaskScheduler);
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
    void processOneTask(Task *task);
    void waitForNewTask();
    void advertiseNewTask();

private:
    stl::vector< TaskStream* > m_taskstream;
    stl::vector< TaskWorker* > m_workers;
    Condition m_cond_new_task;
};

} // namespace ist

#define istTaskSchedulerInitialize(...) TaskScheduler::initializeInstance(__VA_ARGS__)
#define istTaskSchedulerFinalize()

#endif // ist_with_tbb
#endif // ist_Concurrency_TashScheduler_h
