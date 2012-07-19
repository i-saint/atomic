#ifndef __ist_Base_TashScheduler_h__
#define __ist_Base_TashScheduler_h__

#include <vector>
#include <deque>
#include <boost/thread.hpp>
#include "Types.h"
#include "ThreadUtil.h"
#include "SharedObject.h"


namespace ist {

class Task
{
friend class TaskScheduler;
public:
    enum State {
        State_Initialized,
        State_Ready,
        State_Running,
        State_Completed,
    };

public:
    Task();
    virtual ~Task();

    Task*   getParent() const               { return m_parent; }
    int     getPriority() const             { return m_priority; }
    State   getState() const                { return m_state; }
    int32   getActiveChildrenCount() const  { return m_num_active_children; }
    void    setParent(Task *v)              { m_parent=v; }
    void    setPriority(int32 v)            { m_priority=v; }

    void waitForComplete();
    void waitForChildren();

    virtual void exec()=0;

private:
    int32 incrementActiveChildren() { ++m_num_active_children; }
    int32 decrementActiveChildren() { --m_num_active_children; }

private:
    Task *m_parent;
    int32 m_priority;
    State m_state;
    atomic_int32 m_num_active_children;
};


class TaskWorker;
class TaskStream;

class TaskScheduler
{
friend class TaskWorker;
template<class T> friend T* ::call_destructor(T *v);
public:
    enum {
        Priority_Max = 2,
        Priority_Default = 1,
    };

    static bool initializeInstance(int32 numThreads=-1); // -1: hardware_concurrency-1
    static bool finalizeInstance();
    static TaskScheduler* getInstance();

    void enqueue(Task *task);
    bool processOneTask();

private:
    TaskScheduler( int32 numThreads );
    ~TaskScheduler();
    Task* dequeue();
    void waitForNewTask();
    void advertiseNewTask();
    int32 incrementHungryWorker() { ++m_num_hungry_worker; }
    int32 decrementHungryWorker() { --m_num_hungry_worker; }

private:
    std::vector< boost::intrusive_ptr<TaskStream> > m_taskstream;
    std::vector< boost::intrusive_ptr<TaskWorker> > m_worker;
    boost::mutex m_mutex_new_task;
    boost::condition_variable m_cond_new_task;
    atomic_int32 m_num_hungry_worker;
};

} // namespace ist

#endif // __ist_Base_TashScheduler_h__
