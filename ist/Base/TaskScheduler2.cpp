#include "stdafx.h"
#include "New.h"
#include "Assert.h"
#include "TaskScheduler2.h"

namespace ist {


Task::Task()
    : m_parent(NULL)
    , m_priority(TaskScheduler::Priority_Default)
    , m_state(State_Initialized)
{}

Task::~Task()
{}

void Task::waitForComplete()
{
    while(getState()!=State_Completed) {
        TaskScheduler::getInstance()->processOneTask();
    }
}

void Task::waitForChildren()
{
    while(getActiveChildrenCount()>0) {
        TaskScheduler::getInstance()->processOneTask();
    }
}





class TaskStream : public SharedObject
{
public:
    void enqueue(Task *v);
    Task* dequeue();

private:
    std::deque<Task*> m_tasks;
    spin_mutex m_mutex;
};

class TaskWorker : public SharedObject
{
public:
    TaskWorker(int32 cpuIndex);
    ~TaskWorker();
    void requestExit() { m_flag_exit=true; }

    void operator()();

private:
    boost::thread *m_thread;
    bool m_flag_exit;
};


void TaskStream::enqueue( Task *v )
{
    spin_mutex::scoped_lock lock(m_mutex);
    m_tasks.push_back(v);
}

Task* TaskStream::dequeue()
{
    Task *ret = NULL;
    {
        spin_mutex::scoped_lock lock(m_mutex);
        if(!m_tasks.empty()) {
            ret = m_tasks.front();
            m_tasks.pop_front();
        }
    }
    return ret;
}

TaskWorker::TaskWorker( int32 cpuIndex )
{
    m_thread = new boost::thread(*this);
}

TaskWorker::~TaskWorker()
{
    delete m_thread;
}

void TaskWorker::operator()()
{
    TaskScheduler *scheduler = TaskScheduler::getInstance();
    for(;;) {
        scheduler->processOneTask();
        scheduler->incrementHungryWorker();
        scheduler->waitForNewTask();
        bool flag_exit = m_flag_exit;
        scheduler->decrementHungryWorker();
        if(flag_exit) { break; }
    }
}


TaskScheduler *g_task_scheduler = NULL;

bool TaskScheduler::initializeInstance( int32 numThreads/*=-1*/ )
{
    if(g_task_scheduler!=NULL) { return false; }
    g_task_scheduler = istNew(TaskScheduler)(numThreads);
}

bool TaskScheduler::finalizeInstance()
{
    if(g_task_scheduler==NULL) { return false; }
    istSafeDelete(g_task_scheduler);
}

TaskScheduler* TaskScheduler::getInstance()
{
    return g_task_scheduler;
}


void TaskScheduler::enqueue( Task *task )
{
    assert(task->getPriority()<=Priority_Max);
    m_taskstream[task->getPriority()]->enqueue(task);
}

bool TaskScheduler::processOneTask()
{
    if(Task *task=dequeue()) {
        task->exec();
        return true;
    }
    return false;
}

TaskScheduler::TaskScheduler( int32 numThreads )
{
    // todo
}

TaskScheduler::~TaskScheduler()
{
    // todo
}


Task* TaskScheduler::dequeue()
{
    for(int32 i=Priority_Max; i>=0; --i) {
        if(Task *ret=m_taskstream[i]->dequeue()) {
            return ret;
        }
    }
    return NULL;
}

void TaskScheduler::waitForNewTask()
{
    boost::mutex::scoped_lock lock(m_mutex_new_task);
    m_cond_new_task.wait(lock);
}

void TaskScheduler::advertiseNewTask()
{
    m_cond_new_task.notify_all();
}

} // namespace ist
