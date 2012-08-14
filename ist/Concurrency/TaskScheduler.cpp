#include "istPCH.h"
#include "ist/Base/New.h"
#include "ist/Base/Assert.h"
#include "ist/Concurrency/TaskScheduler.h"

namespace ist {


Task::Task()
    : m_priority(Priority_Default)
    , m_state(State_Completed)
{}

Task::~Task()
{}

void Task::setState(State v)
{
    m_state = v;
}

void Task::wait()
{
    while(getState()!=State_Completed) {
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
    Mutex m_mutex;
};

class TaskWorker : public Thread
{
public:
    TaskWorker(int32 cpu_index);
    ~TaskWorker();
    void requestExit() { m_flag_exit.swap(1); }
    bool getExitFlag() { return m_flag_exit.compare_and_swap(m_flag_exit, m_flag_exit)!=0; }

    void exec();

private:
    atomic_int32 m_flag_exit;
};


void TaskStream::enqueue( Task *v )
{
    Mutex::ScopedLock lock(m_mutex);
    m_tasks.push_back(v);
}

Task* TaskStream::dequeue()
{
    Task *ret = NULL;
    {
        Mutex::ScopedLock lock(m_mutex);
        if(!m_tasks.empty()) {
            ret = m_tasks.front();
            m_tasks.pop_front();
        }
    }
    return ret;
}


TaskWorker::TaskWorker( int32 cpu_index )
{
    setName("ist::TaskWorker");
    setAffinityMask(1<<cpu_index);
    setPriority(Thread::Priority_High);
    //setPriority();
    run();
}

TaskWorker::~TaskWorker()
{
    join();
}

void TaskWorker::exec()
{
    TaskScheduler *scheduler = TaskScheduler::getInstance();
    for(;;) {
        while(scheduler->processOneTask()) {}

        scheduler->incrementHungryWorker();
        scheduler->waitForNewTask();
        bool flag_exit = getExitFlag();
        scheduler->decrementHungryWorker();
        if(flag_exit) { break; }
    }
}


TaskScheduler *g_task_scheduler = NULL;

bool TaskScheduler::initializeInstance( int32 numThreads/*=-1*/ )
{
    if(g_task_scheduler!=NULL) { return false; }
    istNew(TaskScheduler)(numThreads);
    return true;
}

bool TaskScheduler::finalizeInstance()
{
    if(g_task_scheduler==NULL) { return false; }
    istDelete(g_task_scheduler);
    return true;
}

TaskScheduler* TaskScheduler::getInstance()
{
    return g_task_scheduler;
}


void TaskScheduler::enqueue( Task *task )
{
    if(task==NULL) { return; }
    assert( task->getPriority()<=Task::Priority_Max );
    assert( task->getState()!=Task::State_Ready && task->getState()!=Task::State_Running );

    task->setState(Task::State_Ready);
    m_taskstream[task->getPriority()]->enqueue(task);
    advertiseNewTask();
}

bool TaskScheduler::processOneTask()
{
    if(Task *task=dequeue()) {
        task->setState(Task::State_Running);
        task->exec();
        task->setState(Task::State_Completed);
        return true;
    }
    return false;
}

void TaskScheduler::waitForAll()
{
    while(processOneTask()) {}
    while(getHungryWorkerCount()!=m_workers.size()) {}
}

TaskScheduler::TaskScheduler( uint32 num_threads )
{
    g_task_scheduler = this;

    Thread::setPriorityToCurrentThread(Thread::Priority_High);


    // task stream 作成
    for(int32 i=0; i<Task::Priority_Max+1; ++i) {
        m_taskstream.push_back( istNew(TaskStream)() );
    }

    // task worker 作成
    int processors = Thread::getLogicalCpuCount();
#ifdef istWindows
    SYSTEM_INFO info;
    GetSystemInfo(&info);
    processors = info.dwNumberOfProcessors;
    SetThreadAffinityMask(GetCurrentThread(), 1);
#endif // istWindows
    if(num_threads == -1) { num_threads = processors; }

    for(size_t i=1; i<num_threads; ++i)
    {
        TaskWorker *worker = istNew(TaskWorker)(i%processors);
        m_workers.push_back(worker);
    }
}

TaskScheduler::~TaskScheduler()
{
    // 全タスクの処理完了を待つ
    waitForAll();
    // worker に終了要求を出してから wait 待ちを解除
    for(size_t i=0; i<m_workers.size(); ++i) {
        m_workers[i]->requestExit();
    }
    while(getHungryWorkerCount()>0) {
        advertiseNewTask();
    }
    for(size_t i=0; i<m_workers.size(); ++i) {
        m_workers[i]->join();
    }
    // ここまできたら worker を delete しても大丈夫なはず
    m_workers.clear();

    m_taskstream.clear();

    g_task_scheduler = NULL;
}


Task* TaskScheduler::dequeue()
{
    for(int32 i=Task::Priority_Max; i>=0; --i) {
        if(Task *ret=m_taskstream[i]->dequeue()) {
            return ret;
        }
    }
    return NULL;
}

void TaskScheduler::waitForNewTask()
{
    m_cond_new_task.wait();
}

void TaskScheduler::advertiseNewTask()
{
    int32 n = getHungryWorkerCount();
    for(int32 i=0; i<n; ++i) {
        m_cond_new_task.signalOne();
    }
}


} // namespace ist
