#include "istPCH.h"
#include "ist/Base/New.h"
#include "ist/Base/Assert.h"
#include "ist/Concurrency/TaskScheduler.h"
#include <deque> // deque は EASTL にはないので標準のを

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
        if(!TaskScheduler::getInstance()->processOneTask()) {
            Thread::sleep(1);
        }
    }
}


class TaskStream
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
    void requestExit()          { m_flag_exit = true; }
    bool getExitFlag()          { return m_flag_exit; }
    void waitUntilCompleteTask(){ m_mutex.lock(); m_mutex.unlock(); }
    bool isWorking()            { return m_mutex.tryLock()==false; }
    bool isCompleted()          { return m_flag_complete; }

    void exec();

private:
    volatile bool m_flag_exit;
    volatile bool m_flag_complete;
    Mutex m_mutex;
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
    : m_flag_exit(false)
    , m_flag_complete(false)
{
    setName("ist::TaskWorker");
    setAffinityMask(1<<cpu_index);
    //setPriority(Thread::Priority_High);
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
        {
            ScopedLock<Mutex> l(m_mutex);
            while(scheduler->processOneTask()) {}
        }
        scheduler->waitForNewTask();
        bool flag_exit = getExitFlag();
        if(flag_exit) { break; }
    }
    m_flag_complete = true;
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
    if(m_workers.empty()) {
        task->setState(Task::State_Running);
        task->exec();
        task->setState(Task::State_Completed);
        return;
    }

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
    for(size_t i=0; i<m_workers.size(); ++i) {
        m_workers[i]->waitUntilCompleteTask();
    }
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
#ifdef ist_env_Windows
    SYSTEM_INFO info;
    GetSystemInfo(&info);
    processors = info.dwNumberOfProcessors;
    SetThreadAffinityMask(GetCurrentThread(), 1);
#endif // ist_env_Windows
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
    for(size_t i=0; i<m_workers.size(); ++i) {
        while(!m_workers[i]->isCompleted()) {
            advertiseNewTask();
        }
    }
    for(size_t i=0; i<m_workers.size(); ++i) {
        m_workers[i]->join();
    }
    // ここまできたら worker を delete しても大丈夫なはず
    for(size_t i=0; i<m_workers.size(); ++i) { istDelete(m_workers[i]); }
    m_workers.clear();

    for(size_t i=0; i<m_taskstream.size(); ++i) { istDelete(m_taskstream[i]); }
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
    for(size_t i=0; i<m_workers.size(); ++i) {
        m_cond_new_task.signalOne();
    }
}


} // namespace ist
