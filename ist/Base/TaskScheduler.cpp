#include "stdafx.h"
#ifdef WIN32
    #include <windows.h>
#endif
#include <EASTL/algorithm.h>
#include "TaskScheduler.h"


namespace ist {


#ifdef WIN32
const DWORD MS_VC_EXCEPTION=0x406D1388;

#pragma pack(push,8)
typedef struct tagTHREADNAME_INFO
{
    DWORD dwType; // Must be 0x1000.
    LPCSTR szName; // Pointer to name (in user addr space).
    DWORD dwThreadID; // Thread ID (-1=caller thread).
    DWORD dwFlags; // Reserved for future use, must be zero.
} THREADNAME_INFO;
#pragma pack(pop)

void SetThreadName( uint32_t dwThreadID, const char* threadName)
{
    THREADNAME_INFO info;
    info.dwType = 0x1000;
    info.szName = threadName;
    info.dwThreadID = dwThreadID;
    info.dwFlags = 0;

    __try
    {
        RaiseException( MS_VC_EXCEPTION, 0, sizeof(info)/sizeof(ULONG_PTR), (ULONG_PTR*)&info );
    }
    __except(EXCEPTION_EXECUTE_HANDLER)
    {
    }
}

void SetThreadName(const char *name)
{
    SetThreadName((uint32_t)::GetCurrentThreadId(), name);
}

#else
void SetThreadName( uint32_t dwThreadID, char* threadName)
{
}

void SetThreadName(const char *name)
{
}
#endif



Task::Task()
: m_finished(true)
{}

Task::~Task()
{}

void Task::beforeExec()
{
    m_finished = false;
}

void Task::afterExec()
{
    m_finished = true;
}

bool Task::isFinished() const
{
    return m_finished;
}


namespace impl
{


class TaskQueue
{
private:
    typedef eastl::list<TaskPtr> task_cont;
    task_cont m_tasks;
    boost::mutex m_suspender;
    boost::condition_variable m_cond;
    size_t m_num_idling_thread;

public:
    TaskQueue();
    bool empty();
    size_t getNumIdlingThread() const;

    void push_back(TaskPtr t);
    void push_back(TaskPtr tasks[], size_t num);
    void push_front(TaskPtr t);
    void push_front(TaskPtr tasks[], size_t num);

    TaskPtr pop();
    TaskPtr waitForQueuingTask();
    void notify();
};

class TaskThread
{
private:
    bool m_stop_flag;
    boost::scoped_ptr<boost::thread> m_thread;
    boost::condition_variable m_cond;
    int m_processor;

public:
    TaskThread(int processor);
    ~TaskThread();
    void requestStop();
    boost::thread::id getID() const;
    static bool processTask_Block();
    static bool processTask_NonBlock();
    void operator()();
};


TaskQueue::TaskQueue()
: m_num_idling_thread(0)
{}

bool TaskQueue::empty()
{
    boost::unique_lock<boost::mutex> lock(m_suspender);
    return m_tasks.empty();
}

size_t TaskQueue::getNumIdlingThread() const
{
    return m_num_idling_thread;
}

void TaskQueue::push_back(TaskPtr t)
{
    push_back(&t, 1);
}

void TaskQueue::push_back(TaskPtr tasks[], size_t num)
{
    if(num==0) { return; }
    {
        boost::lock_guard<boost::mutex> lock(m_suspender);
        for(size_t i=0; i<num; ++i)
        {
            TaskPtr t = tasks[i];
            t->beforeExec();
            m_tasks.push_back(t);
        }
        notify();
    }
}

void TaskQueue::push_front(TaskPtr t)
{
    push_front(&t, 1);
}

void TaskQueue::push_front(TaskPtr tasks[], size_t num)
{
    if(num==0) { return; }
    {
        boost::lock_guard<boost::mutex> lock(m_suspender);
        for(size_t i=0; i<num; ++i)
        {
            TaskPtr t = tasks[i];
            t->beforeExec();
            m_tasks.push_front(t);
        }
        notify();
    }
}


TaskPtr TaskQueue::pop()
{
    boost::unique_lock<boost::mutex> lock(m_suspender);
    TaskPtr t = NULL;
    if(!m_tasks.empty())
    {
        t = m_tasks.front();
        m_tasks.pop_front();
    }
    return t;
}

TaskPtr TaskQueue::waitForQueuingTask()
{
    boost::unique_lock<boost::mutex> lock(m_suspender);
    if(!m_tasks.empty())
    {
        TaskPtr t = m_tasks.front();
        m_tasks.pop_front();
        return t;
    }
    ++m_num_idling_thread;
    m_cond.wait(lock);
    return NULL;
}

void TaskQueue::notify()
{
    m_num_idling_thread = 0;
    m_cond.notify_all();
}



TaskThread::TaskThread(int processor)
: m_stop_flag(false)
, m_processor(processor)
{
    m_thread.reset(new boost::thread(boost::ref(*this)));
#ifdef WIN32
    ::SetThreadAffinityMask(m_thread->native_handle(), 1<<processor);
#endif
}

TaskThread::~TaskThread()
{
    m_thread->join();
}

void TaskThread::requestStop()
{
    m_stop_flag = true;
}

boost::thread::id TaskThread::getID() const
{
    return m_thread->get_id();
}

void TaskThread::operator()()
{
    {
        char name[32];
        sprintf(name, "TaskProcessor %d", m_processor);
        SetThreadName(name);
    }
    while(!m_stop_flag)
    {
        processTask_Block();
    }
}

bool TaskThread::processTask_Block()
{
    TaskQueue* task_queue = TaskScheduler::getInstance()->getTaskQueue();
    if(TaskPtr t = task_queue->waitForQueuingTask())
    {
        t->exec();
        t->afterExec();
        return true;
    }
    return false;
}

bool TaskThread::processTask_NonBlock()
{
    TaskQueue* task_queue = TaskScheduler::getInstance()->getTaskQueue();
    if(TaskPtr t = task_queue->pop())
    {
        t->exec();
        t->afterExec();
        return true;
    }
    return false;
}

} // namespace impl 


TaskScheduler* TaskScheduler::s_instance = NULL;

void TaskScheduler::initializeSingleton(size_t num_thread)
{
    if(!s_instance) {
        s_instance = new TaskScheduler();
        s_instance->initialize(num_thread);
    }
}

void TaskScheduler:: finalizeSingleton()
{
    if(s_instance) {
        delete s_instance;
        s_instance = NULL;
    }
}

TaskScheduler* TaskScheduler::getInstance()
{
    return s_instance;
}


TaskScheduler::TaskScheduler()
{
}

void TaskScheduler::initialize(size_t num_thread)
{
    m_task_queue.reset(new impl::TaskQueue());

    int processors = boost::thread::hardware_concurrency();
#ifdef WIN32
    SYSTEM_INFO info;
    GetSystemInfo(&info);
    processors = info.dwNumberOfProcessors;
    ::SetThreadAffinityMask(::GetCurrentThread(), 1);
#endif

    if(num_thread == 0)
    {
        num_thread = processors;
    }

    for(size_t i=1; i<num_thread; ++i)
    {
        thread_ptr t(new impl::TaskThread(i%processors));
        m_threads.push_back(t);
    }
}

TaskScheduler::~TaskScheduler()
{
    for(size_t i=0; i<m_threads.size(); ++i)
    {
        m_threads[i]->requestStop();
    }
    m_task_queue->notify();
    m_threads.clear();
}

bool TaskScheduler::wait()
{
    if(impl::TaskThread::processTask_NonBlock())
    {
        return true;
    }
    boost::this_thread::yield();
    return false;
}

void TaskScheduler::waitFor(TaskPtr task)
{
    if(!task) { return; }
    while(!task->isFinished())
    {
        if(!impl::TaskThread::processTask_NonBlock())
        {
            boost::this_thread::yield();
        }
    }
}

void TaskScheduler::waitFor(TaskPtr tasks[], size_t num)
{
    if(!tasks || num==0) { return; }
    for(;;) {
        bool finished = true;
        for(size_t i=0; i<num; ++i) {
            if(tasks[i] && !tasks[i]->isFinished()) {
                finished = false;
                break;
            }
        }

        if(finished) {
            break;
        }
        else if(!impl::TaskThread::processTask_NonBlock()) {
            boost::this_thread::yield();
        }
    }
}

void TaskScheduler::waitExclusive( TaskPtr task )
{
    if(!task) { return; }
    if(s_instance->m_threads.empty()) {
        waitFor(task);
    }
    else {
        while(!task->isFinished())
        {
            boost::this_thread::yield();
        }
    }
}

void TaskScheduler::waitExclusive( TaskPtr tasks[], size_t num )
{
    if(!tasks || num==0) { return; }
    if(s_instance->m_threads.empty()) {
        waitFor(tasks, num);
    }
    else {
        for(size_t i=0; i<num; ++i) {
            if(!tasks[i]) { continue; }
            while(!tasks[i]->isFinished()) {
                boost::this_thread::yield();
            }
        }
    }
}

void TaskScheduler::waitForAll()
{
    // タスクキューが空になるのを待つ
    while(!s_instance->m_task_queue->empty())
    {
        impl::TaskThread::processTask_NonBlock();
    }
    // 全スレッドがタスクを処理し終えるのを待つ
    while(s_instance->m_task_queue->getNumIdlingThread() < s_instance->m_threads.size())
    {
        boost::this_thread::yield();
    }
}


void TaskScheduler::push(TaskPtr task)
{
    s_instance->m_task_queue->push_back(task);
}

void TaskScheduler::push(TaskPtr tasks[], size_t num)
{
    s_instance->m_task_queue->push_back(tasks, num);
}

void TaskScheduler::push_front(TaskPtr task)
{
    s_instance->m_task_queue->push_front(task);
}

void TaskScheduler::push_front(TaskPtr tasks[], size_t num)
{
    s_instance->m_task_queue->push_front(tasks, num);
}


size_t TaskScheduler::getThreadCount()
{
    return s_instance->m_threads.size();
}

boost::thread::id TaskScheduler::getThreadId(size_t i)
{
    return s_instance->m_threads[i]->getID();
}

impl::TaskQueue* TaskScheduler::getTaskQueue()
{
    return s_instance->m_task_queue.get();
}



} // namespace ist
