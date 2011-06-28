#include "stdafx.h"
#ifdef WIN32
    #include <windows.h>
#endif
#include <EASTL/algorithm.h>
#include "TaskScheduler.h"


namespace ist
{


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
    typedef stl::list<TaskPtr> task_cont;
    task_cont m_tasks;
    boost::mutex m_suspender;
    boost::condition_variable m_cond;
    size_t m_num_idling_thread;

public:
    TaskQueue();
    bool empty();
    size_t getNumIdlingThread() const;

    void push(TaskPtr t);
    void push(TaskPtr tasks[], size_t num);

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

public:
    TaskThread(int processor);
    ~TaskThread();
    void requestStop();
    boost::thread::id getID() const;
    static bool processTask_Block();
    static bool processTask_NoBlock();
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

void TaskQueue::push(TaskPtr t)
{
    push(&t, 1);
}

void TaskQueue::push(TaskPtr tasks[], size_t num)
{
    if(num==0) { return; }

    // notify() までロック範囲なのは意図的
    boost::lock_guard<boost::mutex> lock(m_suspender);
    for(size_t i=0; i<num; ++i)
    {
        TaskPtr t = tasks[i];
        t->beforeExec();
        m_tasks.push_back(t);
    }

    notify();
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

bool TaskThread::processTask_NoBlock()
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

void TaskScheduler::waitForAll()
{
    // タスクキューが空になるのを待つ
    while(!m_task_queue->empty())
    {
        impl::TaskThread::processTask_NoBlock();
    }
    // 全スレッドがタスクを処理し終えるのを待つ
    while(m_task_queue->getNumIdlingThread() < m_threads.size())
    {
        boost::this_thread::yield();
    }
}

void TaskScheduler::waitFor(TaskPtr task)
{
    if(!task) { return; }
    while(!task->isFinished())
    {
        if(!impl::TaskThread::processTask_NoBlock())
        {
            boost::this_thread::yield();
        }
    }
}

void TaskScheduler::waitFor(TaskPtr tasks[], size_t num)
{
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
        else if(!impl::TaskThread::processTask_NoBlock()) {
            boost::this_thread::yield();
        }
    }
}


void TaskScheduler::schedule(TaskPtr task)
{
    m_task_queue->push(task);
}

void TaskScheduler::schedule(TaskPtr tasks[], size_t num)
{
    m_task_queue->push(tasks, num);
}


size_t TaskScheduler::getThreadCount() const
{
    return m_threads.size();
}

boost::thread::id TaskScheduler::getThreadId(size_t i) const
{
    return m_threads[i]->getID();
}

impl::TaskQueue* TaskScheduler::getTaskQueue()
{
    return m_task_queue.get();
}



} // namespace ist
