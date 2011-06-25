#ifndef __ist_TaskScheduler__
#define __ist_TaskScheduler__

#ifndef _WIN32_WINNT
  #define _WIN32_WINNT 0x0500
  #define WINVER 0x0500
#endif

#include <EASTL/vector.h>
#include <EASTL/list.h>
#include <boost/thread.hpp>
#include <boost/smart_ptr.hpp>

namespace ist
{

class Task;
typedef Task* TaskPtr;


namespace impl
{
    class TaskThread;
    class TaskQueue;
} // namespace impl



// todo: 優先順位
class Task
{
friend class TaskScheduler;
friend class impl::TaskQueue;
friend class impl::TaskThread;
private:
    bool m_finished;

private:
    void beforeExec();
    void afterExec();

public:
    Task();
    virtual ~Task();
    bool isFinished() const;

    virtual void exec()=0;
};




namespace impl
{

    class TaskQueue;
    class TaskThread;


} // namespace impl



class TaskScheduler
{
private:
    typedef boost::shared_ptr<impl::TaskThread> thread_ptr;
    typedef eastl::vector<thread_ptr> thread_cont;
    typedef boost::scoped_ptr<impl::TaskQueue> queue_ptr;

    static TaskScheduler *s_instance;
    queue_ptr m_task_queue;
    thread_cont m_threads;

private:
    TaskScheduler();
    ~TaskScheduler();
    void initialize(size_t num_thread);

public:
    /// num_thread: スレッド数。0 の場合 CPU の数に自動調整。 
    static void initializeSingleton(size_t num_thread=0);
    /// 現在処理中のタスクの完了を待って破棄。(タスクキューが空になるのを待たない) 
    static void finalizeSingleton();
    static TaskScheduler* getInstance();


    /// 全タスクの完了を待つ。タスクキューが空ではない場合、呼び出し元スレッドもタスク処理に加わる。 
    /// タスク内から呼ぶと永久停止するのでやっちゃダメ。 
    void waitForAll();
    /// 指定タスクの完了を待つ。タスクキューが空ではない場合、呼び出し元スレッドもタスク処理に加わる。 
    void waitFor(TaskPtr task);
    /// 範囲指定バージョン 
    void waitFor(TaskPtr tasks[], size_t num);


    /// タスクのスケジューリングを行う。 
    void schedule(TaskPtr task);
    // 範囲指定バージョン 
    void schedule(TaskPtr tasks[], size_t num);


    size_t getThreadCount() const;
    boost::thread::id getThreadId(size_t i) const;


    /// 内部実装用
    impl::TaskQueue* getTaskQueue();
};




// 以下、ユーティリティ系


// 子/チェインを持つタスク
class ChainedTask : public Task
{
protected:
    eastl::vector<TaskPtr> m_children;
    eastl::vector<TaskPtr> m_chain;

public:
    void clear()
    {
        m_children.clear();
        m_chain.clear();
    }

    void appendChild(TaskPtr p) { m_children.push_back(p); }
    void appendChain(TaskPtr p) { m_chain.push_back(p); }

    void exec()
    {
        TaskScheduler& scheduler = *TaskScheduler::getInstance();
        scheduler.schedule(&m_children[0], m_children.size());
        scheduler.waitFor(&m_children[0], m_children.size());
        scheduler.schedule(&m_chain[0], m_chain.size());
    }
};


} // namespace ist
#endif // __ist_TaskScheduler__ 
