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
#include <boost/interprocess/detail/atomic.hpp>


namespace ist {

// スレッドに名前を設定します。デバッガから識別しやすくなります。 
// 生成直後の他スレッドに対しては失敗することがあり、スレッド自身が自分につけるのが望ましいです。
void SetThreadName(uint32_t thread_id, const char *name);
void SetThreadName(const char *name);


class SpinMutex
{
public:
    class Lock
    {
    private:
        SpinMutex &m_mutex;

    public:
        Lock(SpinMutex &m) : m_mutex(m) { m_mutex.lock(); }
        ~Lock() { m_mutex.unlock(); }
    };

private:
    volatile LONG m_lock;

public:
    SpinMutex() : m_lock(0) { }

    void lock()
    {
        while(::InterlockedCompareExchange(&m_lock, 1, 0) != 0) {
            ::Sleep(0);
        }
    }

    void unlock()
    {
        ::InterlockedExchange(&m_lock, 0);
    }
};


class Task;
typedef Task* TaskPtr;


namespace impl
{
    class TaskThread;
    class TaskQueue;
} // namespace impl



class Task
{
friend class TaskScheduler;
friend class impl::TaskQueue;
friend class impl::TaskThread;
private:
    bool m_finished;
    int m_priority;

private:
    void beforeExec();
    void afterExec();

public:
    Task();
    virtual ~Task();    // デストラクタで join() はしないので注意。

    bool isFinished() const { return m_finished; }
    int getPriority() const { return m_priority; }
    void setPriority(int v) { m_priority=v; }

    void kick();
    void join();

    virtual void exec()=0;
};



class TaskScheduler
{
private:
    typedef boost::shared_ptr<impl::TaskThread> thread_ptr;
    typedef stl::vector<thread_ptr> thread_cont;
    typedef boost::scoped_ptr<impl::TaskQueue> queue_ptr;

    static TaskScheduler *s_instance;
    queue_ptr m_task_queue;
    thread_cont m_threads;

private:
    TaskScheduler();
    void initialize(size_t num_thread);
    void finalize();

public:
    ~TaskScheduler();

    /// num_thread: スレッド数。0 の場合 CPU の数に自動調整。 
    static void initializeInstance(size_t num_thread=0);
    /// 現在処理中のタスクの完了を待って破棄。(タスクキューが空になるのを待たない) 
    static void finalizeInstance();
    static TaskScheduler* getInstance();

    /// 実行待ちタスクがあればそれを処理する。なければ偽を返す。 
    static bool wait();
    /// 指定タスクの完了を待つ。実行待ちタスクがある場合、呼び出し元スレッドは待ってる間タスク処理に加わる。 
    /// タスク間で相互に待つようなシチュエーションがあると永久停止する可能性があるので注意。 
    static void waitFor(TaskPtr task);
    static void waitFor(TaskPtr tasks[], size_t num);
    /// 指定タスクの完了を待つ。waitFor() と違い、呼び出し元スレッドはタスク処理に加わらない。
    static void waitExclusive(TaskPtr task);
    static void waitExclusive(TaskPtr tasks[], size_t num);
    /// 全タスクの完了を待つ。実行待ちタスクがある場合、呼び出し元スレッドは待ってる間タスク処理に加わる。 
    /// タスク内から呼ぶと永久停止するのでやっちゃダメ。 
    static void waitForAll();

    /// タスクのスケジューリングを行う。 
    static void addTask(TaskPtr task);
    static void addTask(TaskPtr tasks[], size_t num);

    static size_t getThreadCount();
    static boost::thread::id getThreadId(size_t i);


    /// 内部実装用
    impl::TaskQueue* getTaskQueue();
};




// 以下、ユーティリティ系


// 子/チェインを持つタスク
class ChainedTask : public Task
{
protected:
    stl::vector<TaskPtr> m_children;
    stl::vector<TaskPtr> m_chain;

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
        scheduler.addTask(&m_children[0], m_children.size());
        scheduler.waitFor(&m_children[0], m_children.size());
        scheduler.addTask(&m_chain[0], m_chain.size());
    }
};

template<class Functor>
class FunctionalTask : public Task
{
private:
    Functor m_func;

public:
    FunctionalTask(const Functor& v) : m_func(v) {}

    void exec()
    {
        m_func();
    }
};


} // namespace ist
#endif // __ist_TaskScheduler__ 
