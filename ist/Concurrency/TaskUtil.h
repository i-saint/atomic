#ifndef ist_Concurrency_TaskUtil_h
#define ist_Concurrency_TaskUtil_h

#include "ist/Concurrency/TaskScheduler.h"

#ifndef ist_with_tbb

namespace ist {

template<class TaskType>
void EnqueueTasks(TaskType **tasks, size_t num_tasks)
{
    for(size_t i=0; i<num_tasks; ++i) {
        TaskScheduler::getInstance()->enqueue(tasks[i]);
    }
}

template<class TaskType>
void WaitTasks(TaskType **tasks, size_t num_tasks)
{
    for(size_t i=0; i<num_tasks; ++i) {
        tasks[i]->wait();
    }
}


class TreeTask : public Task
{
    typedef Task super;
public:
    TreeTask();
    TreeTask*   getParent() const               { return m_parent; }
    int32       getActiveChildrenCount() const  { return m_num_active_children; }
    void        setParent(TreeTask *v)          { m_parent=v; }
    void waitChildren();

private:
    virtual void setState(State v);
    int32 incrementActiveChildren() { return ++m_num_active_children; }
    int32 decrementActiveChildren() { return --m_num_active_children; }

private:
    TreeTask *m_parent;
    atomic_int32 m_num_active_children;
};

} // namepspace ist

#endif // ist_with_tbb
#endif // ist_Concurrency_TaskUtil_h
