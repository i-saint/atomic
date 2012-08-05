#ifndef __ist_Concurrency_TashScheduler_h__
#define __ist_Concurrency_TashScheduler_h__

#include <vector>
#include <deque>
#include <boost/intrusive_ptr.hpp>
#include "ist/Base/Types.h"
#include "ist/Base/SharedObject.h"
#include "ist/Concurrency/Atomic.h"
#include "ist/Concurrency/Mutex.h"
#include "ist/Concurrency/Condition.h"
#include "ist/Concurrency/Thread.h"


namespace ist {

class Task
{
friend class TaskScheduler;
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
    /// �p����ŃI�[�o�[���C�h����ꍇ�ATask::setState() ���Ăԃ^�C�~���O�ɒ��ӁB
    /// setState(State_Completed) �����u�ԕʂ̃X���b�h����j�������\�������邽�߁A
    /// �ŏ��� setState() ����ƈȍ~�̏����ŉ���ς݃������ւ̃A�N�Z�X�G���[����������\��������B
    /// ���������đ����̏ꍇ�͍Ō�ɌĂ΂Ȃ��Ƃ����Ȃ��B
    virtual void setState(State v);

private:
    Priority m_priority;
    State m_state;
};


class TaskWorker;
class TaskStream;

class istInterModule TaskScheduler
{
friend class TaskWorker;
template<class T> friend T* ::call_destructor(T *v);
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
    void waitForNewTask();
    void advertiseNewTask();
    int32 getHungryWorkerCount() { return m_num_hungry_worker; }
    int32 incrementHungryWorker() { return ++m_num_hungry_worker; }
    int32 decrementHungryWorker() { return --m_num_hungry_worker; }

private:
    std::vector< boost::intrusive_ptr<TaskStream> > m_taskstream;
    std::vector< boost::intrusive_ptr<TaskWorker> > m_workers;
    Condition m_cond_new_task;
    atomic_int32 m_num_hungry_worker;
};

} // namespace ist

#endif // __ist_Concurrency_TashScheduler_h__