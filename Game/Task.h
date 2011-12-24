#ifndef __atomic_Game_Task_h__
#define __atomic_Game_Task_h__
namespace atomic {

class AtomicTask : public Task
{
public:
    ~AtomicTask() { join(); }
    void join() { TaskScheduler::waitFor(this); }
    void kick() { TaskScheduler::push(this); }
};

template<class T>
class Task_UpdateAsync : public AtomicTask
{
private:
    T *m_target;
    float32 m_dt;

public:
    Task_UpdateAsync(T *v) : m_target(v), m_dt(0.0f) {}
    void setArg(float32 v) { m_dt=v; }
    void exec() { m_target->updateAsync(m_dt); }
};

} // namespace atomic
#endif //__atomic_Game_Task_h__
