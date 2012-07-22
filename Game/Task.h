#ifndef __atomic_Game_Task__
#define __atomic_Game_Task__
namespace atomic {

class AtomicTask : public Task
{
public:
    ~AtomicTask() { wait(); }
};

class AtomicDrawTask : public AtomicTask
{
public:
    AtomicDrawTask() { setPriority(Task::Priority_Low); }
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
    void exec() { m_target->asyncupdate(m_dt); }
};

} // namespace atomic
#endif //__atomic_Game_Task__
