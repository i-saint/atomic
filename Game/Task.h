#ifndef __atomic_Game_Task_h__
#define __atomic_Game_Task_h__
namespace atomic {

template<class T>
class Task_UpdateAsync : public Task
{
private:
    T *m_target;

public:
    Task_UpdateAsync(T *v) : m_target(v) {}
    ~Task_UpdateAsync() {}

    void join() { TaskScheduler::waitFor(this); }
    void kick() { TaskScheduler::push(this); }
    void exec() { m_target->updateAsync(); }
    T* getTarget() { return m_target; }
};

} // namespace atomic
#endif //__atomic_Game_Task_h__
