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

} // namespace atomic
#endif //__atomic_Game_Task__
