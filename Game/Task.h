#ifndef atomic_Game_Task_h
#define atomic_Game_Task_h
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
#endif //atomic_Game_Task_h
