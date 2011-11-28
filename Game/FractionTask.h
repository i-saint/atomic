#ifndef __atomic_FractionTask_h__
#define __atomic_FractionTask_h__

namespace atomic {

class Task_FractionUpdateAsync : public Task
{
private:
    FractionSet *m_owner;

public:
    Task_FractionUpdateAsync();
    ~Task_FractionUpdateAsync();
    void initialize(FractionSet *obj);
    void waitForComplete();
    void kick() { TaskScheduler::push(this); }
    void exec();

    FractionSet* getOwner() { return m_owner; }
};



} // namespace atomic
#endif // __atomic_FractionTask_h__
