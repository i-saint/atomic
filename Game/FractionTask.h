#ifndef __atomic_FractionTask_h__
#define __atomic_FractionTask_h__

namespace atomic {

class Task_FractionBeforeDraw_Block;
class Task_FractionSPHDensity;


class Task_FractionBeforeDraw : public Task
{
private:
    FractionSet *m_owner;
    stl::vector<Task_FractionBeforeDraw_Block*> m_state_tasks;
    stl::vector<Task_FractionSPHDensity*> m_sph_density_tasks;

public:
    Task_FractionBeforeDraw();
    ~Task_FractionBeforeDraw();
    void initialize(FractionSet *obj);
    void waitForComplete();
    void kick() { TaskScheduler::push(this); }
    void exec();

    FractionSet* getOwner() { return m_owner; }
};

class Task_FractionAfterDraw : public Task
{
private:
    FractionSet *m_owner;

public:
    Task_FractionAfterDraw();
    ~Task_FractionAfterDraw();
    void initialize(FractionSet *obj);
    void waitForComplete();
    void kick() { TaskScheduler::push(this); }
    void exec();

    FractionSet* getOwner() { return m_owner; }
};


class Task_FractionCopy : public Task
{
private:
    const FractionSet *m_owner;
    FractionSet *m_dst;

public:
    Task_FractionCopy() : m_owner(NULL), m_dst(NULL) {}
    void initialize(const FractionSet *obj, FractionSet *dst);
    void waitForComplete();
    void kick() { TaskScheduler::push(this); }
    void exec();

    const FractionSet* getOwner() { return m_owner; }
};


} // namespace atomic
#endif // __atomic_FractionTask_h__
