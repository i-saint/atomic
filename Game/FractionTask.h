#ifndef __atomic_FractionTask__
#define __atomic_FractionTask__


namespace atomic {



//class Task_FractionCollisionTest : public Task
//{
//private:
//    FractionSet *m_obj;
//    size_t m_block;
//
//public:
//    void initialize(FractionSet *obj, size_t block) { m_obj=obj; m_block=block; }
//    void exec() { m_obj->collisionTest(m_block); }
//};
//
//class Task_FractionCollisionProcess : public Task
//{
//private:
//    FractionSet *m_obj;
//    size_t m_block;
//
//public:
//    void initialize(FractionSet *obj, size_t block) { m_obj=obj; m_block=block; }
//    void exec() { m_obj->collisionProcess(m_block); }
//};
//
//class Task_FractionMove : public Task
//{
//typedef ist::ChainedTask super;
//private:
//    FractionSet *m_obj;
//    size_t m_block;
//
//public:
//    void initialize(FractionSet *obj, size_t block) { m_obj=obj; m_block=block; }
//    void exec() { m_obj->move(m_block); }
//};

class Task_FractionState : public Task
{
    typedef ist::ChainedTask super;
private:
    FractionSet *m_obj;
    size_t m_block;

public:
    Task_FractionState() : m_obj(NULL) {}
    void initialize(FractionSet *obj, size_t block) { m_obj=obj; m_block=block; }
    void exec() { m_obj->updateState(m_block); }
};

class Task_FractionGrid : public Task
{
private:
    FractionSet *m_obj;
public:
    Task_FractionGrid() : m_obj(NULL) {}
    void initialize(FractionSet *obj) { m_obj=obj; }
    void exec() { m_obj->updateGrid(); }
};


class Task_FractionBeforeDraw : public Task
{
private:
    FractionSet *m_obj;
    stl::vector<Task_FractionState*> m_state_tasks;

public:
    Task_FractionBeforeDraw();
    ~Task_FractionBeforeDraw();
    void initialize(FractionSet *obj);
    void waitForComplete();

    void exec();
};

class Task_FractionAfterDraw : public Task
{
private:
    FractionSet *m_obj;
    Task_FractionGrid *m_grid_task;

public:
    Task_FractionAfterDraw();
    ~Task_FractionAfterDraw();
    void initialize(FractionSet *obj);
    void waitForComplete();

    void exec();
};


} // namespace atomic
#endif // __atomic_FractionTask__
