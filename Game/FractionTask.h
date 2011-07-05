#ifndef __atomic_FractionTask__
#define __atomic_FractionTask__


namespace atomic
{




class Task_FractionSortX : public Task
{
private:
    FractionSet *m_obj;
public:
    void initialize(FractionSet *obj) { m_obj=obj; }
    void exec() { m_obj->sortXOrder(); }
};

class Task_FractionSortY : public Task
{
private:
    FractionSet *m_obj;
public:
    void initialize(FractionSet *obj) { m_obj=obj; }
    void exec() { m_obj->sortYOrder(); }
};

class Task_FractionSortZ : public Task
{
private:
    FractionSet *m_obj;
public:
    void initialize(FractionSet *obj) { m_obj=obj; }
    void exec() { m_obj->sortZOrder(); }
};

class Task_FractionGrid : public Task
{
private:
    FractionSet *m_obj;
public:
    void initialize(FractionSet *obj) { m_obj=obj; }
    void exec() { m_obj->updateGrid(); }
};

class Task_FractionCollisionTest : public Task
{
private:
    FractionSet *m_obj;
    size_t m_block;

public:
    void initialize(FractionSet *obj, size_t block) { m_obj=obj; m_block=block; }
    void exec() { m_obj->collisionTest(m_block); }
};

class Task_FractionCollisionProcess : public Task
{
private:
    FractionSet *m_obj;
    size_t m_block;

public:
    void initialize(FractionSet *obj, size_t block) { m_obj=obj; m_block=block; }
    void exec() { m_obj->collisionProcess(m_block); }
};


class Task_FractionMove : public Task
{
typedef ist::ChainedTask super;
private:
    FractionSet *m_obj;
    size_t m_block;

public:
    void initialize(FractionSet *obj, size_t block) { m_obj=obj; m_block=block; }
    void exec() { m_obj->move(m_block); }
};


class Task_FractionUpdate : public Task
{
private:
    FractionSet *m_obj;
    stl::vector<Task_FractionMove*> m_move_tasks;
    stl::vector<Task_FractionCollisionTest*> m_col_test_tasks;
    stl::vector<Task_FractionCollisionProcess*> m_col_proc_tasks;
    Task_FractionSortX *m_sortx_task;
    Task_FractionSortY *m_sorty_task;
    Task_FractionSortZ *m_sortz_task;
    Task_FractionGrid *m_grid_task;
    uint32 m_blocks;

public:
    Task_FractionUpdate();
    ~Task_FractionUpdate();
    void initialize(FractionSet *obj);
    void waitForCompletion();

    void exec();
};


} // namespace atomic
#endif // __atomic_FractionTask__
