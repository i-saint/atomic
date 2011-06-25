#ifndef __atomic_FractionTask__
#define __atomic_FractionTask__


namespace atomic
{

template<class BaseObjectType>
class PooledFactoryBase
{
protected:
    typedef PooledFactoryBase ThisType;
    eastl::vector<BaseObjectType*> m_unused;

    static eastl::set<ThisType*>& getFactories()
    {
        static eastl::set<ThisType*> s_factories;
        return s_factories;
    }

    template<class ConcreteType>
    ConcreteType* _create()
    {
        ConcreteType* t = NULL;
        if(!m_unused.empty()) {
            t = static_cast<ConcreteType*>(m_unused.back());
            m_unused.pop_back();
        }
        else {
            t = new ConcreteType();
        }
        return t;
    }

    template<class ConcreteType>
    void _unuse(ConcreteType *p)
    {
        m_unused.push_back(p);
    }

    template<class ConcreteType>
    size_t size()
    {
        return m_unused.push_back(p);
    }

    PooledFactoryBase()
    {
        getFactories().insert(this);
    }

    ~PooledFactoryBase()
    {
        for(size_t i=0; i<m_unused.size(); ++i) {
            delete m_unused[i];
        }
        m_unused.clear();
        getFactories().erase(this);
    }

public:
    void deleteAllFactories()
    {
        eastl::set<ThisType*>& cont = getFactories();
        while(!cont.empty())
        {
            delete *cont.begin();
        }
    }
};

template<class ObjectType, class BaseObjectType>
class Factory : public PooledFactoryBase<BaseObjectType>
{
private:
    static Factory* getInstance()
    {
        static Factory *s_factory;
        if(!s_factory) {
            s_factory = new Factory();
        }
        return s_factory;
    }

public:
    static ObjectType* create()
    {
        return getInstance()->_create<ObjectType>();
    }

    static void unuse(ObjectType *p)
    {
        getInstance()->_unuse<ObjectType>(p);
    }

    static size_t size()
    {
        getInstance()->_size<ObjectType>(p);
    }
};







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

class Task_FractionCollisionTest : public Task
{
    typedef ist::ChainedTask super;
private:
    FractionSet *m_obj;
    size_t m_block;

public:
    void initialize(FractionSet *obj, size_t block) { m_obj=obj; m_block=block; }
    void exec() { m_obj->collisionTest(m_block); }
};

class Task_FractionCollisionProcess : public Task
{
    typedef ist::ChainedTask super;
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
    eastl::vector<Task_FractionMove*> m_move_tasks;
    eastl::vector<Task_FractionCollisionTest*> m_col_test_tasks;
    eastl::vector<Task_FractionCollisionProcess*> m_col_proc_tasks;
    Task_FractionSortX *m_sortx_task;
    Task_FractionSortY *m_sorty_task;
    Task_FractionSortZ *m_sortz_task;
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
