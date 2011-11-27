#include "stdafx.h"
#include "ist/ist.h"
#include "types.h"
#include "AtomicApplication.h"
#include "Game/Message.h"
#include "Game/Fraction.h"
#include "Game/Bullet.h"
#include "Game/World.h"

using namespace ist::graphics;


namespace atomic {



class Task_WorldBeforeDraw : public Task
{
private:
    World *m_obj;

public:
    void initialize(World *obj) { m_obj=obj; }
    void waitForComplete() { TaskScheduler::waitFor(this); }
    void kick() { TaskScheduler::push(this); }
    void exec();
    World* getOwner() { return m_obj; }
};

class Task_WorldAfterDraw : public Task
{
private:
    World *m_obj;

public:
    void initialize(World *obj) { m_obj=obj; }
    void waitForComplete() { TaskScheduler::waitFor(this); }
    void kick() { TaskScheduler::push(this); }
    void exec();
    World* getOwner() { return m_obj; }
};

class Task_WorldDraw : public Task
{
private:
    const World *m_obj;

public:
    void initialize(const World *obj) { m_obj=obj; }
    void waitForComplete() { TaskScheduler::waitFor(this); }
    void kick() { TaskScheduler::push(this); }
    void exec();
    const World* getOwner() { return m_obj; }
};

class Task_WorldCopy : public Task
{
private:
    const World *m_obj;
    World *m_dst;

public:
    void initialize(const World *obj, World *dst) { m_obj=obj; m_dst=dst; }
    void waitForComplete() { TaskScheduler::waitFor(this); }
    void kick() { TaskScheduler::push(this); }
    void exec();
    const World* getOwner() { return m_obj; }
};


void Task_WorldBeforeDraw::exec()
{
    Task_WorldCopy *task_copy = World::getInterframe()->getTask_Copy();
    // ƒRƒs[Š®—¹‘Ò‚¿
    task_copy->waitForComplete();

    m_obj->taskBeforeDraw();

    task_copy->initialize(m_obj, m_obj->getNext());
    task_copy->kick();
}

void Task_WorldAfterDraw::exec()
{
    m_obj->taskAfterDraw();
}

void Task_WorldCopy::exec()
{
    if(m_obj==m_dst) {
        return;
    }

    m_obj->taskCopy(m_dst);
}




World::Interframe::Interframe()
: m_current_world(NULL)
{
    m_task_beforedraw = IST_NEW(Task_WorldBeforeDraw)();
    m_task_afterdraw = IST_NEW(Task_WorldAfterDraw)();
    m_task_copy = IST_NEW(Task_WorldCopy)();
}

World::Interframe::~Interframe()
{
    IST_DELETE(m_task_copy);
    IST_DELETE(m_task_afterdraw);
    IST_DELETE(m_task_beforedraw);
}


World::Interframe *World::s_interframe;

void World::initializeInterframe()
{
    if(!s_interframe) {
        s_interframe = IST_NEW16(Interframe)();
    }
    FractionSet::InitializeInterframe();
}

void World::finalizeInterframe()
{
    FractionSet::FinalizeInterframe();
    IST_DELETE(s_interframe);
}


World::World()
: m_prev(NULL)
, m_next(NULL)
, m_fraction_set(NULL)
, m_bullet_set(NULL)
, m_frame(0)
{
    m_fraction_set = IST_NEW16(FractionSet)();
    m_bullet_set = IST_NEW16(BulletSet)();
}

World::~World()
{
    IST_DELETE(m_fraction_set);
    sync();
}

void World::initialize()
{
    m_rand.initialize(0);
    m_camera.setPosition(XMVectorSet(1.0f, 1.0f, 3.0f, 0.0f));
    m_camera.setZNear(0.01f);
    m_camera.setZFar(10.0f);

    m_fraction_set->initialize();
}

void World::update()
{
    Task_WorldBeforeDraw *task = getInterframe()->getTask_BeforeDraw();
    task->waitForComplete();

    ++m_frame;

    getInterframe()->setCurrentWorld(this);
    task->initialize(this);
    task->kick();

    m_fraction_set->update();
}

void World::draw() const
{
    m_fraction_set->draw();
}

void World::sync() const
{
    m_fraction_set->sync();

    Task_WorldBeforeDraw *task_before = getInterframe()->getTask_BeforeDraw();
    Task_WorldAfterDraw *task_after = getInterframe()->getTask_AfterDraw();
    Task_WorldCopy *task_copy = getInterframe()->getTask_Copy();
    if(task_before->getOwner()==this) { task_before->waitForComplete(); }
    if(task_after->getOwner()==this) { task_after->waitForComplete(); }
    if(task_copy->getOwner()==this) { task_copy->waitForComplete(); }
}


void World::setNext( World *next )
{
    m_next = next;
    if(next) {
        m_next->m_prev = this;
        m_fraction_set->setNext(next->m_fraction_set);
    }
}

void World::taskBeforeDraw()
{
    m_camera.setPosition(XMVector3Transform(m_camera.getPosition(), XMMatrixRotationY(XMConvertToRadians(0.05f))));
    m_camera.setAspect(atomicGetWindowAspectRatio());
}

void World::taskAfterDraw()
{
}

void World::taskCopy(World *dst) const
{
    dst->m_prev = this;
    dst->m_rand = m_rand;
    dst->m_camera = m_camera;
    dst->m_frame = m_frame;
}


} // namespace atomic
