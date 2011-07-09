#include "stdafx.h"
#include "../ist/ist.h"
#include "../types.h"
#include "AtomicApplication.h"
#include "Message.h"
#include "Fraction.h"
#include "World.h"

using namespace ist::graphics;


namespace atomic {



class Task_WorldBeforeDraw : public Task
{
private:
    World *m_obj;

public:
    void initialize(World *obj) { m_obj=obj; }
    void waitForComplete() { TaskScheduler::waitFor(this); }
    void exec() { m_obj->update_BeforeDraw(); }
};

class Task_WorldAfterDraw : public Task
{
private:
    World *m_obj;

public:
    void initialize(World *obj) { m_obj=obj; }
    void waitForComplete() { TaskScheduler::waitFor(this); }
    void exec() { m_obj->update_AfterDraw(); }
};



World::Interframe::Interframe()
: m_current_world(NULL)
{
    m_task_beforedraw = AT_NEW(Task_WorldBeforeDraw)();
    m_task_afterdraw = AT_NEW(Task_WorldAfterDraw)();
}

World::Interframe::~Interframe()
{
    AT_DELETE(m_task_afterdraw);
    AT_DELETE(m_task_beforedraw);
}


World::Interframe *World::s_interframe;

void World::initializeInterframe()
{
    if(!s_interframe) {
        s_interframe = AT_ALIGNED_NEW(Interframe, 16)();
    }
    FractionSet::InitializeInterframe();
}

void World::finalizeInterframe()
{
    FractionSet::FinalizeInterframe();
    AT_DELETE(s_interframe);
}


World::World()
: m_prev(NULL)
, m_fraction_set(NULL)
{
    m_fraction_set = AT_ALIGNED_NEW(FractionSet, 16)();
}

World::~World()
{
    AT_DELETE(m_fraction_set);
}

void World::initialize( World* prev )
{
    m_prev = prev;
    if(prev) {
        Task_WorldBeforeDraw *task = getInterframe()->getTask_BeforeDraw();
        task->waitForComplete();

        m_rand = prev->m_rand;
        m_camera = prev->m_camera;
    }
    else {
        m_rand.initialize(0);
        m_camera.setPosition(XMVectorSet(100.0f, 100.0f, 500.0f, 0.0f));
        m_camera.setZNear(1.0f);
        m_camera.setZFar(1000.0f);
    }

    m_fraction_set->initialize(prev ? prev->m_fraction_set : NULL);
}

void World::update()
{
    getInterframe()->setCurrentWorld(this);

    Task_WorldBeforeDraw *task = getInterframe()->getTask_BeforeDraw();
    task->waitForComplete();
    task->initialize(this);
    TaskScheduler::schedule(task);

    m_fraction_set->update();
}

void World::draw()
{
    m_fraction_set->draw();
    TaskScheduler::waitFor(getInterframe()->getTask_BeforeDraw());
}

void World::sync()
{
    m_fraction_set->sync();
    TaskScheduler::waitFor(getInterframe()->getTask_BeforeDraw());
}


void World::update_BeforeDraw()
{
    m_camera.setPosition(XMVector3Transform(m_camera.getPosition(), XMMatrixRotationY(XMConvertToRadians(0.1f))));
    m_camera.setAspect(atomicGetWindowAspectRatio());
}

void World::update_AfterDraw()
{

}


} // namespace atomic
