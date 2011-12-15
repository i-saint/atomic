#include "stdafx.h"
#include "ist/ist.h"
#include "types.h"
#include "AtomicApplication.h"
#include "Game/Message.h"
#include "Game/Fraction.h"
#include "Game/Bullet.h"
#include "Game/World.h"
#include "Graphics/Renderer.h"

using namespace ist::graphics;


namespace atomic {



class Task_WorlUpdateAsync : public Task
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


void Task_WorlUpdateAsync::exec()
{
    m_obj->updateAsync();
}







World::World()
: m_fraction_set(NULL)
, m_bullet_set(NULL)
, m_frame(0)
{
    m_task_updateasync = IST_NEW(UpdateAsyncTask)(this);

    m_fraction_set = IST_NEW(FractionSet)();
    m_bullet_set = IST_NEW(BulletSet)();
}

World::~World()
{
    sync();

    IST_SAFE_DELETE(m_task_updateasync);

    IST_SAFE_DELETE(m_fraction_set);
    IST_SAFE_DELETE(m_bullet_set);
}

void World::initialize()
{
    m_rand.initialize(0);
    m_camera.setPosition(vec4(1.0f, 1.0f, 3.0f, 0.0f));
    m_camera.setZNear(0.01f);
    m_camera.setZFar(10.0f);

    m_fraction_set->initialize();
}

void World::update()
{
    ++m_frame;

    m_task_updateasync->kick();
    m_fraction_set->update();
    // todo: add module
    sync();
}

void World::draw() const
{
    m_fraction_set->draw();

    DirectionalLight dl;
    dl.direction = glm::normalize(vec4(1.0f, -1.0f, -0.5f, 0.0f));
    dl.diffuse_color = vec4(0.3f, 0.3f, 0.3f, 1.0f);
    dl.ambient_color = vec4(0.0f, 0.0f, 0.0f, 0.0f);
    atomicGetDirectionalLights()->pushInstance(dl);
}

void World::sync() const
{
    m_fraction_set->sync();

    m_task_updateasync->join();
}


void World::updateAsync()
{
    //mat4 rot = glm::rotate(mat4(), 0.05f, vec3(0.0f, 1.0f, 0.0f));
    //m_camera.setPosition(rot * m_camera.getPosition());
    m_camera.setAspect(atomicGetWindowAspectRatio());

}

} // namespace atomic
