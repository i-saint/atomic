#include "stdafx.h"
#include "ist/ist.h"
#include "types.h"
#include "AtomicApplication.h"
#include "AtomicGame.h"
#include "Game/Message.h"
#include "Game/SPHManager.h"
#include "Game/Entity.h"
#include "Game/World.h"
#include "Graphics/Renderer.h"
#include "Character/Enemy.h"

using namespace ist::graphics;


namespace atomic {




World::World()
: m_entity_set(NULL)
, m_sph(NULL)
, m_frame(0)
{
    m_entity_set = IST_NEW(EntitySet)();
    m_sph = IST_NEW(SPHManager)();

    m_task_update_world = IST_NEW(Task_UpdateAsync<World>)(this);
    m_task_update_entity= IST_NEW(Task_UpdateAsync<EntitySet>)(m_entity_set);
    m_task_update_sph   = IST_NEW(Task_UpdateAsync<SPHManager>)(m_sph);
}

World::~World()
{
    IST_SAFE_DELETE(m_task_update_sph);
    IST_SAFE_DELETE(m_task_update_entity);
    IST_SAFE_DELETE(m_task_update_world);

    IST_SAFE_DELETE(m_sph);
    IST_SAFE_DELETE(m_entity_set);
}

void World::initialize()
{
    m_rand.initialize(0);
    m_camera.setPosition(vec4(1.0f, 1.0f, 3.0f, 0.0f));
    m_camera.setZNear(0.01f);
    m_camera.setZFar(10.0f);

    m_sph->initialize();
}


void World::kickAsyncUpdate(float32 dt)
{
    m_task_update_world->setArg(dt);
    m_task_update_entity->setArg(dt);
    m_task_update_sph->setArg(dt);

    m_task_update_world->kick();
    m_task_update_entity->kick();
    m_task_update_sph->kick();
}

void World::joinAsyncUpdate()
{
    m_task_update_sph->join();
    m_task_update_entity->join();
    m_task_update_world->join();
}

inline vec4 GenRotateAxis()
{
    vec4 axis( atomicGenRandFloat(), atomicGenRandFloat(), atomicGenRandFloat(), 0.0f );
    axis -= vec4(0.5f, 0.5f, 0.5f, 0.0f);
    axis *= 2.0f;
    return glm::normalize(axis);
}

void World::update(float32 dt)
{
    ++m_frame;

    if(m_frame==1) {
        {
            IEntity *e =  m_entity_set->createEntity<Enemy_Cube>();
            e->call(ECALL_setPosition, vec4(0.5f, 0.0f, 0.0f, 1.0f));
            e->call(ECALL_setAxis1, GenRotateAxis());
            e->call(ECALL_setAxis2, GenRotateAxis());
            e->call(ECALL_setRotateSpeed1, 0.1f);
            e->call(ECALL_setRotateSpeed2, 0.1f);
        }
        {
            IEntity *e =  m_entity_set->createEntity<Enemy_Sphere>();
            e->call(ECALL_setPosition, vec4(-0.5f, 0.0f, 0.0f, 1.0f));
            e->call(ECALL_setAxis1, GenRotateAxis());
            e->call(ECALL_setAxis2, GenRotateAxis());
            e->call(ECALL_setRotateSpeed1, 0.1f);
            e->call(ECALL_setRotateSpeed2, 0.1f);
        }
    }

    m_sph->updateBegin(dt);

    m_entity_set->update(dt);
    m_sph->update(dt);

    kickAsyncUpdate(dt);
    joinAsyncUpdate();
}

void World::updateAsync(float32 dt)
{
    //mat4 rot = glm::rotate(mat4(), 0.05f, vec3(0.0f, 1.0f, 0.0f));
    //m_camera.setPosition(rot * m_camera.getPosition());
    m_camera.setAspect(atomicGetWindowAspectRatio());
}

void World::draw() const
{
    m_sph->draw();

    DirectionalLight dl;
    dl.direction = glm::normalize(vec4(1.0f, -1.0f, -0.5f, 0.0f));
    dl.diffuse_color = vec4(0.3f, 0.3f, 0.3f, 1.0f);
    dl.ambient_color = vec4(0.0f, 0.0f, 0.0f, 0.0f);
    atomicGetDirectionalLights()->pushInstance(dl);
}


} // namespace atomic
