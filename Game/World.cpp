#include "stdafx.h"
#include "ist/ist.h"
#include "types.h"
#include "AtomicApplication.h"
#include "AtomicGame.h"
#include "Game/Message.h"
#include "Game/SPHManager.h"
#include "Game/Entity.h"
#include "Game/Collision.h"
#include "Game/World.h"
#include "Graphics/Renderer.h"
#include "EntityQuery.h"
#include "EntityClass.h"
#include "Util.h"

using namespace ist::graphics;


namespace atomic {




World::World()
: m_entity_set(NULL)
, m_collision_set(NULL)
, m_sph(NULL)
, m_frame(0)
{
    m_entity_set    = IST_NEW(EntitySet)();
    m_collision_set = IST_NEW(CollisionSet)();
    m_sph           = IST_NEW(SPHManager)();

    m_task_update_world     = IST_NEW(Task_UpdateAsync<World>)(this);
    m_task_update_entity    = IST_NEW(Task_UpdateAsync<EntitySet>)(m_entity_set);
    m_task_update_collision = IST_NEW(Task_UpdateAsync<CollisionSet>)(m_collision_set);
    m_task_update_sph       = IST_NEW(Task_UpdateAsync<SPHManager>)(m_sph);

    m_camera.setAspect(atomicGetWindowAspectRatio());
}

World::~World()
{
    IST_SAFE_DELETE(m_task_update_sph);
    IST_SAFE_DELETE(m_task_update_collision);
    IST_SAFE_DELETE(m_task_update_entity);
    IST_SAFE_DELETE(m_task_update_world);

    IST_SAFE_DELETE(m_sph);
    IST_SAFE_DELETE(m_collision_set);
    IST_SAFE_DELETE(m_entity_set);
}

void World::initialize()
{
    m_rand.initialize(0);
    m_camera.setPosition(vec4(0.0f, 0.0f, 3.0f, 0.0f));
    m_camera.setZNear(0.01f);
    m_camera.setZFar(10.0f);

    m_sph->initialize();
}


void World::update(float32 dt)
{
    ++m_frame;

    if(m_frame==1) {
        m_entity_set->createEntity<Level_Test>();
    }

    m_entity_set->updateBegin(dt);
    m_collision_set->updateBegin(dt);
    m_sph->updateBegin(dt);

    m_entity_set->update(dt);
    m_collision_set->update(dt);
    m_sph->update(dt);

    m_collision_set->updateEnd();
}

void World::asyncupdateBegin(float32 dt)
{
    m_task_update_world->setArg(dt);
    m_task_update_entity->setArg(dt);
    m_task_update_sph->setArg(dt);

    m_task_update_world->kick();
    m_task_update_entity->kick();
    m_task_update_collision->kick();
    m_task_update_sph->kick();
}

void World::asyncupdateEnd()
{
    m_task_update_sph->join();
    m_task_update_collision->join();
    m_task_update_entity->join();
    m_task_update_world->join();
}

void World::asyncupdate(float32 dt)
{
}

void World::draw() const
{
    m_sph->draw();
    m_entity_set->draw();

    DirectionalLight dl;
    dl.direction = glm::normalize(vec4(1.0f, -1.0f, -0.5f, 0.0f));
    dl.diffuse_color = vec4(0.3f, 0.3f, 0.3f, 1.0f);
    dl.ambient_color = vec4(0.0f, 0.0f, 0.0f, 0.0f);
    atomicGetDirectionalLights()->addInstance(dl);
}


} // namespace atomic
