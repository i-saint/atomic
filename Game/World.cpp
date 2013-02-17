#include "stdafx.h"
#include "ist/ist.h"
#include "types.h"
#include "AtomicApplication.h"
#include "AtomicGame.h"
#include "Game/Message.h"
#include "Game/SPHManager.h"
#include "Game/Entity.h"
#include "Game/Collision.h"
#include "Game/VFX.h"
#include "Game/World.h"
#include "Graphics/Renderer.h"
#include "EntityQuery.h"
#include "EntityClass.h"
#include "Util.h"

using namespace ist::i3dgl;


namespace atomic {



World::World()
: m_collision_set(NULL)
, m_entity_set(NULL)
, m_sph(NULL)
, m_vfx(NULL)
{
    m_collision_set = istNew(CollisionSet)();
    m_modules.push_back(m_collision_set);

    m_sph = istNew(SPHManager)();
    m_modules.push_back(m_sph);

    m_entity_set = istNew(EntitySet)();
    m_modules.push_back(m_entity_set);

    m_vfx = istNew(VFXSet)();
    m_modules.push_back(m_vfx);

    const uvec2 &wsize = atomicGetWindowSize();
    m_camera_game.setAspect((float32)wsize.x/(float32)wsize.y);

    istCommandlineRegister("setCameraFovy", &i3d::PerspectiveCamera::setFovy, &m_camera_game);
}

World::~World()
{
    istCommandlineUnregister("setCameraFovy");

    for(ModuleCont::reverse_iterator i=m_modules.rbegin(); i!=m_modules.rend(); ++i) {
        istDelete(*i);
    }
}

void World::initialize()
{
    m_camera_game.setPosition(vec4(0.0f, 0.0f, 3.0f, 0.0f));
    m_camera_game.setZNear(0.01f);
    m_camera_game.setZFar(10.0f);
}


void World::frameBegin()
{
    m_collision_set->frameBegin();
    m_sph->frameBegin();
    m_entity_set->frameBegin();
}

void World::update(float32 dt)
{
    if(atomicGetFrame()==0) {
        m_entity_set->createEntity<Level_Test>();
    }

    for(ModuleCont::iterator i=m_modules.begin(); i!=m_modules.end(); ++i) {
        (*i)->update(dt);
    }

    m_camera_game.updateMatrix();
    m_frustum.constructFromViewProjectionMatrix(m_camera_game.getViewProjectionMatrix());
}

void World::asyncupdateBegin(float32 dt)
{
    m_asyncupdate.run(std::bind(&World::asyncupdate, this, dt));
}

void World::asyncupdateEnd()
{
    m_asyncupdate.wait();
}

void World::asyncupdate(float32 dt)
{
    ist::parallel_for(size_t(0), m_modules.size(),
        [&](size_t i){
            m_modules[i]->asyncupdate(dt);
        });
}

void World::draw()
{
    for(ModuleCont::iterator i=m_modules.begin(); i!=m_modules.end(); ++i) {
        (*i)->draw();
    }
}

void World::frameEnd()
{
    for(ModuleCont::reverse_iterator i=m_modules.rbegin(); i!=m_modules.rend(); ++i) {
        (*i)->frameEnd();
    }
}

void World::handleEntitiesQuery( EntitiesQueryContext &ctx )
{
    m_entity_set->handleEntitiesQuery(ctx);
}


} // namespace atomic
