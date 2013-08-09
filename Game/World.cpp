#include "stdafx.h"
#include "ist/ist.h"
#include "types.h"
#include "AtomicApplication.h"
#include "AtomicGame.h"
#include "Game/Message.h"
#include "Game/FluidModule.h"
#include "Game/EntityModule.h"
#include "Game/CollisionModule.h"
#include "Game/BulletModule.h"
#include "Game/VFXModule.h"
#include "Game/World.h"
#include "Graphics/Renderer.h"
#include "EntityQuery.h"
#include "EntityClass.h"
#include "Util.h"

using namespace ist::i3dgl;


namespace atm {

atmExportClass(World);

istSerializeBlockImpl(World,
    istSerializeBase(IAtomicGameModule)
    istSerialize(m_collision_module)
    istSerialize(m_fluid_module)
    istSerialize(m_entity_module)
    istSerialize(m_bullet_module)
    istSerialize(m_vfx_module)
    istSerialize(m_modules)
    istSerialize(m_camera_game)
    istSerialize(m_camera_bg)
    istSerialize(m_field_size)
)

World::World()
: m_collision_module(nullptr)
, m_fluid_module(nullptr)
, m_entity_module(nullptr)
, m_bullet_module(nullptr)
, m_vfx_module(nullptr)
{
    wdmAddNode("Game/World/cameraFovy", &m_camera_game, &i3d::PerspectiveCamera::getFovy, &i3d::PerspectiveCamera::setFovy, 1.0f, 180.0f);
}

World::~World()
{
    istCommandlineUnregister("setCameraFovy");

    for(ModuleCont::reverse_iterator i=m_modules.rbegin(); i!=m_modules.rend(); ++i) {
        (*i)->finalize();
    }
    for(ModuleCont::reverse_iterator i=m_modules.rbegin(); i!=m_modules.rend(); ++i) {
        (*i)->release();
    }

    wdmEraseNode("Game/World");
}

void World::initialize()
{
    m_collision_module = istNew(CollisionModule)();
    m_modules.push_back(m_collision_module);

    m_fluid_module = istNew(FluidModule)();
    m_modules.push_back(m_fluid_module);

    m_bullet_module = istNew(BulletModule)();
    m_modules.push_back(m_bullet_module);

    m_vfx_module = istNew(VFXModule)();
    m_modules.push_back(m_vfx_module);

    m_entity_module = istNew(EntityModule)();
    m_modules.push_back(m_entity_module);

    for(ModuleCont::iterator i=m_modules.begin(); i!=m_modules.end(); ++i) {
        (*i)->initialize();
    }

    const uvec2 &wsize = atmGetWindowSize();
    m_camera_game.setAspect((float32)wsize.x/(float32)wsize.y);
    m_camera_game.setPosition(vec3(0.0f, 0.0f, 3.0f));
    m_camera_game.setZNear(0.01f);
    m_camera_game.setZFar(10.0f);
}


void World::frameBegin()
{
    m_collision_module->frameBegin();
    m_fluid_module->frameBegin();
    m_entity_module->frameBegin();
}

void World::update(float32 dt)
{
    if(atmGetFrame()==0) {
        atmCreateEntityT(LevelTest);
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
    atmGetRenderer()->setTime((float32)atmGetFrame());
    atmGetRenderer()->setGameCamera(m_camera_game);

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
    if(m_modules.empty()) { return; }
    m_entity_module->handleStateQuery(ctx);
    m_bullet_module->handleStateQuery(ctx);
    m_fluid_module->handleStateQuery(ctx);
}


} // namespace atm
