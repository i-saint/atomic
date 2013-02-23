#include "stdafx.h"
#include "types.h"
#include "AtomicGame.h"
#include "Graphics/ResourceManager.h"
#include "Entity.h"
#include "EntityQuery.h"
#include "Collision.h"
#include "World.h"
#include "Task.h"

#ifdef atomic_enable_strict_handle_check
    #define atomicStrictHandleCheck(h) if(!isValidHandle(h)) { istAssert("invalid entity handle\n"); }
#else
    #define atomicStrictHandleCheck(h)
#endif

namespace atomic {


EntityCreator* GetEntityCreatorTable( EntityClassID entity_classid )
{
    static EntityCreator s_table_player[EC_Player_End & 0x1FF];
    static EntityCreator s_table_enemy[EC_Enemy_End & 0x1FF];
    static EntityCreator s_table_bullet[EC_Bullet_End & 0x1FF];
    static EntityCreator s_table_obstacle[EC_Obstacle_End & 0x1FF];
    static EntityCreator s_table_level[EC_Level_End & 0x1FF];
    static EntityCreator s_table_vfx[EC_VFX_End & 0x1FF];
    static EntityCreator *s_table_list[ECA_End] = {
        NULL,
        s_table_player,
        s_table_enemy,
        s_table_bullet,
        s_table_obstacle,
        s_table_level,
        s_table_vfx,
    };
    return s_table_list[(entity_classid & 0xE00) >> 9];
}

void AddEntityCreator(EntityClassID entity_classid, EntityCreator creator)
{
    GetEntityCreatorTable(entity_classid)[entity_classid & 0x1FF] = creator;
}

IEntity* CreateEntity( EntityClassID entity_classid )
{
    return GetEntityCreatorTable(entity_classid)[entity_classid & 0x1FF]();
}



IEntity::IEntity()
    : m_ehandle(atomicGetEntitySet()->getGeneratedHandle())
{
}



EntitySet::EntitySet()
{
}

EntitySet::~EntitySet()
{
    EntityCont &entities = m_entities;
    uint32 s = entities.size();
    for(uint32 k=0; k<s; ++k) {
        istSafeDelete(entities[k]);
    }
    entities.clear();
    m_vacant.clear();

    m_all.clear();
}

void EntitySet::frameBegin()
{
}

void EntitySet::update( float32 dt )
{
    // update
    uint32 num_entities = m_all.size();
    for(uint32 i=0; i<num_entities; ++i) {
        if(IEntity *entity = getEntity(m_all[i])) {
            entity->update(dt);
        }
        else {
            m_all[i] = 0;
        }
    }

    // erase invalid handles
    m_all.erase(stl::remove(m_all.begin(), m_all.end(), 0), m_all.end());

    // append new entities
    for(uint32 i=0; i<m_new_entities.size(); ++i) {
        IEntity *entity = m_new_entities[i];
        EntityHandle handle = entity->getHandle();
        m_entities[EntityGetIndex(handle)] = entity;
        m_all.push_back(handle);
        entity->update(dt);
    }
    m_new_entities.clear();


    // asyncupdate
    atomicDbgLockSyncMethods();
    ist::parallel_for(
        ist::size_range(size_t(0), m_all.size(), 32),
        [&](const ist::size_range &r) {
            for(size_t i=r.begin(); i!=r.end(); ++i) {
                if(IEntity *e=getEntity(m_all[i])) {
                    e->asyncupdate(dt);
                }
            }
        });
    atomicDbgUnlockSyncMethods();
}

void EntitySet::asyncupdate(float32 dt)
{
}

void EntitySet::draw()
{
    uint32 s = m_entities.size();
    for(uint32 k=0; k<s; ++k) {
        IEntity *entity = m_entities[k];
        if(entity) { entity->draw(); }
    }
}

void EntitySet::frameEnd()
{
}


IEntity* EntitySet::getEntity( EntityHandle h )
{
    if(h==0) { return NULL; }
    uint32 cid = EntityGetClassID(h);
    uint32 iid = EntityGetIndex(h);

    EntityCont &entities = m_entities;
    if(iid >= entities.size()) {
        return NULL;
    }
    return entities[iid];
}

void EntitySet::deleteEntity( EntityHandle h )
{
    atomicDbgAssertSyncLock();
    uint32 cid = EntityGetClassID(h);
    uint32 iid = EntityGetIndex(h);
    EntityCont &entities = m_entities;
    HandleCont &vacants = m_vacant;
    entities[iid]->finalize();
    istSafeDelete(entities[iid]);
    vacants.push_back(EntityGetIndex(h));
}

void EntitySet::generateHandle(EntityClassID classid)
{
    atomicDbgAssertSyncLock();
    EntityCont &entities = m_entities;
    HandleCont &vacant = m_vacant;
    EntityHandle h = 0;
    if(!vacant.empty()) {
        h = vacant.back();
        vacant.pop_back();
    }
    else {
        h = entities.size();
    }
    entities.push_back(NULL); // reserve
    h = EntityCreateHandle(classid, h);
    m_tmp_handle = h;
}

EntityHandle EntitySet::getGeneratedHandle()
{
    return m_tmp_handle;
}

IEntity* EntitySet::createEntity( EntityClassID classid )
{
    generateHandle(classid);
    IEntity *e = CreateEntity(classid);
    m_new_entities.push_back(e);
    e->initialize();
    return e;
}


void EntitySet::handleEntitiesQuery( EntitiesQueryContext &ctx )
{
    uint32 num_entities = m_all.size();
    for(uint32 i=0; i<num_entities; ++i) {
        EntityHandle handle = m_all[i];
        IEntity *entity = getEntity(handle);
        if(entity) {
            variant var;
            if(!entity->query(FID_getCollisionHandle, var)) { continue; }
            CollisionHandle ch = var.cast<CollisionHandle>();
            CollisionEntity *ce = atomicGetCollision(ch);
            if(ce) {
                const BoundingBox &bb = ce->bb;
                vec4 bb_size = bb.ur - bb.bl;
                vec4 bb_pos = (bb.ur + bb.bl) * 0.5f;
                ctx.id.push_back( entity->getHandle() );
                ctx.type.push_back( EntityGetCategory(entity->getHandle()) );
                ctx.size.push_back( vec2(bb_size) );
                ctx.pos.push_back( vec2(bb_pos) );
            }
        }
    }
}


} // namespace atomic
