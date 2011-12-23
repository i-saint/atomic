#include "stdafx.h"
#include "types.h"
#include "AtomicGame.h"
#include "Graphics/ResourceManager.h"
#include "Entity.h"

#ifdef __atomic_EnableStrictHandleCheck__
    #define atomicStrictHandleCheck(h) if(!isValidHandle(h)) { IST_ASSERT("invalid entity handle\n"); }
#else
    #define atomicStrictHandleCheck(h)
#endif

namespace atomic {


void EntitySet::addEntity( uint32 categoryid, uint32 classid, IEntity *e )
{
    EntityCont &entities = m_entities[categoryid][classid];
    HandleCont &vacant = m_vacant[categoryid][classid];
    EntityHandle h = 0;
    if(!vacant.empty()) {
        h = vacant.back();
        vacant.pop_back();
    }
    else {
        h = EntityCreateHandle(categoryid, classid, entities.size());
    }
    e->setHandle(h);
    e->initialize();
    entities.push_back(NULL); // reserve
    m_new_entities.push_back(e);
}

EntitySet::EntitySet()
{

}

EntitySet::~EntitySet()
{
    for(uint32 i=0; i<ECID_END; ++i) {
        for(uint32 j=0; j<ESID_MAX; ++j) {
            EntityCont &entities = m_entities[i][j];
            uint32 s = entities.size();
            for(uint32 k=0; k<s; ++k) {
                IST_SAFE_DELETE(entities[k]);
            }
            entities.clear();
            m_vacant[i][j].clear();
        }
    }
    m_all.clear();
}

void EntitySet::update( float32 dt )
{
    // update
    uint32 num_entities = m_all.size();
    for(uint32 i=0; i<num_entities; ++i) {
        EntityHandle handle = m_all[i];
        IEntity *entity = getEntity(handle);
        if(entity) {
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
        uint32 cid = EntityGetCategory(handle);
        uint32 sid = EntityGetClass(handle);
        uint32 iid = EntityGetID(handle);
        EntityCont &entities = m_entities[cid][sid];
        entities[iid] = entity;
        entity->update(dt);
        m_all.push_back(handle);
    }
    m_new_entities.clear();


    // todo updateAsync()
}

void EntitySet::sync()
{
}

IEntity* EntitySet::getEntity( EntityHandle h )
{
    if(h==0) { return NULL; }
    uint32 cid = EntityGetCategory(h);
    uint32 sid = EntityGetClass(h);
    uint32 iid = EntityGetID(h);
    return m_entities[cid][sid][iid];
}

void EntitySet::deleteEntity( EntityHandle h )
{
    uint32 cid = EntityGetCategory(h);
    uint32 sid = EntityGetClass(h);
    uint32 iid = EntityGetID(h);
    EntityCont &entities = m_entities[cid][sid];
    HandleCont &vacants = m_vacant[cid][sid];
    entities[iid]->finalize();
    IST_SAFE_DELETE(entities[iid]);
    vacants.push_back(h);
}

} // namespace atomic
