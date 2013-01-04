#include "stdafx.h"
#include "types.h"
#include "AtomicGame.h"
#include "Graphics/ResourceManager.h"
#include "Entity.h"
#include "Task.h"

#ifdef __atomic_enable_strict_handle_check__
    #define atomicStrictHandleCheck(h) if(!isValidHandle(h)) { istAssert("invalid entity handle\n"); }
#else
    #define atomicStrictHandleCheck(h)
#endif

namespace atomic {


class EntityUpdateTask : public AtomicTask
{
private:
    EntitySet *m_eset;
    EntityHandle *m_begin, *m_end;
    float32 m_dt;

public:
    EntityUpdateTask(EntitySet *eset) : m_eset(eset) {}
    void setup(EntityHandle *begin, EntityHandle *end, float32 dt)
    {
        m_begin = begin;
        m_end = end;
        m_dt = dt;
    }

    void exec()
    {
        const float32 dt = m_dt;
        for(EntityHandle *i=m_begin; i!=m_end; ++i) {
            if(IEntity *e=m_eset->getEntity(*i)) {
                e->asyncupdate(dt);
            }
        }
    }
};



void EntitySet::addEntity( uint32 categoryid, uint32 classid, IEntity *e )
{
    atomicAssertSyncLock("");
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

void EntitySet::resizeTasks( uint32 n )
{
    while(m_tasks.size() < n) {
        m_tasks.push_back( istNew(EntityUpdateTask)(this) );
    }
}

EntitySet::EntitySet()
{

}

EntitySet::~EntitySet()
{
    for(uint32 i=0; i<ECID_End; ++i) {
        for(uint32 j=0; j<ESID_MAX; ++j) {
            EntityCont &entities = m_entities[i][j];
            uint32 s = entities.size();
            for(uint32 k=0; k<s; ++k) {
                istSafeDelete(entities[k]);
            }
            entities.clear();
            m_vacant[i][j].clear();
        }
    }
    m_all.clear();

    for(uint32 i=0; i<m_tasks.size(); ++i) {
        istSafeDelete(m_tasks[i]);
    }
    m_tasks.clear();
}

void EntitySet::frameBegin()
{
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


    // asyncupdate
    atomicLockSyncMethods();
    ist::parallel_for(
        size_t(0), m_all.size(), 32,
        [&](size_t first, size_t last) {
            for(size_t i=first; i!=last; ++i) {
                if(IEntity *e=getEntity(m_all[i])) {
                    e->asyncupdate(dt);
                }
            }
        });
    atomicUnlockSyncMethods();
}

void EntitySet::asyncupdate(float32 dt)
{
}

void EntitySet::draw()
{
    for(uint32 i=0; i<ECID_End; ++i) {
        for(uint32 j=0; j<ESID_MAX; ++j) {
            EntityCont &entities = m_entities[i][j];
            uint32 s = entities.size();
            for(uint32 k=0; k<s; ++k) {
                IEntity *entity = entities[k];
                if(entity) { entity->draw(); }
            }
        }
    }
}

void EntitySet::frameEnd()
{
}


IEntity* EntitySet::getEntity( EntityHandle h )
{
    if(h==0) { return NULL; }
    uint32 cid = EntityGetCategory(h);
    uint32 sid = EntityGetClass(h);
    uint32 iid = EntityGetID(h);

    if(iid >= m_entities[cid][sid].size()) {
        return NULL;
    }
    return m_entities[cid][sid][iid];
}

void EntitySet::deleteEntity( EntityHandle h )
{
    atomicAssertSyncLock("");
    uint32 cid = EntityGetCategory(h);
    uint32 sid = EntityGetClass(h);
    uint32 iid = EntityGetID(h);
    EntityCont &entities = m_entities[cid][sid];
    HandleCont &vacants = m_vacant[cid][sid];
    entities[iid]->finalize();
    istSafeDelete(entities[iid]);
    vacants.push_back(h);
}

} // namespace atomic
