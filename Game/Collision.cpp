#include "stdafx.h"
#include "types.h"
#include "Task.h"
#include "Game/AtomicApplication.h"
#include "Game/AtomicGame.h"
#include "Game/World.h"
#include "Game/SPHManager.h"
#include "Collision.h"

namespace atomic {

class CollisionTask : public Task
{
private:
    typedef stl::vector<CollisionEntity*>::iterator CollisionIterator;
    typedef stl::vector<CollisionMessage> MessageCont;
    CollisionSet        *m_manager;
    MessageCont         m_messages;
    CollisionIterator   m_begin;
    CollisionIterator   m_end;

public:
    CollisionTask(CollisionSet *m) : m_manager(m) {}
    void setup(CollisionIterator begin, CollisionIterator end) { m_begin=begin; m_end=end; }

    void exec()
    {
        for(CollisionIterator i=m_begin; i!=m_end; ++i) {
            CollisionEntity *ce = *i;
            if(!ce) { continue; }

        }
    }
};



CollisionSet::CollisionSet()
{
    m_tasks.reserve(32);
    m_entities.reserve(1024);
    m_vacant.reserve(1024);
}

CollisionSet::~CollisionSet()
{
    for(uint32 i=0; i<m_tasks.size(); ++i)      { IST_DELETE(m_tasks[i]); }
    for(uint32 i=0; i<m_entities.size(); ++i)   { IST_DELETE(m_entities[i]); }
    m_tasks.clear();
    m_entities.clear();
    m_vacant.clear();
}

void CollisionSet::updateBegin(float32 dt)
{
    // message passing
}

void CollisionSet::update(float32 dt)
{
}

void CollisionSet::updateEnd()
{
    uint32 num = m_entities.size();
    for(uint32 i=0; i<num; ++i) {
        const CollisionEntity *ce = m_entities[i];
        if(!ce) { continue; }

        switch(ce->getShape()) {
        case CS_PLANE:  break;
        case CS_SPHERE: atomicGetSPHManager()->addRigid(reinterpret_cast<const sphRigidSphere&>(*ce)); break;
        case CS_BOX:    atomicGetSPHManager()->addRigid(reinterpret_cast<const sphRigidBox&>(*ce)); break;
        }
    }
}

void CollisionSet::asyncupdate(float32 dt)
{
}

void CollisionSet::addEntity(CollisionEntity *e)
{
    CollisionHandle h = 0;
    if(!m_vacant.empty()) {
        h = m_vacant.back();
        m_vacant.pop_back();
        m_entities[h] = e;
    }
    else {
        h = (CollisionHandle)m_entities.size();
        m_entities.push_back(e);
    }
    e->SetCollisionHandle(h);

}

template<> CollisionPlane* CollisionSet::createEntity<CollisionPlane>()
{
    CollisionPlane *e = IST_NEW(CollisionPlane)();
    addEntity(e);
    return e;
}

template<> CollisionSphere* CollisionSet::createEntity<CollisionSphere>()
{
    CollisionSphere *e = IST_NEW(CollisionSphere)();
    addEntity(e);
    return e;
}

template<> CollisionBox* CollisionSet::createEntity<CollisionBox>()
{
    CollisionBox *e = IST_NEW(CollisionBox)();
    addEntity(e);
    return e;
}

void CollisionSet::deleteEntity(CollisionEntity *e)
{
    CollisionHandle h = e->getCollisionHandle();
    IST_SAFE_DELETE(m_entities[h]);
    m_vacant.push_back(h);
}



} // namespace atomic
