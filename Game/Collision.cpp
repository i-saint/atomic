#include "stdafx.h"
#include "types.h"
#include "Task.h"
#include "Collision.h"

namespace atomic {

class CollisionTask : public Task
{
private:
    typedef stl::vector<CollisionEntity*>::iterator CollisionIterator;
    CollisionSet                *m_manager;
    stl::vector<CollisionMessage>   m_messages;
    CollisionIterator               m_begin;
    CollisionIterator               m_end;

public:
    CollisionTask(CollisionSet *m) : m_manager(m) {}
    void setup(CollisionIterator begin, CollisionIterator end) { m_begin=begin; m_end=end; }

    void exec()
    {
        for(CollisionIterator i=m_begin; i!=m_end; ++i) {

        }
    }
};



CollisionSet::CollisionSet()
{

}

CollisionSet::~CollisionSet()
{
    for(uint32 i=0; i<m_tasks.size(); ++i)      { IST_DELETE(m_tasks[i]); }
    for(uint32 i=0; i<m_entities.size(); ++i)   { IST_DELETE(m_entities[i]); }
    m_tasks.clear();
    m_entities.clear();
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
    m_sph_spheres.clear();
    m_sph_boxes.clear();

    uint32 num = m_entities.size();
    for(uint32 i=0; i<num; ++i) {
        const CollisionEntity *ce = m_entities[i];
        switch(ce->getShape()) {
        case CS_PLANE:  break;
        case CS_SPHERE: m_sph_spheres.push_back(reinterpret_cast<const sphRigidSphere&>(*ce)); break;
        case CS_BOX:    m_sph_boxes.push_back(reinterpret_cast<const sphRigidBox&>(*ce)); break;
        }
    }
}

void CollisionSet::asyncupdate(float32 dt)
{
    m_entities.erase(std::remove(m_entities.begin(), m_entities.end(), (CollisionEntity*)NULL), m_entities.end());
}

void CollisionSet::addEntity(CollisionEntity *e)
{

}

template<> CollisionPlane* CollisionSet::createEntity<CollisionPlane>()
{
    CollisionPlane *p = IST_NEW(CollisionPlane)();
    p->SetCollisionHandle((CollisionHandle)m_entities.size());
    m_entities.push_back(p);
    return p;
}

template<> CollisionSphere* CollisionSet::createEntity<CollisionSphere>()
{
    CollisionSphere *p = IST_NEW(CollisionSphere)();
    p->SetCollisionHandle((CollisionHandle)m_entities.size());
    m_entities.push_back(p);
    return p;
}

template<> CollisionBox* CollisionSet::createEntity<CollisionBox>()
{
    CollisionBox *p = IST_NEW(CollisionBox)();
    p->SetCollisionHandle((CollisionHandle)m_entities.size());
    m_entities.push_back(p);
    return p;
}

void CollisionSet::deleteEntity(CollisionHandle h)
{
    IST_SAFE_DELETE(m_entities[h]);
}



} // namespace atomic
