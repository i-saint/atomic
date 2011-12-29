#include "stdafx.h"
#include "types.h"
#include "Task.h"
#include "Game/AtomicApplication.h"
#include "Game/AtomicGame.h"
#include "Game/World.h"
#include "Game/Entity.h"
#include "Game/EntityQuery.h"
#include "Game/SPHManager.h"
#include "Collision.h"

namespace atomic {

inline bool BoundingBoxIntersect(const BoundingBox &bb1, const vec4 &pos)
{
    return 
        pos.x <= bb1.ur.x && pos.x >= bb1.bl.x &&
        pos.y <= bb1.ur.y && pos.y >= bb1.bl.y &&
        pos.z <= bb1.ur.z && pos.z >= bb1.bl.z;
}

inline bool BoundingBoxIntersect(const BoundingBox &bb1, const BoundingBox &bb2)
{
    float32 rabx = std::abs(bb1.ur.x + bb1.bl.x - bb2.ur.x - bb2.bl.x);
    float32 raby = std::abs(bb1.ur.y + bb1.bl.y - bb2.ur.y - bb2.bl.y);
    float32 rabz = std::abs(bb1.ur.z + bb1.bl.z - bb2.ur.z - bb2.bl.z);
    float32 raxPrbx = bb1.ur.x - bb1.bl.x + bb2.ur.x - bb2.bl.x;
    float32 rayPrby = bb1.ur.y - bb1.bl.y + bb2.ur.y - bb2.bl.y;
    float32 rayPrbz = bb1.ur.z - bb1.bl.z + bb2.ur.z - bb2.bl.z;
    return (rabx <= raxPrbx && raby <= rayPrby && rabz <= rayPrbz);
}


bool _Collide(CollisionPlane *sender, CollisionPlane *receiver, CollideMessage &m)
{
    float32 d = glm::dot(vec3(sender->plane), vec3(receiver->plane));
    if(std::abs(d) < 1.0f) {
        m.direction = vec4(vec3(sender->plane), 0.0f);
        return true;
    }
    return false;
}

bool _Collide(CollisionPlane *sender, CollisionSphere *receiver, CollideMessage &m)
{
    vec3 spos = vec3(receiver->pos_r);
    float32 radius = receiver->pos_r.w;
    float32 d = glm::dot(vec4(spos, 1.0f), sender->plane) - radius;
    if(d <= 0.0f) {
        m.direction = vec4(vec3(sender->plane), -d);
        return true;
    }
    return false;
}

bool _Collide(CollisionPlane *sender, CollisionBox *receiver, CollideMessage &m)
{
    return false;
}


bool _Collide(CollisionSphere *sender, CollisionPlane *receiver, CollideMessage &m)
{
    if(_Collide(receiver, sender, m)) {
        //m.direction *= vec4(-1.0f, -1.0f, -1.0f, 1.0f);
        return true;
    }
    return false;
}

bool _Collide(CollisionSphere *sender, CollisionSphere *receiver, CollideMessage &m)
{
    float32 r = sender->pos_r.w + receiver->pos_r.w;
    float32 r2 = r*r;
    vec3 diff = vec3(sender->pos_r) - vec3(receiver->pos_r);
    float32 d = glm::dot(diff, diff);
    if(d <= r2) {
        float32 len = std::sqrt(d);
        vec3 n = diff / len * -1.0f;
        m.direction = vec4(n, r-len);
        return true;
    }
    return false;
}

bool _Collide(CollisionSphere *sender, CollisionBox *receiver, CollideMessage &m)
{
    return false;
}


bool _Collide(CollisionBox *sender, CollisionPlane *receiver, CollideMessage &m)
{
    return false;
}

bool _Collide(CollisionBox *sender, CollisionSphere *receiver, CollideMessage &m)
{
    if(!BoundingBoxIntersect(sender->bb, receiver->bb)) { return false; }

    vec4 pos = receiver->pos_r - sender->position;
    pos.w = 1.0f;

    int inside = 0;
    int closest_index = 0;
    float closest_dinstance = -9999.0f;
    for(int p=0; p<6; ++p) {
        vec4 plane = sender->planes[p];
        float32 radius = receiver->pos_r.w;
        float d = glm::dot(pos, plane) - radius;
        if(d <= 0.0f) {
            ++inside;
            if(d > closest_dinstance) {
                closest_dinstance = d;
                closest_index = p;
            }
        }
    }
    if(inside==6) {
        vec4 dir = sender->planes[closest_index];
        dir.w = -closest_dinstance;
        m.direction = dir;
        return true;
    }
    return false;
}

bool _Collide(CollisionBox *sender, CollisionBox *receiver, CollideMessage &m)
{
    return false;
}


bool Collide(CollisionEntity *sender, CollisionEntity *receiver, CollideMessage &m)
{
    switch(sender->getShape()) {
    case CS_PLANE:
        switch(receiver->getShape()) {
        case CS_PLANE:  return _Collide(static_cast<CollisionPlane*>(sender), static_cast<CollisionPlane*>(receiver), m);
        case CS_SPHERE: return _Collide(static_cast<CollisionPlane*>(sender), static_cast<CollisionSphere*>(receiver), m);
        case CS_BOX:    return _Collide(static_cast<CollisionPlane*>(sender), static_cast<CollisionBox*>(receiver), m);
        }
        break;

    case CS_SPHERE:
        switch(receiver->getShape()) {
        case CS_PLANE:  return _Collide(static_cast<CollisionSphere*>(sender), static_cast<CollisionPlane*>(receiver), m);
        case CS_SPHERE: return _Collide(static_cast<CollisionSphere*>(sender), static_cast<CollisionSphere*>(receiver), m);
        case CS_BOX:    return _Collide(static_cast<CollisionSphere*>(sender), static_cast<CollisionBox*>(receiver), m);
        }
        break;

    case CS_BOX:
        switch(receiver->getShape()) {
        case CS_PLANE:  return _Collide(static_cast<CollisionBox*>(sender), static_cast<CollisionPlane*>(receiver), m);
        case CS_SPHERE: return _Collide(static_cast<CollisionBox*>(sender), static_cast<CollisionSphere*>(receiver), m);
        case CS_BOX:    return _Collide(static_cast<CollisionBox*>(sender), static_cast<CollisionBox*>(receiver), m);
        }
        break;
    }
    return false;
}



class CollideTask : public Task
{
private:
    typedef stl::vector<CollisionEntity*>::iterator CollisionIterator;
    typedef stl::vector<CollideMessage> MessageCont;
    CollisionSet        *m_manager;
    MessageCont         m_messages;
    CollisionIterator   m_begin;
    CollisionIterator   m_end;

public:
    CollideTask(CollisionSet *m) : m_manager(m) {}
    void setup(CollisionIterator begin, CollisionIterator end) { m_begin=begin; m_end=end; }

    void exec()
    {
        m_messages.clear();
        for(CollisionIterator i=m_begin; i!=m_end; ++i) {
            m_manager->collide(*i, m_messages);
        }
    }

    MessageCont& getMessages() { return m_messages; }
};


uint32 CollisionSet::collide(CollisionEntity *sender, MessageCont &m)
{
    if(!sender || (sender->getFlags() & CF_SENDER)==0 ) { return 0; }

    uint32 n = 0;
    CollideMessage message;
    for(uint32 i=0; i<m_entities.size(); ++i) {
        if(CollisionEntity *receiver = m_entities[i]) {
            if((receiver->getFlags() & CF_RECEIVER) == 0 ) { continue; }
            if(receiver->getGObjHandle() == sender->getGObjHandle()) { continue; }

            if(Collide(sender, receiver, message)) {
                message.to = receiver->getGObjHandle();
                message.from = sender->getGObjHandle();
                m.push_back(message);
                ++n;
            }
        }
    }
    return n;
}


CollisionSet::CollisionSet()
: m_active_tasks(0)
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
}

void CollisionSet::update(float32 dt)
{
    for(uint32 ti=0; ti<m_active_tasks; ++ti) {
        const MessageCont &messages = m_tasks[ti]->getMessages();
        uint32 num_messages = messages.size();
        for(uint32 mi=0; mi<num_messages; ++mi) {
            if(IEntity *e = atomicGetEntity(messages[mi].to)) {
                atomicCall(e, eventCollide, static_cast<const CollideMessage*>(&messages[mi]));
            }
        }
    }
}

void CollisionSet::updateEnd()
{
    uint32 num = m_entities.size();
    for(uint32 i=0; i<num; ++i) {
        const CollisionEntity *ce = m_entities[i];
        if(!ce || (ce->getFlags() & CF_AFFECT_SPH)==0) { continue; }

        // SPH 側の剛体情報とメモリレイアウト同じにしてるので強引に突っ込む
        switch(ce->getShape()) {
        case CS_PLANE:  atomicGetSPHManager()->addRigid(reinterpret_cast<const sphRigidPlane&>(*ce)); break;
        case CS_SPHERE: atomicGetSPHManager()->addRigid(reinterpret_cast<const sphRigidSphere&>(*ce)); break;
        case CS_BOX:    atomicGetSPHManager()->addRigid(reinterpret_cast<const sphRigidBox&>(*ce)); break;
        }
    }
}

void CollisionSet::resizeTasks(uint32 n)
{
    while(m_tasks.size() < n) {
        m_tasks.push_back(IST_NEW(CollideTask)(this));
    }
}

void CollisionSet::asyncupdate(float32 dt)
{
    const uint32 block_size = 32;
    uint32 num_entities = m_entities.size();
    m_active_tasks = num_entities / block_size + (num_entities%block_size==0 ? 0 : 1);
    resizeTasks(m_active_tasks);
    for(uint32 i=0; i<m_active_tasks; ++i) {
        m_tasks[i]->setup(m_entities.begin()+(block_size*i), m_entities.begin()+std::min<uint32>(block_size*(i+1), m_entities.size()));
        m_tasks[i]->kick();
    }
    for(uint32 i=0; i<m_active_tasks; ++i) {
        m_tasks[i]->join();
    }
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

CollisionEntity* CollisionSet::getEntity(CollisionHandle h)
{
    if(h==0) { return NULL; }
    return m_entities[h];
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

void CollisionSet::deleteEntity(CollisionHandle h)
{
    IST_SAFE_DELETE(m_entities[h]);
    m_vacant.push_back(h);
}

void CollisionSet::deleteEntity(CollisionEntity *e)
{
    deleteEntity(e->getCollisionHandle());
}


} // namespace atomic
