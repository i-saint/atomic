﻿#include "atmPCH.h"
#include "types.h"
#include "Task.h"
#include "Util.h"
#include "Engine/Game/AtomicApplication.h"
#include "Engine/Game/AtomicGame.h"
#include "Engine/Game/World.h"
#include "Engine/Game/EntityModule.h"
#include "Engine/Game/EntityQuery.h"
#include "Engine/Game/FluidModule.h"
#include "CollisionModule.h"

namespace atm {

atmExportClass(CollisionEntity);
atmExportClass(CollisionPlane);
atmExportClass(CollisionSphere);
atmExportClass(CollisionBox);
atmExportClass(CollisionModule);


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


bool _Collide(const CollisionPlane *sender, const CollisionPlane *receiver, CollideMessage &m)
{
    float32 d = glm::dot(vec3(sender->plane), vec3(receiver->plane));
    if(std::abs(d) < 1.0f) {
        m.direction = vec4(vec3(sender->plane), 0.0f);
        return true;
    }
    return false;
}

bool _Collide(const CollisionPlane *sender, const CollisionSphere *receiver, CollideMessage &m)
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

bool _Collide(const CollisionPlane *sender, const CollisionBox *receiver, CollideMessage &m)
{
    return false;
}


bool _Collide(const CollisionSphere *sender, const CollisionPlane *receiver, CollideMessage &m)
{
    if(_Collide(receiver, sender, m)) {
        m.direction *= vec4(-1.0f, -1.0f, -1.0f, 1.0f);
        return true;
    }
    return false;
}

bool _Collide(const CollisionSphere *sender, const CollisionSphere *receiver, CollideMessage &m)
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


bool _Collide(const CollisionBox *sender, const CollisionPlane *receiver, CollideMessage &m)
{
    return false;
}

bool _Collide(const CollisionBox *sender, const CollisionSphere *receiver, CollideMessage &m)
{
    if(!BoundingBoxIntersect(sender->bb, receiver->bb)) { return false; }

    vec4 pos = receiver->pos_r - sender->position;
    pos.w = 1.0f;

    int inside = 0;
    int closest_index = 0;
    float closest_dinstance = -FLT_MAX;
    float closest = -FLT_MAX;
    for(int p=0; p<6; ++p) {
        vec4 plane = sender->planes[p];
        float32 radius = receiver->pos_r.w;
        float d = glm::dot(pos, plane) - radius;
        if(d <= 0.0f) {
            ++inside;
            float32 pd = d / (1.0f-glm::abs(plane.z)); // z 方向を向いてるものは優先度を下げる
            if(pd > closest) {
                closest = pd;
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

bool _Collide(const CollisionSphere *sender, const CollisionBox *receiver, CollideMessage &m)
{
    if(_Collide(receiver, sender, m)) {
        m.direction *= vec4(-1.0f, -1.0f, -1.0f, 1.0f);
        return true;
    }
    return false;
}

bool _Collide(const CollisionBox *sender, const CollisionBox *receiver, CollideMessage &m)
{
    if(!BoundingBoxIntersect(sender->bb, receiver->bb)) { return false; }
    {
        const vec4 &size = receiver->size;
        simdmat4 t(receiver->trans);
        vec4 vertices[] = {
            glm::vec4_cast(t * simdvec4( size.x, size.y, size.z, 1.0f)),
            glm::vec4_cast(t * simdvec4(-size.x, size.y, size.z, 1.0f)),
            glm::vec4_cast(t * simdvec4(-size.x,-size.y, size.z, 1.0f)),
            glm::vec4_cast(t * simdvec4( size.x,-size.y, size.z, 1.0f)),
            glm::vec4_cast(t * simdvec4( size.x, size.y,-size.z, 1.0f)),
            glm::vec4_cast(t * simdvec4(-size.x, size.y,-size.z, 1.0f)),
            glm::vec4_cast(t * simdvec4(-size.x,-size.y,-size.z, 1.0f)),
            glm::vec4_cast(t * simdvec4( size.x,-size.y,-size.z, 1.0f)),
        };
        {
            CollisionSphere sphere;
            float32 r = std::abs(vertices[0].x);
            for(int i=0; i<_countof(vertices); ++i) {
                r = absmin(r, absmin(absmin(vertices[i].x, vertices[i].y), vertices[i].z));
            }
            vec3 center = vec3(receiver->trans[3]);
            sphere.pos_r = vec4(center, r);
            sphere.bb.bl = vec4(center-r, 1.0f);
            sphere.bb.ur = vec4(center+r, 1.0f);
            if(_Collide(sender, &sphere, m)) {
                goto HIT;
            }
        }
        for(size_t i=0; i<_countof(vertices); ++i) {
            CollisionSphere sphere;
            sphere.bb.bl = sphere.bb.ur = sphere.pos_r = vec4(vec3(vertices[i]), 0.0f);
            if(_Collide(sender, &sphere, m)) {
                goto HIT;
            }
        }
    }
    {
        const vec4 &size = sender->size;
        simdmat4 t(sender->trans);
        vec4 vertices[] = {
            glm::vec4_cast(t * simdvec4( size.x, size.y, size.z, 1.0f)),
            glm::vec4_cast(t * simdvec4(-size.x, size.y, size.z, 1.0f)),
            glm::vec4_cast(t * simdvec4(-size.x,-size.y, size.z, 1.0f)),
            glm::vec4_cast(t * simdvec4( size.x,-size.y, size.z, 1.0f)),
            glm::vec4_cast(t * simdvec4( size.x, size.y,-size.z, 1.0f)),
            glm::vec4_cast(t * simdvec4(-size.x, size.y,-size.z, 1.0f)),
            glm::vec4_cast(t * simdvec4(-size.x,-size.y,-size.z, 1.0f)),
            glm::vec4_cast(t * simdvec4( size.x,-size.y,-size.z, 1.0f)),
        };
        {
            CollisionSphere sphere;
            float32 r = std::abs(vertices[0].x);
            for(int i=0; i<_countof(vertices); ++i) {
                r = absmin(r, absmin(absmin(vertices[i].x, vertices[i].y), vertices[i].z));
            }
            vec3 center = vec3(sender->trans[3]);
            sphere.pos_r = vec4(center, r);
            sphere.bb.bl = vec4(center-r, 1.0f);
            sphere.bb.ur = vec4(center+r, 1.0f);
            if(_Collide(&sphere, receiver, m)) {
                goto HIT;
            }
        }
        for(size_t i=0; i<_countof(vertices); ++i) {
            CollisionSphere sphere;
            sphere.bb.bl = sphere.bb.ur = sphere.pos_r = vec4(vec3(vertices[i]), 0.0f);
            if(_Collide(&sphere, receiver, m)) {
                goto HIT;
            }
        }
    }
    return false;

HIT:
    //vec3 dir = glm::normalize(vec3(receiver->position)-vec3(sender->position));
    //(vec3&)m.direction = glm::normalize((vec3&)m.direction+dir);
    //(vec3&)m.direction = dir;
    return true;
}


bool Collide(const CollisionEntity *sender, const CollisionEntity *receiver, CollideMessage &m)
{
    switch(sender->getShapeType()) {
    case CS_Plane:
        switch(receiver->getShapeType()) {
        case CS_Plane:  return _Collide(static_cast<const CollisionPlane*>(sender), static_cast<const CollisionPlane*>(receiver), m);
        case CS_Sphere: return _Collide(static_cast<const CollisionPlane*>(sender), static_cast<const CollisionSphere*>(receiver), m);
        case CS_Box:    return _Collide(static_cast<const CollisionPlane*>(sender), static_cast<const CollisionBox*>(receiver), m);
        }
        break;

    case CS_Sphere:
        switch(receiver->getShapeType()) {
        case CS_Plane:  return _Collide(static_cast<const CollisionSphere*>(sender), static_cast<const CollisionPlane*>(receiver), m);
        case CS_Sphere: return _Collide(static_cast<const CollisionSphere*>(sender), static_cast<const CollisionSphere*>(receiver), m);
        case CS_Box:    return _Collide(static_cast<const CollisionSphere*>(sender), static_cast<const CollisionBox*>(receiver), m);
        }
        break;

    case CS_Box:
        switch(receiver->getShapeType()) {
        case CS_Plane:  return _Collide(static_cast<const CollisionBox*>(sender), static_cast<const CollisionPlane*>(receiver), m);
        case CS_Sphere: return _Collide(static_cast<const CollisionBox*>(sender), static_cast<const CollisionSphere*>(receiver), m);
        case CS_Box:    return _Collide(static_cast<const CollisionBox*>(sender), static_cast<const CollisionBox*>(receiver), m);
        }
        break;
    }
    return false;
}





const ivec2 CollisionGrid::GRID_DIV = ivec2(32, 32);
const vec2 CollisionGrid::CELL_SIZE = vec2(PSYM_GRID_SIZE / GRID_DIV.x, PSYM_GRID_SIZE / GRID_DIV.y);

CollisionGrid::CollisionGrid()
{
}

void CollisionGrid::updateGrid( ist::vector<CollisionEntity*> &entities )
{
    for(int32 yi=0; yi<GRID_DIV.y; ++yi) {
        for(int32 xi=0; xi<GRID_DIV.x; ++xi) {
            Cell &gd = m_grid[yi][xi];
            gd.num = 0;
        }
    }

    uint32 num_entities = entities.size();
    ivec2 bl, ur;
    for(uint32 i=0; i<num_entities; ++i) {
        CollisionEntity *ce = entities[i];
        if(!ce) { continue; }
        getGridRange(ce->bb, bl, ur);
        for(int32 yi=bl.y; yi<ur.y; ++yi) {
            for(int32 xi=bl.x; xi<ur.x; ++xi) {
                Cell &gd = m_grid[yi][xi];
                if(gd.num==_countof(gd.handles)) {
                    istPrint("warning: max reached.\n");
                    break;
                }
                gd.handles[gd.num++] = ce->getCollisionHandle();
            }
        }
    }
}

ivec2 CollisionGrid::getGridCoord( const vec4 &pos )
{
    const vec2 grid_pos = vec2(-PSYM_GRID_SIZE*0.5f, -PSYM_GRID_SIZE*0.5f);
    const ivec2 grid_coord = ivec2((vec2(pos)-grid_pos)/CELL_SIZE);
    return glm::max(glm::min(grid_coord, GRID_DIV-ivec2(1,1)), ivec2(0,0));
}

void CollisionGrid::getGridRange( const BoundingBox &bb, ivec2 &out_bl, ivec2 &out_ur )
{
    out_bl = getGridCoord(bb.bl);
    out_ur = getGridCoord(bb.ur) + ivec2(1,1);
}

void CollisionGrid::getEntities( const BoundingBox &bb, ist::vector<CollisionHandle> &out_entities )
{
    out_entities.clear();

    ivec2 bl, ur;
    getGridRange(bb, bl, ur);
    for(int32 yi=bl.y; yi<ur.y; ++yi) {
        for(int32 xi=bl.x; xi<ur.x; ++xi) {
            Cell &gd = m_grid[yi][xi];
            out_entities.insert(out_entities.end(), gd.handles, gd.handles+gd.num);
        }
    }
    stl::sort(out_entities.begin(), out_entities.end());
}



uint32 CollisionModule::collideSend(CollisionEntity *sender, CollisionContext &ctx)
{
    if(!sender || (sender->getFlags() & CF_Sender)==0) { return 0; }
    CollisionGroup group = sender->getCollisionGroup();
    MessageCont &m = ctx.messages;
    HandleCont &neighbors = ctx.neighbors;

    uint32 n = 0;
    m_grid.getEntities(sender->bb, neighbors);
    unique_iterator<HandleCont::iterator> iter(neighbors.begin(), neighbors.end());
    for(; iter!=neighbors.end(); ++iter) {
        CollisionEntity *receiver = getEntity(*iter);
        if(!receiver) { continue; }
        if((receiver->getFlags() & CF_Receiver) == 0 ) { continue; }
        if(group!=0 && group==receiver->getCollisionGroup()) { continue; }
        if(receiver->getEntityHandle()==sender->getEntityHandle()) { continue; }

        CollideMessage message;
        if(Collide(sender, receiver, message)) {
            message.to = receiver->getEntityHandle();
            message.cto = receiver->getCollisionHandle();
            message.from = sender->getEntityHandle();
            message.cfrom = sender->getCollisionHandle();
            m.push_back(message);
            ++n;
        }
    }

    return n;
}

uint32 CollisionModule::collideRecv(CollisionEntity *receiver, CollisionContext &ctx)
{
    if(!receiver || (receiver->getFlags() & CF_Receiver)==0) { return 0; }
    CollisionGroup group = receiver->getCollisionGroup();
    MessageCont &m = ctx.messages;
    HandleCont &neighbors = ctx.neighbors;

    uint32 n = 0;
    m_grid.getEntities(receiver->bb, neighbors);
    unique_iterator<HandleCont::iterator> iter(neighbors.begin(), neighbors.end());
    for(; iter!=neighbors.end(); ++iter) {
        CollisionEntity *sender = getEntity(*iter);
        if(!sender) { continue; }
        if((sender->getFlags() & CF_Sender) == 0 ) { continue; }
        if(group!=0 && group==sender->getCollisionGroup()) { continue; }
        if(receiver->getEntityHandle()==sender->getEntityHandle()) { continue; }

        CollideMessage message;
        if(Collide(sender, receiver, message)) {
            message.to = receiver->getEntityHandle();
            message.cto = receiver->getCollisionHandle();
            message.from = sender->getEntityHandle();
            message.cfrom = sender->getCollisionHandle();
            m.push_back(message);
            ++n;
        }
    }

    return n;
}

CollisionModule::CollisionModule()
    : m_groupgen(0)
{
    m_entities.reserve(1024);
    m_vacant.reserve(1024);

    m_entities.push_back(nullptr); // id:0 は無効とする
}

CollisionModule::~CollisionModule()
{
    for(uint32 i=0; i<m_acons.size(); ++i) { istDelete(m_acons[i]); }
    m_acons.clear();
    for(uint32 i=0; i<m_entities.size(); ++i) { deleteEntity(m_entities[i]); }
    m_entities.clear();
    m_vacant.clear();
}

void CollisionModule::initialize()
{
}

void CollisionModule::frameBegin()
{
}

void CollisionModule::update(float32 dt)
{
    for(uint32 ti=0; ti<m_acons.size(); ++ti) {
        MessageCont &messages = m_acons[ti]->messages;
        uint32 num_messages = messages.size();
        for(uint32 mi=0; mi<num_messages; ++mi) {
            if(IEntity *e = atmGetEntity(messages[mi].to)) {
                atmCall(e, eventCollide, static_cast<const CollideMessage*>(&messages[mi]));
            }
        }
        messages.clear();
    }
}

void CollisionModule::asyncupdate(float32 dt)
{
    m_grid.updateGrid(m_entities);

    const uint32 block_size = 32;
    uint32 num_entities = m_entities.size();
    if(num_entities==0) { return; }

    uint32 num_tasks = ceildiv(num_entities, block_size);
    while(m_acons.size() < num_tasks) { m_acons.push_back(istNew(CollisionContext)()); }

    ist::parallel_for(uint32(0), num_tasks,
        [&](uint32 i) {
            uint32 first = i*block_size;
            uint32 last = std::min<uint32>((i+1)*block_size, num_entities);
            CollisionContext &ctx = *m_acons[i];
            ctx.messages.clear();
            for(uint32 i=first; i!=last; ++i) {
                collideSend(m_entities[i], ctx);
            }
        });
}

void CollisionModule::draw()
{
}

void CollisionModule::frameEnd()
{
}

void CollisionModule::copyRigitsToPSym()
{
    uint32 num = m_entities.size();
    for(uint32 i=0; i<num; ++i) {
        const CollisionEntity *ce = m_entities[i];
        if(!ce || (ce->getFlags() & CF_SPH_Sender)==0) { continue; }
        atmGetFluidModule()->addRigid(*ce);
    }
}

void CollisionModule::addEntity(CollisionEntity *e)
{
    atmDbgAssertSyncLock();
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
    e->setCollisionHandle(h);
}

CollisionEntity* CollisionModule::getEntity(CollisionHandle h)
{
    if(h >= m_entities.size()) { return nullptr; }
    return m_entities[h];
}

template<> CollisionPlane* CollisionModule::createEntity<CollisionPlane>()
{
    atmDbgAssertSyncLock();
    CollisionPlane *e = istNew(CollisionPlane)();
    addEntity(e);
    return e;
}
template<> CollisionSphere* CollisionModule::createEntity<CollisionSphere>()
{
    atmDbgAssertSyncLock();
    CollisionSphere *e = istNew(CollisionSphere)();
    addEntity(e);
    return e;
}
template<> CollisionBox* CollisionModule::createEntity<CollisionBox>()
{
    atmDbgAssertSyncLock();
    CollisionBox *e = istNew(CollisionBox)();
    addEntity(e);
    return e;
}

void CollisionModule::deleteEntity(CollisionHandle h)
{
    atmDbgAssertSyncLock();
    CollisionEntity *&ce = m_entities[h];
    if(ce) {
        ce->release();
        ce = nullptr;
        m_vacant.push_back(h);
    }
}

void CollisionModule::deleteEntity(CollisionEntity *e)
{
    atmDbgAssertSyncLock();
    if(e != nullptr) {
        deleteEntity(e->getCollisionHandle());
    }
}

CollisionGrid* CollisionModule::getCollisionGrid()
{
    return &m_grid;
}

atm::CollisionGroup CollisionModule::genGroup()
{
    return ++m_groupgen;
}



vec4 GetCollisionPosition( CollisionEntity *ce )
{
    if(ce) {
        switch(ce->getShapeType()) {
        case CS_Plane:  static_cast<CollisionPlane*>(ce); break; // 位置なし
        case CS_Sphere: return static_cast<CollisionSphere*>(ce)->pos_r; break;
        case CS_Box:    return static_cast<CollisionBox*>(ce)->position; break;
        }
    }
    return vec4();
}

} // namespace atm
