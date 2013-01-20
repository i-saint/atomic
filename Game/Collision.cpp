#include "stdafx.h"
#include "types.h"
#include "Task.h"
#include "Util.h"
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
        //m.direction *= vec4(-1.0f, -1.0f, -1.0f, 1.0f);
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

bool _Collide(const CollisionSphere *sender, const CollisionBox *receiver, CollideMessage &m)
{
    if(_Collide(receiver, sender, m)) {
        return true;
    }
    return false;
}

bool _Collide(const CollisionBox *sender, const CollisionBox *receiver, CollideMessage &m)
{
    return false;
}


bool Collide(const CollisionEntity *sender, const CollisionEntity *receiver, CollideMessage &m)
{
    switch(sender->getShape()) {
    case CS_PLANE:
        switch(receiver->getShape()) {
        case CS_PLANE:  return _Collide(static_cast<const CollisionPlane*>(sender), static_cast<const CollisionPlane*>(receiver), m);
        case CS_SPHERE: return _Collide(static_cast<const CollisionPlane*>(sender), static_cast<const CollisionSphere*>(receiver), m);
        case CS_BOX:    return _Collide(static_cast<const CollisionPlane*>(sender), static_cast<const CollisionBox*>(receiver), m);
        }
        break;

    case CS_SPHERE:
        switch(receiver->getShape()) {
        case CS_PLANE:  return _Collide(static_cast<const CollisionSphere*>(sender), static_cast<const CollisionPlane*>(receiver), m);
        case CS_SPHERE: return _Collide(static_cast<const CollisionSphere*>(sender), static_cast<const CollisionSphere*>(receiver), m);
        case CS_BOX:    return _Collide(static_cast<const CollisionSphere*>(sender), static_cast<const CollisionBox*>(receiver), m);
        }
        break;

    case CS_BOX:
        switch(receiver->getShape()) {
        case CS_PLANE:  return _Collide(static_cast<const CollisionBox*>(sender), static_cast<const CollisionPlane*>(receiver), m);
        case CS_SPHERE: return _Collide(static_cast<const CollisionBox*>(sender), static_cast<const CollisionSphere*>(receiver), m);
        case CS_BOX:    return _Collide(static_cast<const CollisionBox*>(sender), static_cast<const CollisionBox*>(receiver), m);
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

void CollisionGrid::updateGrid( stl::vector<CollisionEntity*> &entities )
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

void CollisionGrid::getEntities( const BoundingBox &bb, stl::vector<CollisionHandle> &out_entities )
{
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



class CollideTask : public AtomicTask
{
private:
    typedef stl::vector<CollisionHandle> HandleCont;
    typedef stl::vector<CollisionEntity*> EntityCont;
    typedef EntityCont::iterator CollisionIterator;
    typedef stl::vector<CollideMessage> MessageCont;
    CollisionSet        *m_manager;
    HandleCont          m_neighbors;
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
            m_manager->collide(*i, m_messages, m_neighbors);
        }
    }

    MessageCont& getMessages() { return m_messages; }
};

uint32 CollisionSet::collide(CollisionEntity *sender, MessageCont &m, HandleCont &neighbors)
{
    if(!sender || (sender->getFlags() & CF_SENDER)==0) { return 0; }

    uint32 n = 0;
    m_grid.getEntities(sender->bb, neighbors);
    unique_iterator<HandleCont::iterator> iter(neighbors.begin(), neighbors.end());
    for(; iter!=neighbors.end(); ++iter) {
        CollisionEntity *receiver = getEntity(*iter);
        if((receiver->getFlags() & CF_RECEIVER) == 0 ) { continue; }
        if(receiver->getGObjHandle() == sender->getGObjHandle()) { continue; }
        CollideMessage message;
        if(Collide(sender, receiver, message)) {
            message.to = receiver->getGObjHandle();
            message.cto = receiver->getCollisionHandle();
            message.from = sender->getGObjHandle();
            message.cfrom = sender->getCollisionHandle();
            m.push_back(message);
            ++n;
        }
    }
    neighbors.clear();

    return n;
}


CollisionSet::CollisionSet()
    : m_plane_allocator(sizeof(CollisionPlane), 64)
    , m_sphere_allocator(sizeof(CollisionSphere), 2048)
    , m_box_allocator(sizeof(CollisionBox), 1024)
    , m_active_tasks(0)
{
    m_tasks.reserve(32);
    m_entities.reserve(1024);
    m_vacant.reserve(1024);
#ifdef atomic_enable_distance_field
    for(uint32 i=0; i<_countof(m_df); ++i) {
        m_df[i] = istNew(DistanceField)();
    }
    m_df_current = 0;
#endif // atomic_enable_distance_field
}

CollisionSet::~CollisionSet()
{
#ifdef atomic_enable_distance_field
    for(uint32 i=0; i<_countof(m_df); ++i)      { istSafeDelete(m_df[i]); }
#endif // atomic_enable_distance_field
    for(uint32 i=0; i<m_tasks.size(); ++i)      { istSafeDelete(m_tasks[i]); }
    for(uint32 i=0; i<m_entities.size(); ++i)   { deleteEntity(m_entities[i]); }
    m_tasks.clear();
    m_entities.clear();
    m_vacant.clear();
}

void CollisionSet::frameBegin()
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

void CollisionSet::resizeTasks(uint32 n)
{
    while(m_tasks.size() < n) {
        m_tasks.push_back(istNew(CollideTask)(this));
    }
}

void CollisionSet::asyncupdate(float32 dt)
{
    m_grid.updateGrid(m_entities);

    const uint32 block_size = 256;
    uint32 num_entities = m_entities.size();
    m_active_tasks = num_entities / block_size + (num_entities%block_size==0 ? 0 : 1);
    resizeTasks(m_active_tasks);
    for(uint32 i=0; i<m_active_tasks; ++i) {
        m_tasks[i]->setup(m_entities.begin()+(block_size*i), m_entities.begin()+stl::min<uint32>(block_size*(i+1), m_entities.size()));
    }
    ist::EnqueueTasks(&m_tasks[0], m_tasks.size());
}

void CollisionSet::draw()
{
#ifdef atomic_enable_distance_field
    m_df[(m_df_current+1) % _countof(m_df)]->updateEnd();
#endif // atomic_enable_distance_field
}

void CollisionSet::frameEnd()
{
    ist::WaitTasks(&m_tasks[0], m_tasks.size());

#ifdef atomic_enable_distance_field
    m_df[(m_df_current+1) % _countof(m_df)]->updateEnd();
#endif // atomic_enable_distance_field
}

void CollisionSet::copyRigitsToPSym()
{
#ifdef atomic_enable_distance_field
    m_df[m_df_current]->updateBegin(m_entities);
    m_df_current = (m_df_current+1) % _countof(m_df);
#endif // atomic_enable_distance_field

    uint32 num = m_entities.size();
    for(uint32 i=0; i<num; ++i) {
        const CollisionEntity *ce = m_entities[i];
        if(!ce || (ce->getFlags() & CF_SPH_SENDER)==0) { continue; }
        atomicGetSPHManager()->addRigid(*ce);
    }
}

void CollisionSet::addEntity(CollisionEntity *e)
{
    atomicDbgAssertSyncLock("");
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

CollisionEntity* CollisionSet::getEntity(CollisionHandle h)
{
    //if(h==0) { return NULL; }
    if(h >= m_entities.size()) { return NULL; }
    return m_entities[h];
}

template<> CollisionPlane* CollisionSet::createEntity<CollisionPlane>()
{
    atomicDbgAssertSyncLock("");
    CollisionPlane *e = istNewA(CollisionPlane, m_plane_allocator)();
    addEntity(e);
    return e;
}
template<> CollisionSphere* CollisionSet::createEntity<CollisionSphere>()
{
    atomicDbgAssertSyncLock("");
    CollisionSphere *e = istNewA(CollisionSphere, m_sphere_allocator)();
    addEntity(e);
    return e;
}
template<> CollisionBox* CollisionSet::createEntity<CollisionBox>()
{
    atomicDbgAssertSyncLock("");
    CollisionBox *e = istNewA(CollisionBox, m_box_allocator)();
    addEntity(e);
    return e;
}

void CollisionSet::deleteEntity(CollisionHandle h)
{
    atomicDbgAssertSyncLock("");
    CollisionEntity *&ce = m_entities[h];
    if(ce) {
        switch(ce->getShape()) {
        case CS_PLANE:  istSafeDeleteA(ce, m_plane_allocator); break;
        case CS_SPHERE: istSafeDeleteA(ce, m_sphere_allocator); break;
        case CS_BOX:    istSafeDeleteA(ce, m_box_allocator); break;
        }
        m_vacant.push_back(h);
    }
}

void CollisionSet::deleteEntity(CollisionEntity *e)
{
    atomicDbgAssertSyncLock("");
    if(e != NULL) {
        deleteEntity(e->getCollisionHandle());
    }
}

CollisionGrid* CollisionSet::getCollisionGrid()
{
    return &m_grid;
}



vec4 GetCollisionPosition( CollisionEntity *ce )
{
    if(ce) {
        switch(ce->getShape()) {
        case CS_PLANE:  static_cast<CollisionPlane*>(ce); break;
        case CS_SPHERE: return static_cast<CollisionSphere*>(ce)->pos_r; break;
        case CS_BOX:    return static_cast<CollisionBox*>(ce)->position; break;
        }
    }
    return vec4();
}

} // namespace atomic
