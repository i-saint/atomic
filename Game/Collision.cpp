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
#include "GPGPU/SPH.cuh"

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




const ivec3 DistanceField::grid_div  = ivec3(SPH_DISTANCE_FIELD_DIV_X, SPH_DISTANCE_FIELD_DIV_Y, SPH_DISTANCE_FIELD_DIV_Z);
const ivec3 DistanceField::block_size= ivec3(16, 16, 8);
const ivec3 DistanceField::block_num = grid_div/block_size;
const vec3 DistanceField::grid_size  = vec3(SPH_GRID_SIZE, SPH_GRID_SIZE, 0.32f);
const vec3 DistanceField::grid_pos   = vec3(-SPH_GRID_SIZE*0.5f, -SPH_GRID_SIZE*0.5f, 0.0f);
const vec3 DistanceField::cell_size  = grid_size/vec3(grid_div);

ivec3 DistanceField::getDistanceFieldCoord( const vec3 &pos )
{
    ivec3 fc = ivec3((pos-grid_pos) / cell_size);
    return glm::min(grid_div-ivec3(1,1,1), glm::max(ivec3(0,0,0), fc));
}

class DistanceTask : public AtomicTask
{
private:
    typedef stl::vector<CollisionEntity*>   EntityCont;

    ivec3               m_bl, m_ur;
    DistanceField       *m_df;
    EntityCont          *m_entities;

public:
    DistanceTask(DistanceField *df) : m_df(df) { setPriority(90); }
    void setup(const ivec3 &bl, const ivec3 &ur, EntityCont &ec) { m_bl=bl; m_ur=ur; m_entities=&ec; }

    void getOverlaped(const BoundingBox &bb, ivec3 &out_ubl, ivec3 &out_uur)
    {
        ivec3 ubl = DistanceField::getDistanceFieldCoord(vec3(bb.bl));
        ivec3 uur = DistanceField::getDistanceFieldCoord(vec3(bb.ur)) + ivec3(1,1,1);
        out_ubl = uvec3(std::max<int32>(ubl.x, out_ubl.x), std::max<int32>(ubl.y, out_ubl.y), std::max<int32>(ubl.z, out_ubl.z));
        out_uur = uvec3(std::min<int32>(uur.x, out_uur.x), std::min<int32>(uur.y, out_uur.y), std::min<int32>(uur.z, out_uur.z));
    }

    void doCollide(const CollisionEntity *ce, const ivec3 &ubl, const ivec3 &uur)
    {
        vec4 *dist = m_df->getDistances();
        EntityHandle *entities = m_df->getEntities();

        for(int32 zi=ubl.z; zi<uur.z; ++zi) {
            for(int32 yi=ubl.y; yi<uur.y; ++yi) {
                for(int32 xi=ubl.x; xi<uur.x; ++xi) {
                    int32 gi = DistanceField::grid_div.y*DistanceField::grid_div.x*zi + DistanceField::grid_div.x*yi + xi;
                    vec3 center = DistanceField::grid_pos + DistanceField::cell_size*(vec3(xi, yi, zi)+vec3(0.5f));
                    CollisionSphere s;
                    s.pos_r = vec4(center, DistanceField::cell_size.x*0.5f);
                    s.updateBoundingBox();
                    CollideMessage m;
                    if(Collide(ce, &s, m)) {
                        dist[gi] = vec4(vec3(m.direction), -m.direction.w);
                        entities[gi] = ce->getGObjHandle();
                    }
                }
            }
        }
    }

    void clearBlock()
    {
        vec4 *dist = m_df->getDistances();
        EntityHandle *entities = m_df->getEntities();

        for(int32 zi=m_bl.z; zi<m_ur.z; ++zi) {
            for(int32 yi=m_bl.y; yi<m_ur.y; ++yi) {
                for(int32 xi=m_bl.x; xi<m_ur.x; ++xi) {
                    int32 gi = DistanceField::grid_div.y*DistanceField::grid_div.x*zi + DistanceField::grid_div.x*yi + xi;
                    dist[gi] = vec4(0.0f, 0.0f, 0.0f, 1.0f);
                    entities[gi] = 0;
                }
            }
        }
    }

    void exec()
    {
        clearBlock();
        for(uint32 ci=0; ci<m_entities->size(); ++ci) {
            const CollisionEntity *ce = (*m_entities)[ci];
            if(!ce) { continue; }

            ivec3 ubl = m_bl;
            ivec3 uur = m_ur;
            switch(ce->getShape()) {
            case CS_BOX:
                getOverlaped(static_cast<const CollisionBox*>(ce)->bb, ubl, uur);
                doCollide(ce, ubl, uur);
                break;

            case CS_SPHERE:
                getOverlaped(static_cast<const CollisionSphere*>(ce)->bb, ubl, uur);
                doCollide(ce, ubl, uur);
                break;
            }
            
        }
    }
};


DistanceField::DistanceField()
{
    std::fill_n(m_handle, _countof(m_handle), 0);
    for(int32 i=0; i<block_num.x*block_num.y; ++i) {
        m_tasks.push_back( istNew(DistanceTask)(this) );
    }
}

DistanceField::~DistanceField()
{
    for(uint32 i=0; i<m_tasks.size(); ++i) { istDelete(m_tasks[i]); }
}

void DistanceField::updateBegin( EntityCont &v )
{
    for(int32 yi=0; yi<block_num.y; ++yi) {
        for(int32 xi=0; xi<block_num.x; ++xi) {
            const ivec3 bl = ivec3(block_size.x*xi, block_size.y*yi, 0);
            const ivec3 ur = ivec3(block_size.x*(xi+1), block_size.y*(yi+1), SPH_DISTANCE_FIELD_DIV_Z);
            DistanceTask *dt = m_tasks[block_num.x*yi + xi];
            dt->setup(bl, ur, v);
        }
    }
    TaskScheduler::addTask((Task**)&m_tasks[0], m_tasks.size());
}

void DistanceField::updateEnd()
{
    for(uint32 i=0; i<m_tasks.size(); ++i) { m_tasks[i]->join(); }
}




const ivec2 CollisionGrid::GRID_DIV = ivec2(32, 32);
const vec2 CollisionGrid::CELL_SIZE = vec2(SPH_GRID_SIZE / GRID_DIV.x, SPH_GRID_SIZE / GRID_DIV.y);

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
    const vec2 grid_pos = vec2(-SPH_GRID_SIZE*0.5f, -SPH_GRID_SIZE*0.5f);
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
    : m_active_tasks(0)
{
    m_tasks.reserve(32);
    m_entities.reserve(1024);
    m_vacant.reserve(1024);
#ifdef __atomic_enable_distance_field__
    for(uint32 i=0; i<_countof(m_df); ++i) {
        m_df[i] = istNew(DistanceField)();
    }
    m_df_current = 0;
#endif // __atomic_enable_distance_field__
}

CollisionSet::~CollisionSet()
{
#ifdef __atomic_enable_distance_field__
    for(uint32 i=0; i<_countof(m_df); ++i)      { istSafeDelete(m_df[i]); }
#endif // __atomic_enable_distance_field__
    for(uint32 i=0; i<m_tasks.size(); ++i)      { istSafeDelete(m_tasks[i]); }
    for(uint32 i=0; i<m_entities.size(); ++i)   { istSafeDelete(m_entities[i]); }
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
        m_tasks[i]->setup(m_entities.begin()+(block_size*i), m_entities.begin()+std::min<uint32>(block_size*(i+1), m_entities.size()));
    }
    TaskScheduler::addTask((Task**)&m_tasks[0], m_tasks.size());
}

void CollisionSet::draw()
{
#ifdef __atomic_enable_distance_field__
    m_df[(m_df_current+1) % _countof(m_df)]->updateEnd();
#endif // __atomic_enable_distance_field__
}

void CollisionSet::frameEnd()
{
    TaskScheduler::waitFor((Task**)&m_tasks[0], m_tasks.size());

#ifdef __atomic_enable_distance_field__
    m_df[(m_df_current+1) % _countof(m_df)]->updateEnd();
#endif // __atomic_enable_distance_field__
}

void CollisionSet::copyRigitsToGPU()
{
#ifdef __atomic_enable_distance_field__
    m_df[m_df_current]->updateBegin(m_entities);
    m_df_current = (m_df_current+1) % _countof(m_df);
#endif // __atomic_enable_distance_field__

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
    //if(h==0) { return NULL; }
    return m_entities[h];
}

template<> CollisionPlane* CollisionSet::createEntity<CollisionPlane>()
{
    CollisionPlane *e = istNew(CollisionPlane)();
    addEntity(e);
    return e;
}

template<> CollisionSphere* CollisionSet::createEntity<CollisionSphere>()
{
    CollisionSphere *e = istNew(CollisionSphere)();
    addEntity(e);
    return e;
}

template<> CollisionBox* CollisionSet::createEntity<CollisionBox>()
{
    CollisionBox *e = istNew(CollisionBox)();
    addEntity(e);
    return e;
}

void CollisionSet::deleteEntity(CollisionHandle h)
{
    istSafeDelete(m_entities[h]);
    m_vacant.push_back(h);
}

void CollisionSet::deleteEntity(CollisionEntity *e)
{
    deleteEntity(e->getCollisionHandle());
}

CollisionGrid* CollisionSet::getCollisionGrid()
{
    return &m_grid;
}

DistanceField* CollisionSet::getDistanceField()
{
#ifdef __atomic_enable_distance_field__
    return m_df[m_df_current];
#endif // __atomic_enable_distance_field__
    return NULL;
}

} // namespace atomic
