#ifndef atm_Game_Collision_h
#define atm_Game_Collision_h

namespace atm {

enum CollisionShapeType {
    CS_Null,
    CS_Plane,
    CS_Sphere,
    CS_Box,
    CS_End,
};
enum CollisionFlag {
    CF_Receiver     = 1 << 0,
    CF_Sender       = 1 << 1,
    CF_SPH_Sender   = 1 << 2,
    CF_SPH_Receiver = 1 << 3,
};

struct istAlign(16) BoundingBox
{
    vec4 bl;
    vec4 ur;

    void adjust() {
        if(ur.x < bl.x) { stl::swap(ur.x, bl.x); }
        if(ur.y < bl.y) { stl::swap(ur.y, bl.y); }
        if(ur.z < bl.z) { stl::swap(ur.z, bl.z); }
    }
    float32 getXLength() const { return ur.x-bl.x; }
    float32 getYLength() const { return ur.y-bl.y; }
    float32 getZLength() const { return ur.z-bl.z; }
    vec4 getUUU() const { return vec4(ur.x, ur.y, ur.z, 0.0f); }
    vec4 getBUU() const { return vec4(bl.x, ur.y, ur.z, 0.0f); }
    vec4 getUBU() const { return vec4(ur.x, bl.y, ur.z, 0.0f); }
    vec4 getUUB() const { return vec4(ur.x, ur.y, bl.z, 0.0f); }
    vec4 getBBU() const { return vec4(bl.x, bl.y, ur.z, 0.0f); }
    vec4 getBUB() const { return vec4(bl.x, ur.y, bl.z, 0.0f); }
    vec4 getUBB() const { return vec4(ur.x, bl.y, bl.z, 0.0f); }
    vec4 getBBB() const { return vec4(bl.x, bl.y, bl.z, 0.0f); }
};
atmSerializeRaw(atm::BoundingBox);

class CollisionSet;


struct CollisionEntity
{
friend class CollisionSet;
private:
    CollisionShapeType m_shape_type;
    CollisionHandle m_col_handle;
    EntityHandle    m_entity_handle;
    int32           m_flags; // COLLISION_FLAG
public:
    BoundingBox bb;

private:
    istSerializeBlock(
        istSerialize(m_shape_type)
        istSerialize(m_col_handle)
        istSerialize(m_entity_handle)
        istSerialize(m_flags)
        istSerialize(bb)
    )

    void setCollisionHandle(CollisionHandle v) { m_col_handle=v; }

protected:
    void setShape(CollisionShapeType v) { m_shape_type=v; }

public:
    CollisionEntity() : m_shape_type(CS_Null), m_col_handle(0), m_entity_handle(0), m_flags(CF_Receiver|CF_Sender|CF_SPH_Sender) {}
    virtual ~CollisionEntity() {}
    void release() { istDelete(this); }
    CollisionShapeType getShapeType() const           { return m_shape_type; }
    CollisionHandle getCollisionHandle() const  { return m_col_handle; }
    EntityHandle    getEntityHandle() const     { return m_entity_handle; }
    int32           getFlags() const            { return m_flags; }

    void setEntityHandle(EntityHandle v)    { m_entity_handle=v; }
    void setFlags(int32 v)                  { m_flags=v; }
};

struct LessCollisionHandle { bool operator()(CollisionEntity *a, CollisionEntity *b) { return a->getCollisionHandle() < b->getCollisionHandle(); }};


struct CollisionPlane : public CollisionEntity
{
typedef CollisionEntity super;
istDefinePoolNewST(CollisionPlane);
public:
    vec4 plane;

    istSerializeBlock(
        istSerializeBase(super)
        istSerialize(plane)
        )

public:
    CollisionPlane() { setShape(CS_Plane); }
};

struct CollisionSphere : public CollisionEntity
{
typedef CollisionEntity super;
istDefinePoolNewST(CollisionSphere);
public:
    vec4 pos_r; // w=radius

    istSerializeBlock(
        istSerializeBase(super)
        istSerialize(pos_r)
        )

public:
    CollisionSphere() { setShape(CS_Sphere); }
    void updateBoundingBox()
    {
        bb.ur = vec4(vec3(pos_r)+vec3(pos_r.w), 1.0f);
        bb.bl = vec4(vec3(pos_r)-vec3(pos_r.w), 1.0f);
    }
};

struct CollisionBox : public CollisionEntity
{
typedef CollisionEntity super;
istDefinePoolNewST(CollisionBox);
public:
    vec4 position;
    vec4 size;
    vec4 planes[6];

    istSerializeBlock(
        istSerializeBase(super)
        istSerialize(position)
        istSerialize(size)
        istSerialize(planes)
        )

public:
    CollisionBox() { setShape(CS_Box); }
};


struct CollideMessage
{
    union {
        struct {
            EntityHandle from;
            EntityHandle to;
            CollisionHandle cfrom;
            CollisionHandle cto;
        };
        float padding[4];
    };
    vec4 direction; // w=めり込み量

    CollideMessage() : from(0), to(0), cfrom(0), cto(0) {}
};


class CollideTask;
class DistanceTask;


class CollisionGrid
{
public:
    static const int32 MAX_ENTITIES_IN_CELL = 64;
    static const ivec2 GRID_DIV;
    static const uint32 GRID_XDIV = 32;
    static const uint32 GRID_YDIV = 32;
    static const vec2 CELL_SIZE;
    struct Cell
    {
        CollisionHandle handles[MAX_ENTITIES_IN_CELL-1];
        uint32 num;
        Cell()
        {
            num = 0;
            stl::fill_n(handles, _countof(handles), 0);
        }
    };

private:
    Cell m_grid[GRID_YDIV][GRID_XDIV];

public:
    CollisionGrid();
    void updateGrid(ist::vector<CollisionEntity*> &entities);
    ivec2 getGridCoord(const vec4 &pos);
    void getGridRange(const BoundingBox &bb, ivec2 &out_bl, ivec2 &out_ur);

    // BoundingBox の範囲のセルの要素を取得。
    // 注意: out_handles の結果はソートされているが、同じ要素が複数入っている可能性がある。
    //       atm::unique_iterator で重複要素を回避しながら巡回すること。(stl::unique() が非常に遅いため、こうなっている)
    void getEntities(const BoundingBox &bb, ist::vector<CollisionHandle> &out_handles);
};


class CollisionSet : public IAtomicGameModule
{
typedef IAtomicGameModule super;
public:
    typedef ist::vector<CollisionHandle>    HandleCont;
    typedef ist::vector<CollisionEntity*>   EntityCont;
    typedef ist::vector<CollideMessage>     MessageCont;

    struct AsyncContext
    {
        HandleCont  neighbors;
        MessageCont messages;
    };
    typedef ist::vector<AsyncContext*> AsyncCtxCont;

public:
    CollisionSet();
    ~CollisionSet();

    void initialize();
    void frameBegin();
    void update(float32 dt);
    void asyncupdate(float32 dt);
    void draw();
    void frameEnd();

    void copyRigitsToPSym();

    CollisionEntity* getEntity(CollisionHandle h);
    template<class T> T* createEntity();
    void deleteEntity(CollisionHandle e);
    void deleteEntity(CollisionEntity *e);

    CollisionGrid* getCollisionGrid();

private:
    void addEntity(CollisionEntity *e);
    uint32 collide(CollisionEntity *e, MessageCont &m, HandleCont &neighbors_placeholder);

    EntityCont      m_entities;
    HandleCont      m_vacant;

    // 以下 serialize 不要
    CollisionGrid   m_grid;
    AsyncCtxCont    m_acons;

    istSerializeBlock(
        istSerializeBase(super)
        istSerialize(m_entities)
        istSerialize(m_vacant)
        )
};


vec4 GetCollisionPosition(CollisionEntity *ce);


} // namespace atm
#endif //atm_Game_Collision_h
