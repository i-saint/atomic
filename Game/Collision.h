#ifndef __atomic_Game_Collision__
#define __atomic_Game_Collision__

#include "GPGPU/SPH.cuh"

namespace atomic {

enum COLLISION_SHAPE {
    CS_UNKNOWN,
    CS_PLANE,
    CS_SPHERE,
    CS_BOX,
    CS_BEAM,
    CS_END,
};

struct BoundingBox
{
    vec4 ur;
    vec4 bl;
};

class CollisionSet;
typedef uint32 CollisionHandle;


// virtual なデストラクタがないのは意図的。
// これを継承するオブジェクトはデータ保持以外のことはやってはいけない。
struct CollisionEntity
{
friend class CollisionSet;
private:
    union {
        struct {
            COLLISION_SHAPE m_shape;
            CollisionHandle m_col_handle;
            EntityHandle m_gobj_handle;
        };
        float4 padding;
    };

    void SetCollisionHandle(CollisionHandle v) { m_col_handle=v; }

protected:
    void setShape(COLLISION_SHAPE v) { m_shape=v; }

public:
    CollisionEntity() : m_shape(CS_UNKNOWN), m_col_handle(0), m_gobj_handle(0) {}
    COLLISION_SHAPE getShape() const { return m_shape; }
    CollisionHandle getCollisionHandle() const { return m_col_handle; }
    EntityHandle    getGameHandle() const { return m_gobj_handle; }

    void setGObjHandle(EntityHandle v) { m_gobj_handle=v; }
};

struct CollisionPlane : public CollisionEntity
{
public:
    vec4 plane;

    CollisionPlane() { setShape(CS_PLANE); }

};

struct CollisionSphere : public CollisionEntity
{
public:
    BoundingBox bb;
    vec4 pos_r; // w=radius

    CollisionSphere() { setShape(CS_SPHERE); }
};

struct CollisionBox : public CollisionEntity
{
public:
    BoundingBox bb;
    vec4 position;
    vec4 planes[6];

    CollisionBox() { setShape(CS_BOX); }
};


struct CollisionMessage
{
    union {
        struct {
            int from; // SPH_RIGID_SHAPE
            int to;
        };
        float4 padding;
    };
    vec4 direction; // w=めり込み量
};


class CollisionTask;

// 現状総当り方式…
class CollisionSet : boost::noncopyable
{
friend class CollisionTask;
private:
    stl::vector<CollisionTask*>     m_tasks;
    stl::vector<CollisionEntity*>   m_entities;
    thrust::host_vector<sphRigidSphere> m_sph_spheres;
    thrust::host_vector<sphRigidBox>    m_sph_boxes;

    void addEntity(CollisionEntity *e);

public:
    CollisionSet();
    ~CollisionSet();

    void updateBegin(float32 dt);
    void update(float32 dt);
    void updateEnd();
    void asyncupdate(float32 dt);

    template<class T> T* createEntity();
    void deleteEntity(CollisionHandle h);
};



} // namespace atomic
#endif //__atomic_Game_Collision__
