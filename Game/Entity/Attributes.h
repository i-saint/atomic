#ifndef atomic_Game_Entity_Attributes_h
#define atomic_Game_Entity_Attributes_h

#include "Util.h"
#include "Game/Collision.h"
#include "psym/psym.h"

struct sphFluidMessage;
typedef psym::Particle FluidMessage;

namespace atomic {

class Attr_RefCount
{
typedef Attr_RefCount this_t;
private:
    uint32 m_ref_count;

protected:
    void setRefCount(uint32 v) { m_ref_count=v; }

public:
    istIntrospectionBlock(
        istName(Attr_RefCount)
        istMember(m_ref_count)
    )
    atomicECallBlock(
        atomicMethodBlock(
        atomicECall(setRefCount)
        atomicECall(addRefCount)
        atomicECall(release)
        )
    )
    atomicEQueryBlock(
        atomicMethodBlock(
        atomicEQuery(getRefCount)
        )
    )

public:
    Attr_RefCount() : m_ref_count(1) {}
    uint32 getRefCount() const  { return m_ref_count; }
    uint32 addRefCount()        { return ++m_ref_count; }
    uint32 release()            { return --m_ref_count; }
};


class Attr_Translate
{
typedef Attr_Translate this_t;
protected:
    vec4 m_pos;

public:
    istIntrospectionBlock(
        istName(Attr_Translate)
        istMember(m_pos)
    )
    atomicECallBlock(
        atomicMethodBlock(
        atomicECall(setPosition)
        )
    )
    atomicEQueryBlock(
        atomicMethodBlock(
        atomicEQuery(getPosition)
        )
    )

public:
    Attr_Translate() {}
    const vec4& getPosition() const { return m_pos; }
    void setPosition(const vec4& v) { m_pos=v; }

    mat4 computeMatrix() const
    {
        mat4 mat;
        mat = glm::translate(mat, reinterpret_cast<const vec3&>(m_pos));
        return mat;
    }
};

class Attr_Transform
{
typedef Attr_Transform this_t;
private:
    vec4 m_pos;
    vec4 m_scale;
    vec4 m_axis;
    float32 m_rot;

public:
    istIntrospectionBlock(
        istName(Attr_Transform)
        istMember(m_pos)
        istMember(m_scale)
        istMember(m_axis)
        istMember(m_rot)
    )
    atomicECallBlock(
        atomicMethodBlock(
        atomicECall(setPosition)
        atomicECall(setScale)
        atomicECall(setAxis)
        atomicECall(setRotate)
        )
    )
    atomicEQueryBlock(
        atomicMethodBlock(
        atomicEQuery(getPosition)
        atomicEQuery(getScale)
        atomicEQuery(getAxis)
        atomicEQuery(getRotate)
        )
    )

public:
    Attr_Transform()
        : m_scale(1.0f, 1.0f, 1.0f, 0.0f)
        , m_axis(0.0f, 0.0f, 1.0f, 0.0f)
        , m_rot(0.0f)
    {}

    const vec4& getPosition() const { return m_pos; }
    const vec4& getScale() const    { return m_scale; }
    const vec4& getAxis() const     { return m_axis; }
    float32 getRotate() const       { return m_rot; }

    void setPosition(const vec4& v) { m_pos=v; }
    void setScale(const vec4& v)    { m_scale=v; }
    void setAxis(const vec4& v)     { m_axis=v; }
    void setRotate(float32 v)       { m_rot=v; }

    mat4 computeMatrix() const
    {
        mat4 mat;
        mat = glm::translate(mat, reinterpret_cast<const vec3&>(m_pos));
        mat = glm::rotate(mat, m_rot, reinterpret_cast<const vec3&>(m_axis));
        mat = glm::scale(mat, reinterpret_cast<const vec3&>(m_scale));
        return mat;
    }

    void update(float32 dt) {}
    void asyncupdate(float32 dt) {}
};

class Attr_DoubleAxisRotation
{
typedef Attr_DoubleAxisRotation this_t;
private:
    vec4 m_pos;
    vec4 m_scale;
    vec4 m_axis1;
    vec4 m_axis2;
    float32 m_rot1;
    float32 m_rot2;

public:
    istIntrospectionBlock(
        istName(Attr_DoubleAxisRotation)
        istMember(m_pos)
        istMember(m_scale)
        istMember(m_axis1)
        istMember(m_axis2)
        istMember(m_rot1)
        istMember(m_rot2)
    )
    atomicECallBlock(
        atomicMethodBlock(
        atomicECall(setPosition)
        atomicECall(setScale)
        atomicECall(setAxis1)
        atomicECall(setAxis2)
        atomicECall(setRotate1)
        atomicECall(setRotate2)
        )
    )
    atomicEQueryBlock(
        atomicMethodBlock(
        atomicEQuery(getPosition)
        atomicEQuery(getScale)
        atomicEQuery(getAxis1)
        atomicEQuery(getAxis2)
        atomicEQuery(getRotate1)
        atomicEQuery(getRotate2)
        )
    )

public:
    Attr_DoubleAxisRotation()
        : m_scale(1.0f, 1.0f, 1.0f, 0.0f)
        , m_axis1(0.0f, 1.0f, 0.0f, 0.0f)
        , m_axis2(0.0f, 0.0f, 1.0f, 0.0f)
        , m_rot1(0.0f), m_rot2(0.0f)
    {
    }

    const vec4& getPosition() const { return m_pos; }
    const vec4& getScale() const    { return m_scale; }
    const vec4& getAxis1() const    { return m_axis1; }
    const vec4& getAxis2() const    { return m_axis2; }
    float32 getRotate1() const      { return m_rot1; }
    float32 getRotate2() const      { return m_rot2; }

    void setPosition(const vec4& v) { m_pos=v; }
    void setScale(const vec4& v)    { m_scale=v; }
    void setAxis1(const vec4& v)    { m_axis1=v; }
    void setAxis2(const vec4& v)    { m_axis2=v; }
    void setRotate1(float32 v)      { m_rot1=v; }
    void setRotate2(float32 v)      { m_rot2=v; }

    mat4 computeMatrix() const
    {
        mat4 mat;
        mat = glm::translate(mat, reinterpret_cast<const vec3&>(m_pos));
        mat = glm::rotate(mat, m_rot2, reinterpret_cast<const vec3&>(m_axis2));
        mat = glm::rotate(mat, m_rot1, reinterpret_cast<const vec3&>(m_axis1));
        mat = glm::scale(mat, reinterpret_cast<const vec3&>(m_scale));
        return mat;
    }
};

template<class T>
class TAttr_RotateSpeed : public T
{
typedef TAttr_RotateSpeed this_t;
typedef T super;
private:
    float32 m_rspeed1;
    float32 m_rspeed2;

public:
    istIntrospectionBlock(
        istName(TAttr_RotateSpeed)
        istSuper(super)
        istMember(m_rspeed1)
        istMember(m_rspeed2)
    )
    atomicECallBlock(
        atomicMethodBlock(
        atomicECall(setRotateSpeed1)
        atomicECall(setRotateSpeed2)
        )
        atomicECallSuper(super)
    )
    atomicEQueryBlock(
        atomicMethodBlock(
        atomicEQuery(getRotateSpeed1)
        atomicEQuery(getRotateSpeed2)
        )
        atomicEQuerySuper(super)
    )

public:
    TAttr_RotateSpeed()
        : m_rspeed1(0.0f), m_rspeed2(0.0f)
    {}

    float32 getRotateSpeed1() const { return m_rspeed1; }
    float32 getRotateSpeed2() const { return m_rspeed2; }
    void setRotateSpeed1(float32 v) { m_rspeed1=v; }
    void setRotateSpeed2(float32 v) { m_rspeed2=v; }

    void updateRotate(float32 dt)
    {
        this->setRotate1(this->getRotate1()+getRotateSpeed1());
        this->setRotate2(this->getRotate2()+getRotateSpeed2());
    }
};

template<class T>
class TAttr_TransformMatrix : public T
{
typedef TAttr_TransformMatrix this_t;
typedef T super;
private:
    mat4 m_transform;

public:
    istIntrospectionBlock(
        istName(TAttr_TransformMatrix)
        istSuper(super)
        istMember(m_transform)
    )

    const mat4& getTransform() const    { return m_transform; }

    void setTransform(const mat4 &v) { m_transform=v; }

    void updateTransformMatrix()
    {
        setTransform(super::computeMatrix());
    }
};

template<class T>
class TAttr_TransformMatrixI : public T
{
typedef TAttr_TransformMatrixI this_t;
typedef T super;
private:
    mat4 m_transform;
    mat4 m_itransform;

public:
    istIntrospectionBlock(
        istName(TAttr_TransformMatrixI)
        istSuper(super)
        istMember(m_transform)
        istMember(m_itransform)
    )

public:
    const mat4& getTransform() const        { return m_transform; }
    const mat4& getInverseTransform() const { return m_itransform; }

    void setTransform(const mat4 &v)
    {
        m_transform = v;
        m_itransform = glm::inverse(v);
    }

    void updateTransformMatrix()
    {
        setTransform(super::computeMatrix());
    }
};




class Attr_ParticleSet
{
typedef Attr_ParticleSet this_t;
private:
    vec4 m_diffuse_color;
    vec4 m_glow_color;
    PSET_RID m_psetid;

public:
    istIntrospectionBlock(
        istName(Attr_ParticleSet)
        istMember(m_diffuse_color)
        istMember(m_glow_color)
        istMember(m_psetid)
    )
    atomicECallBlock(
        atomicMethodBlock(
        atomicECall(setDiffuseColor)
        atomicECall(setGlowColor)
        atomicECall(setModel)
        )
    )
    atomicEQueryBlock(
        atomicMethodBlock(
        atomicEQuery(getDiffuseColor)
        atomicEQuery(getGlowColor)
        atomicEQuery(getModel)
        )
    )

public:
    Attr_ParticleSet() : m_psetid(PSET_CUBE_SMALL)
    {}

    void setDiffuseColor(const vec4 &v) { m_diffuse_color=v; }
    void setGlowColor(const vec4 &v)    { m_glow_color=v; }
    void setModel(PSET_RID v)           { m_psetid=v; }
    const vec4& getDiffuseColor() const { return m_diffuse_color; }
    const vec4& getGlowColor() const    { return m_glow_color; }
    PSET_RID getModel() const           { return m_psetid; }
};


class Attr_Collision
{
typedef Attr_Collision this_t;
private:
    CollisionHandle m_collision;
    EntityHandle m_owner_handle;

public:
    istIntrospectionBlock(
        istName(Attr_Collision)
        istMember(m_collision)
        istMember(m_owner_handle)
    )
    atomicECallBlock(
        atomicMethodBlock(
        atomicECall(setCollisionFlags)
        atomicECall(setCollisionShape)
        )
    )
    atomicEQueryBlock(
        atomicMethodBlock(
        atomicEQuery(getCollisionFlags)
        atomicEQuery(getCollisionHandle)
        )
    )

public:
    Attr_Collision() : m_collision(0), m_owner_handle(0)
    {
    }

    ~Attr_Collision()
    {
        finalizeCollision();
    }

    void setCollisionFlags(int32 v)
    {
        if(CollisionEntity *ce=atomicGetCollision(m_collision)) {
            ce->setFlags(v);
        }
    }

    uint32 getCollisionFlags() const
    {
        if(CollisionEntity *ce=atomicGetCollision(m_collision)) {
            return ce->getFlags();
        }
        return 0;
    }

    void initializeCollision(EntityHandle h)
    {
        m_owner_handle = h;
    }

    void finalizeCollision()
    {
        if(m_collision!=0) {
            atomicDeleteCollision(m_collision);
            m_collision = 0;
        }
    }

    void setCollisionShape(COLLISION_SHAPE cs)
    {
        finalizeCollision();
        if(cs==CS_NULL) {
            m_collision = 0;
            return;
        }

        CollisionEntity *ce = NULL;
        switch(cs) {
        case CS_BOX:    ce = atomicCreateCollision(CollisionBox);   break;
        case CS_SPHERE: ce = atomicCreateCollision(CollisionSphere);break;
        default: istAssert(false, "unknown collision shape\n"); return;
        }
        ce->setGObjHandle(m_owner_handle);
        m_collision = ce->getCollisionHandle();
    }

    CollisionHandle getCollisionHandle() const
    {
        return m_collision;
    }

    void updateCollisionAsSphere(const mat4 &t, float32 radius)
    {
        if(CollisionEntity *ce = atomicGetCollision(m_collision)) {
            switch(ce->getShape()) {
            case CS_SPHERE:
                {
                    vec4 pos = t * vec4(0.0f, 0.0f, 0.0f, 1.0f);
                    UpdateCollisionSphere(*static_cast<CollisionSphere*>(ce), pos, radius);
                }
                break;
            }
        }
    }

    void updateCollisionByParticleSet(PSET_RID psid, const mat4 &t, float32 scale)
    {
        if(CollisionEntity *ce = atomicGetCollision(m_collision)) {
            switch(ce->getShape()) {
            case CS_SPHERE:
                {
                    vec4 pos = t * vec4(0.0f, 0.0f, 0.0f, 1.0f);
                    float radius = atomicGetRigidInfo(psid)->sphere_radius * scale;
                    UpdateCollisionSphere(*static_cast<CollisionSphere*>(ce), pos, radius);
                }
                break;
            case CS_BOX:
                {
                    vec4 box_size = (vec4&)atomicGetRigidInfo(psid)->box_size * scale;
                    UpdateCollisionBox(*static_cast<CollisionBox*>(ce), t, box_size);
                }
                break;
            }
        }
    }
};



struct CollideMessage;
struct DamageMessage;
struct DestroyMessage;
struct KillMessage;

class Attr_MessageHandler
{
typedef Attr_MessageHandler this_t;
public:
    istIntrospectionBlock(
        istName(Attr_MessageHandler)
    )
    atomicECallBlock(
        atomicMethodBlock(
        atomicECall(eventCollide)
        atomicECall(eventFluid)
        atomicECall(eventDamage)
        atomicECall(eventDestroy)
        atomicECall(eventKill)
        )
    )


    virtual void eventCollide(const CollideMessage *m)  {}
    virtual void eventFluid(const FluidMessage *m)   {}
    virtual void eventDamage(const DamageMessage *m)    {}
    virtual void eventDestroy(const DestroyMessage *m)  {}
    virtual void eventKill(const KillMessage *m)        {}
};


// 流体を浴びた時血痕を残すエフェクトを実現する
class Attr_Bloodstain
{
typedef Attr_Bloodstain this_t;
private:
    // 血痕を残す頻度。流体がこの回数衝突したとき残す。
    static const uint32 bloodstain_frequency = 128;

    stl::vector<BloodstainParticle> m_bloodstain;
    uint32 m_bloodstain_hitcount;

public:
    istIntrospectionBlock(
        istName(Attr_Bloodstain)
        istMember(m_bloodstain)
        istMember(m_bloodstain_hitcount)
    )

public:
    Attr_Bloodstain() : m_bloodstain_hitcount(0)
    {
        m_bloodstain.reserve(256);
    }

    void addBloodstain(const vec4 pos)
    {
        if(!atomicGetConfig()->show_bloodstain) { return; }

        if(++m_bloodstain_hitcount % bloodstain_frequency == 0) {
            BloodstainParticle tmp;
            tmp.position = pos;
            tmp.lifetime = 1.0f;
            m_bloodstain.push_back(tmp);
        }
    }

    void updateBloodstain()
    {
        uint32 n = m_bloodstain.size();
        for(uint32 i=0; i<n; ++i) {
            BloodstainParticle &bsp = m_bloodstain[i];
            bsp.lifetime -= 0.002f;
        }
        m_bloodstain.erase(
            stl::remove_if(m_bloodstain.begin(), m_bloodstain.end(), BloodstainParticle_IsDead()),
            m_bloodstain.end());
    }

    uint32 getNumBloodstainParticles() const { return m_bloodstain.size(); }
    const BloodstainParticle* getBloodStainParticles() const { return &m_bloodstain[0]; }
};

} // namespace atomic
#endif // atomic_Game_Entity_Attributes_h
