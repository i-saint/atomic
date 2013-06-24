#ifndef atm_Game_Entity_EntityAttributes_h
#define atm_Game_Entity_EntityAttributes_h

#include "Util.h"
#include "Game/Collision.h"
#include "Game/SPHManager.h"
#include "Graphics/Renderer.h"
#include "psym/psym.h"

struct sphFluidMessage;
typedef psym::Particle FluidMessage;


namespace atm {

class Attr_Null
{
    istSerializeBlock()
public:
    atmECallBlock()
    wdmScope( void addDebugNodes(const wdmString &path) {} )
};

class Attr_RefCount
{
private:
    uint32 m_refcount;

    istSerializeBlock(
        istSerialize(m_refcount)
    )
protected:
    void setRefCount(uint32 v) { m_refcount=v; }

public:
    atmECallBlock(
        atmMethodBlock(
        atmECall(setRefCount)
        atmECall(incRefCount)
        atmECall(decRefCount)
        atmECall(getRefCount)
        )
    )

    wdmScope(
    void addDebugNodes(const wdmString &path)
    {
        wdmAddNode(path+"/m_refcount", (const uint32*)&m_refcount);
    }
    )

    Attr_RefCount() : m_refcount(0) {}
    uint32 getRefCount() const  { return m_refcount; }
    uint32 incRefCount()        { return ++m_refcount; }
    uint32 decRefCount()        { return --m_refcount; }
};


class Attr_ParticleSet
{
private:
    vec4 m_diffuse_color;
    vec4 m_glow_color;
    PSET_RID m_psetid;

    istSerializeBlock(
        istSerialize(m_diffuse_color)
        istSerialize(m_glow_color)
        istSerialize(m_psetid)
    )

public:
    atmECallBlock(
        atmMethodBlock(
            atmECall(getDiffuseColor)
            atmECall(setDiffuseColor)
            atmECall(getGlowColor)
            atmECall(setGlowColor)
            atmECall(getModel)
            atmECall(setModel)
        )
    )

    wdmScope(
    void addDebugNodes(const wdmString &path)
    {
        wdmAddNode(path+"/m_diffuse_color", &m_diffuse_color, 0.0f, 1.0f);
        wdmAddNode(path+"/m_glow_color", &m_glow_color, 0.0f, 1.0f);
    }
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

    void drawModel(const mat4 &trans)
    {
        PSetInstance inst;
        inst.diffuse = getDiffuseColor();
        inst.glow = getGlowColor();
        inst.flash = vec4();
        inst.elapsed = 0.0f;
        inst.appear_radius = 10000.0f;
        inst.translate = trans;
        atmGetSPHPass()->addPSetInstance(getModel(), inst);
    }
};


class Attr_Collision
{
private:
    CollisionHandle m_collision;
    EntityHandle m_owner_handle;

    istSerializeBlock(
        istSerialize(m_collision)
        istSerialize(m_owner_handle)
    )

public:
    atmECallBlock(
        atmMethodBlock(
            atmECall(getCollisionFlags)
            atmECall(setCollisionFlags)
            atmECall(getCollisionHandle)
            atmECall(setCollisionShape)
        )
    )
    wdmScope(void addDebugNodes(const wdmString &path) {})

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
        if(CollisionEntity *ce=atmGetCollision(m_collision)) {
            ce->setFlags(v);
        }
    }

    uint32 getCollisionFlags() const
    {
        if(CollisionEntity *ce=atmGetCollision(m_collision)) {
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
            atmDeleteCollision(m_collision);
            m_collision = 0;
        }
    }

    void setCollisionShape(CollisionShapeType cs)
    {
        finalizeCollision();
        if(cs==CS_Null) {
            m_collision = 0;
            return;
        }
        CollisionEntity *ce = NULL;
        switch(cs) {
        case CS_Box:    ce = atmCreateCollision(CollisionBox);   break;
        case CS_Sphere: ce = atmCreateCollision(CollisionSphere);break;
        default: istAssert(false); return;
        }
        ce->setEntityHandle(m_owner_handle);
        m_collision = ce->getCollisionHandle();
    }

    CollisionHandle getCollisionHandle() const { return m_collision; }
    CollisionSphere& getCollisionSphere() {
        CollisionEntity *ce = atmGetCollision(m_collision);
        istAssert(ce!=nullptr && ce->getShapeType()==CS_Sphere);
        return *static_cast<CollisionSphere*>(ce);
    }
    CollisionBox& getCollisionBox() {
        CollisionEntity *ce = atmGetCollision(m_collision);
        istAssert(ce!=nullptr && ce->getShapeType()==CS_Box);
        return *static_cast<CollisionBox*>(ce);
    }

    void updateCollision(const mat4 &t)
    {
        if(CollisionEntity *ce = atmGetCollision(m_collision)) {
            switch(ce->getShapeType()) {
            case CS_Sphere:
                {
                    CollisionSphere &shape = *static_cast<CollisionSphere*>(ce);
                    vec3 pos = vec3(t * vec4(0.0f, 0.0f, 0.0f, 1.0f));
                    UpdateCollisionSphere(shape, pos, shape.pos_r.w);
                }
                break;
            case CS_Box:
                {
                    CollisionBox &shape = *static_cast<CollisionBox*>(ce);
                    UpdateCollisionBox(shape, t, vec3(shape.size));
                }
                break;
            }
        }
    }

    void updateCollisionAsSphere(const mat4 &t, float32 radius)
    {
        if(CollisionEntity *ce = atmGetCollision(m_collision)) {
            switch(ce->getShapeType()) {
            case CS_Sphere:
                {
                    vec3 pos = vec3(t * vec4(0.0f, 0.0f, 0.0f, 1.0f));
                    UpdateCollisionSphere(*static_cast<CollisionSphere*>(ce), pos, radius);
                }
                break;
            }
        }
    }

    void updateCollisionByParticleSet(PSET_RID psid, const mat4 &t, const vec3 &scale=vec3(1.0f))
    {
        if(CollisionEntity *ce = atmGetCollision(m_collision)) {
            switch(ce->getShapeType()) {
            case CS_Sphere:
                {
                    vec3 pos = vec3(t * vec4(0.0f, 0.0f, 0.0f, 1.0f));
                    float radius = atmGetRigidInfo(psid)->sphere_radius * scale.x;
                    UpdateCollisionSphere(*static_cast<CollisionSphere*>(ce), pos, radius);
                }
                break;
            case CS_Box:
                {
                    vec3 box_size = (vec3&)atmGetRigidInfo(psid)->box_size * scale;
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

    istSerializeBlock()

public:
    atmECallBlock(
        atmMethodBlock(
            atmECall(eventCollide)
            atmECall(eventFluid)
            atmECall(eventDamage)
            atmECall(eventDestroy)
            atmECall(eventKill)
        )
    )
    wdmScope(void addDebugNodes(const wdmString &path) {})

    virtual void eventCollide(const CollideMessage *m)  {}
    virtual void eventFluid(const FluidMessage *m)      {}
    virtual void eventDamage(const DamageMessage *m)    {}
    virtual void eventDestroy(const DestroyMessage *m)  {}
    virtual void eventKill(const KillMessage *m)        {}
};


// 流体を浴びた時血痕を残すエフェクトを実現する
class Attr_Bloodstain
{
private:
    // 血痕を残す頻度。流体がこの回数衝突するたびに一つ血痕を残す。
    static const uint32 bloodstain_frequency = 256;

    ist::raw_vector<BloodstainParticle> m_bloodstain;
    uint32 m_bloodstain_hitcount;

    istSerializeBlock(
        istSerialize(m_bloodstain)
        istSerialize(m_bloodstain_hitcount)
    )

public:
    atmECallBlock(
        atmMethodBlock(
            atmECall(addBloodstain)
        )
    )
    wdmScope(void addDebugNodes(const wdmString &path) {})

public:
    Attr_Bloodstain() : m_bloodstain_hitcount(0)
    {
        m_bloodstain.reserve(256);
    }

    void addBloodstain(const mat4 &imat, const vec4& pos)
    {
        if(!atmGetConfig()->show_bloodstain) { return; }
        if(m_bloodstain.size()==m_bloodstain.capacity()) { return; }

        if(++m_bloodstain_hitcount % bloodstain_frequency == 0) {
            BloodstainParticle tmp;
            tmp.position = imat * pos;
            tmp.lifetime = 1.0f;
            m_bloodstain.push_back(tmp);
        }
    }

    void updateBloodstain(float32 dt)
    {
        uint32 n = m_bloodstain.size();
        for(uint32 i=0; i<n; ++i) {
            BloodstainParticle &bsp = m_bloodstain[i];
            bsp.lifetime -= 0.002f*dt;
        }
        m_bloodstain.erase(
            stl::remove_if(m_bloodstain.begin(), m_bloodstain.end(), BloodstainParticle_IsDead()),
            m_bloodstain.end());
    }

    uint32 getNumBloodstainParticles() const { return m_bloodstain.size(); }
    const BloodstainParticle* getBloodStainParticles() const { return m_bloodstain.empty() ? NULL : &m_bloodstain[0]; }
};

} // namespace atm
#endif // atm_Game_Entity_EntityAttributes_h
