#include "stdafx.h"
#include "types.h"
#include "Graphics/ResourceManager.h"
#include "Graphics/Renderer.h"
#include "Game/AtomicApplication.h"
#include "Game/AtomicGame.h"
#include "Game/World.h"
#include "Game/SPHManager.h"
#include "Game/Message.h"
#include "GPGPU/SPH.cuh"
#include "Util.h"
#include "Enemy.h"

namespace atomic {


class Player
    : public Breakable
    , public TAttr_RotateSpeed<Attr_DoubleAxisRotation>
{
typedef Breakable super;
typedef TAttr_RotateSpeed<Attr_DoubleAxisRotation> transform;
private:
    sphRigidSphere m_rigid;
    float32 m_dash;
    int32 m_cooldown;

public:
    Player() : m_cooldown(0), m_dash(0.0f)
    {
    }

    virtual void update(float32 dt)
    {
        m_cooldown = std::max<int32>(0, m_cooldown-1);
        m_dash = std::max<float32>(0.0f, m_dash-0.04f);
        if(m_cooldown==0 && atomicGetInputs()->isButtonTriggered(1)) {
            m_dash = 2.0f;
            m_cooldown = 20;
        }

        vec4 pos = getPosition();
        vec2 move = atomicGetInputs()->getMove()*0.01f * (m_dash+1.0f);
        pos.x += move.x;
        pos.y += move.y;
        setPosition(pos);

        {
            sphForcePointGravity pg;
            pg.position = (float4&)pos;
            pg.strength = 1.0f;
            atomicGetSPHManager()->addPointGravity(pg);
        }


        CB_RID cclass = CB_CLASS_SPHERE_MEDIUM;
        super::update(dt);
        transform::update(dt);
        setTransform(computeMatrix());
        CreateRigidSphere(m_rigid, getHandle(), getPosition(), SPHGetRigidClass(cclass)->sphere_radius*getScale().x);
        atomicGetSPHManager()->addRigidSphere(cclass, getHandle(), getTransform(), m_rigid);
    }

    virtual void draw()
    {
        PointLight light;
        light.position  = getPosition() + vec4(0.0f, 0.0f, 0.3f, 0.0f);
        light.color     = vec4(0.1f, 0.2f, 1.0f, 1.0f);
        atomicGetPointLights()->addInstance(light);
    }

    bool call(uint32 call_id, const variant &v)
    {
        return super::call(call_id, v) || transform::call(call_id, v);
    }

    bool query(uint32 query_id, variant &v) const
    {
        return super::query(query_id, v) || transform::query(query_id, v);
    }
};

atomicImplementEntity(Player, ECID_PLAYER, ESID_PLAYER);

} // namespace atomic
