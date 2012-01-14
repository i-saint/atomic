#include "stdafx.h"
#include "types.h"
#include "Graphics/ResourceManager.h"
#include "Game/AtomicApplication.h"
#include "Game/AtomicGame.h"
#include "Game/World.h"
#include "Game/Collision.h"
#include "Game/Entity.h"
#include "Game/EntityQuery.h"
#include "Game/Entity/Attributes.h"
#include "Routine.h"
#include "Util.h"

namespace atomic {

    RoutineCreator g_routine_creators[ROUTINE_END];

    IRoutine* CreateRoutine(ROUTINE_CLASSID rcid)
    {
        if(rcid==ROUTINE_NULL) { return NULL; }
        return g_routine_creators[rcid]();
    }


    class Routine_Shoot : public IRoutine
    {
    private:
        int32 m_frame;

    public:
        Routine_Shoot() : m_frame(0) {}

        void update(float32 dt)
        {
            ++m_frame;
            if(m_frame % 300 == 0) {
                IEntity *e = getEntity();
                vec4 pos = atomicQuery(getEntity(), getPosition, vec4);
                vec4 player_pos = GetNearestPlayerPosition(pos);
                vec2 vel = glm::normalize(vec2(player_pos)-vec2(pos)) * 0.01f;
                ShootSimpleBullet(e->getHandle(), pos, vec4(vel, 0.0f, 0.0f));
            }
        }
    };
    atomicImplementRoutine(Routine_Shoot, ROUTINE_SHOOT);


    class Routine_HomingPlayer : public IRoutine, public Attr_MessageHandler
    {
    typedef Attr_MessageHandler mhandler;
    private:
        vec4 m_vel;

    public:
        Routine_HomingPlayer() {}

        void update(float32 dt)
        {
            IEntity *e = getEntity();
            vec4 pos = atomicQuery(getEntity(), getPosition, vec4);
            vec4 player_pos = GetNearestPlayerPosition(pos);

            m_vel *= 0.98f;
            m_vel += glm::normalize(player_pos-pos) * 0.0002f;
            pos += m_vel;

            atomicCall(e, setPosition, pos);
        }

        virtual void eventCollide(const CollideMessage *m)
        {
            vec4 v = m->direction * m->direction.w * 0.1f;
            m_vel += v;
            m_vel.z = 0.0f;
            m_vel.w = 0.0f;

            float32 len = glm::length(m_vel);
            const float32 max_speed = 0.01f;
            if(len > max_speed) { m_vel = m_vel / len * max_speed; }
        }

        virtual bool call(uint32 call_id, const variant &v)
        {
            return mhandler::call(call_id, v);
        }
    };
    atomicImplementRoutine(Routine_HomingPlayer, ROUTINE_HOMING_PLAYER);


    class Routine_Pinball : public IRoutine, public Attr_MessageHandler
    {
        typedef Attr_MessageHandler mhandler;
    private:
        vec4 m_vel;

    public:
        Routine_Pinball() {}

        void setVelocity(const vec4 &v) { m_vel=v; }

        void update(float32 dt)
        {
            IEntity *e = getEntity();
            vec4 pos = atomicQuery(getEntity(), getPosition, vec4);
            pos += m_vel;
            atomicCall(e, setPosition, pos);
        }

        virtual void eventCollide(const CollideMessage *m)
        {
            vec4 base_dir = m->direction;
            vec4 dir = base_dir;
            dir.z = dir.w = 0.0f;
            dir = glm::normalize(dir);

            if(glm::dot(dir, m_vel) < 0.0f) {
                m_vel = glm::reflect(m_vel, dir);
            }
        }

        virtual bool call(uint32 call_id, const variant &v)
        {
            switch(call_id) {
            DEFINE_ECALL1(setVelocity, vec4);
            default: return mhandler::call(call_id, v);
            }
        }
    };
    atomicImplementRoutine(Routine_Pinball, ROUTINE_PINBALL);

} // namespace atomic
