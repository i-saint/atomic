#include "stdafx.h"
#include "types.h"
#include "Game/Entity.h"
#include "Game/EntityQuery.h"
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

    class Routine_HomingPlayer : public IRoutine
    {
    private:
        vec4 m_vel;

    public:
        Routine_HomingPlayer() {}

        void asyncupdate(float32 dt)
        {
            IEntity *e = getEntity();
            vec4 pos = atomicQuery(getEntity(), getPosition, vec4);
            vec4 player_pos = GetNearestPlayerPosition(pos);

            m_vel *= 0.98f;
            m_vel += glm::normalize(player_pos-pos) * 0.0001f;
            pos += m_vel;

            atomicCall(e, setPosition, pos);
        }
    };
    atomicImplementRoutine(Routine_HomingPlayer, ROUTINE_HOMING_PLAYER);


} // namespace atomic
