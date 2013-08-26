#include "entityPCH.h"
#include "Entity/EntityCommon.h"
#include "Entity/Level.h"

namespace atm {

class Boss1 : public EntityWithPosition
{
typedef EntityWithPosition super;
private:
    enum State {
        St_Deploy,
        St_Form1,
        St_Form1ToForm2,
        St_Form2,
        St_Destroy,
        St_Dead,
    };
    EntityHandle m_gears[4];
    EntityHandle m_turrents[4];
    EntityHandle m_walls[4];
    EntityHandle m_barrier;
    EntityHandle m_core;
    State m_state;
    int32 m_state_frame;
    int32 m_total_frame;

    istSerializeBlock(
        istSerializeBase(super)
        istSerialize(m_gears)
        istSerialize(m_turrents)
        istSerialize(m_walls)
        istSerialize(m_barrier)
        istSerialize(m_core)
        istSerialize(m_state)
        istSerialize(m_state_frame)
        istSerialize(m_total_frame)
    )
    atmECallBlock(
        atmECallSuper(super)
    )

public:
    Boss1()
        : m_state(St_Deploy), m_state_frame(), m_total_frame()
    {
        clear(m_gears);
        clear(m_turrents);
        clear(m_walls);
        clear(m_barrier);
        clear(m_core);
    }

    ~Boss1()
    {
    }

    void initialize() override
    {
        {
            vec3 upvector[] = {
                vec3(-1.0f, 0.0f, 0.0f),
                vec3( 1.0f, 0.0f, 0.0f),
                vec3(-1.0f, 0.0f, 0.0f),
                vec3( 1.0f, 0.0f, 0.0f),
            };
            vec3 gearpos[] = {
                vec3( 0.5f, 0.2f, 0.0f),
                vec3(-0.5f, 0.2f, 0.0f),
                vec3( 0.9f, 0.2f, 0.0f),
                vec3(-0.9f, 0.2f, 0.0f),
            };
            vec3 wallpos[] = {
                vec3( 0.5f, 0.2f, 0.0f),
                vec3(-0.5f, 0.2f, 0.0f),
                vec3( 0.9f, 0.2f, 0.0f),
                vec3(-0.9f, 0.2f, 0.0f),
            };
            for(size_t i=0; i<_countof(m_gears); ++i) {
                IEntity *gear = atmCreateEntityT(GearSmall);
                atmCall(gear, setPosition, gearpos[i]);
                atmCall(gear, setParent, getHandle());

                IEntity *wall = atmCreateEntityT(GroundBlock);
                atmCall(wall, setPosition, wallpos[i]);
                atmCall(wall, setParent, getHandle());

                m_gears[i] = gear->getHandle();
                m_walls[i] = wall->getHandle();
            }
        }
    }

    void update(float32 dt) override
    {
        super::update(dt);
        SweepDeadEntities(m_gears);
        SweepDeadEntities(m_turrents);
        SweepDeadEntities(m_walls);
        SweepDeadEntities(m_barrier);
        SweepDeadEntities(m_core);

        switch(m_state) {
        case St_Deploy:         updateDeploy(dt);       break;
        case St_Form1:          updateForm1(dt);        break;
        case St_Form1ToForm2:   updateForm1ToForm2(dt); break;
        case St_Form2:          updateForm2(dt);        break;
        case St_Destroy:        updateDestroy(dt);      break;
        }
        updateLink();
    }

    void updateLink()
    {
        for(size_t i=0; i<_countof(m_gears); ++i) {
            IEntity *gear = atmGetEntity(m_gears[i]);
            IEntity *wall = atmGetEntity(m_walls[i]);
            if(gear && wall) {
                float32 rot; atmQuery(gear, getSpinAngle, rot);
                vec3 dir = glm::rotateZ(vec3(1.0f, 0.0f, 0.0f), rot);
                atmCall(wall, setDirection, dir*0.2f);
            }
        }
    }

    void updateDeploy(float32 dt)
    {
    }

    void updateForm1(float32 dt)
    {
    }

    void updateForm1ToForm2(float32 dt)
    {
    }

    void updateForm2(float32 dt)
    {
    }

    void updateDestroy(float32 dt)
    {
    }
};
atmImplementEntity(Boss1);
atmExportClass(Boss1);

} // namespace atm
