#include "stdafx.h"
#include "types.h"
#include "Sound/AtomicSound.h"
#include "Graphics/ResourceManager.h"
#include "Graphics/Renderer.h"
#include "Game/AtomicApplication.h"
#include "Game/AtomicGame.h"
#include "Game/World.h"
#include "Game/SPHManager.h"
#include "Game/Collision.h"
#include "Game/Message.h"
#include "Util.h"
#include "Enemy.h"
#include "Routine.h"

namespace atomic {


class Level_Test : public IEntity
{
typedef IEntity super;
private:
    enum STATE {
        ST_BEGIN,
        ST_NORMAL,
        ST_GAME_OVER,
    };

    int m_frame;
    CollisionHandle m_planes[4];

    EntityHandle m_player;
    stl::vector<EntityHandle> m_small_enemies;
    stl::vector<EntityHandle> m_medium_enemies;
    stl::vector<EntityHandle> m_large_enemies;

    int32 m_level;
    int32 m_loop;
    STATE m_state;

public:
    Level_Test() : m_frame(0), m_player(0), m_level(0), m_loop(0), m_state(ST_BEGIN)
    {
        stl::fill_n(m_planes, _countof(m_planes), 0);
    }

    void initialize()
    {
        super::initialize();

        const vec4 field_size = vec4(PSYM_GRID_SIZE*0.5f);
        atomicGetWorld()->setFieldSize(field_size);

        vec4 planes[] = {
            vec4(-1.0f, 0.0f, 0.0f, field_size.x),
            vec4( 1.0f, 0.0f, 0.0f, field_size.x),
            vec4( 0.0f,-1.0f, 0.0f, field_size.y),
            vec4( 0.0f, 1.0f, 0.0f, field_size.y),
        };
        BoundingBox bboxes[] = {
            {vec4( field_size.x,-field_size.y, 0.0f, 1.0f), vec4( field_size.x, field_size.y, 0.0f, 1.0f)},
            {vec4(-field_size.x,-field_size.y, 0.0f, 1.0f), vec4(-field_size.x, field_size.y, 0.0f, 1.0f)},
            {vec4(-field_size.x, field_size.y, 0.0f, 1.0f), vec4( field_size.x, field_size.y, 0.0f, 1.0f)},
            {vec4(-field_size.x,-field_size.y, 0.0f, 1.0f), vec4( field_size.x,-field_size.y, 0.0f, 1.0f)},
        };
        for(uint32 i=0; i<_countof(planes); ++i) {
            CollisionPlane *p = atomicCreateCollision(CollisionPlane);
            p->setGObjHandle(getHandle());
            p->setFlags(CF_SENDER|CF_SPH_SENDER);
            p->bb = bboxes[i];
            p->plane = planes[i];
            m_planes[i] = p->getCollisionHandle();
        }
    }

    void finalize()
    {
        for(uint32 i=0; i<_countof(m_planes); ++i) {
            atomicDeleteCollision(m_planes[i]);
        }
        super::finalize();
    }

    float32 getLoopBoost() const { return 1.0f+(0.2f * m_loop); }

    IEntity* putSmallEnemy()
    {
        IEntity *e = NULL;
        e = atomicCreateEntity(Enemy_Test);
        atomicCall(e, setCollisionShape, CS_SPHERE);
        atomicCall(e, setModel, PSET_SPHERE_SMALL);
        atomicCall(e, setPosition, GenRandomVector2() * 2.2f);
        atomicCall(e, setHealth, 15.0f * getLoopBoost());
        atomicCall(e, setAxis1, GenRandomUnitVector3());
        atomicCall(e, setAxis2, GenRandomUnitVector3());
        atomicCall(e, setRotateSpeed1, 2.4f);
        atomicCall(e, setRotateSpeed2, 2.4f);
        atomicCall(e, setRoutine, ROUTINE_HOMING_PLAYER);
        atomicCall(e, setLightRadius, 0.5f);
        m_small_enemies.push_back(e->getHandle());
        return e;
    }

    IEntity* putPinballEnemy()
    {
        IEntity *e = NULL;
        e = atomicCreateEntity(Enemy_Test);
        atomicCall(e, setCollisionShape, CS_SPHERE);
        atomicCall(e, setCollisionFlags, CF_RECEIVER|CF_SENDER);
        atomicCall(e, setModel, PSET_SPHERE_BULLET);
        atomicCall(e, setPosition, GenRandomVector2() * 0.5f + vec4(1.5f, 1.5f, 0.0f, 0.0f));
        atomicCall(e, setHealth, 5.0f * getLoopBoost());
        atomicCall(e, setAxis1, GenRandomUnitVector3());
        atomicCall(e, setAxis2, GenRandomUnitVector3());
        atomicCall(e, setRotateSpeed1, 2.4f);
        atomicCall(e, setRotateSpeed2, 2.4f);
        atomicCall(e, setRoutine, ROUTINE_PINBALL);
        atomicCall(e, setLightRadius, 0.0f);
        atomicCall(e, setVelocity, (vec4(0.0f, -0.99f, 0.0f, 0.0f))*0.005f);
        //atomicCall(e, setAccel, GenRandomUnitVector2()*0.00005f);
        atomicCall(e, setAccel, vec4(0.0f, -1.0f, 0.0f, 0.0f)*0.00005f);
        m_small_enemies.push_back(e->getHandle());
        return e;
    }

    IEntity* putMediumEnemy()
    {
        IEntity *e = atomicCreateEntity(Enemy_Test);
        switch(atomicGetRandom()->genInt32() % 2) {
        case 0: atomicCall(e, setModel, PSET_CUBE_MEDIUM);  atomicCall(e, setCollisionShape, CS_BOX); break;
        case 1: atomicCall(e, setModel, PSET_SPHERE_MEDIUM);atomicCall(e, setCollisionShape, CS_SPHERE); break;
        }
        atomicCall(e, setPosition, GenRandomVector2() * 2.1f);
        atomicCall(e, setHealth, 100.0f * getLoopBoost());
        atomicCall(e, setAxis1, GenRandomUnitVector3());
        atomicCall(e, setAxis2, GenRandomUnitVector3());
        atomicCall(e, setRotateSpeed1, 0.4f);
        atomicCall(e, setRotateSpeed2, 0.4f);
        atomicCall(e, setRoutine, ROUTINE_SHOOT);
        atomicCall(e, setLightRadius, 0.8f);
        atomicCall(e, setExplosionSE, SE_EXPLOSION4);
        atomicCall(e, setExplosionChannel, SE_CHANNEL4);
        m_medium_enemies.push_back(e->getHandle());
        return e;
    }

    IEntity* putLargeEnemy()
    {
        IEntity *e = atomicCreateEntity(Enemy_Test);
        //atomicCall(e, setModel, PSET_CUBE_LARGE); atomicCall(e, setCollisionShape, CS_BOX);
        switch(atomicGetRandom()->genInt32() % 2) {
        case 0: atomicCall(e, setModel, PSET_CUBE_LARGE);   atomicCall(e, setCollisionShape, CS_BOX); break;
        case 1: atomicCall(e, setModel, PSET_SPHERE_LARGE); atomicCall(e, setCollisionShape, CS_SPHERE); break;
        }
        atomicCall(e, setPosition, GenRandomVector2() * 1.5f);
        atomicCall(e, setHealth, 1800.0f * getLoopBoost());
        atomicCall(e, setAxis1, GenRandomUnitVector3());
        atomicCall(e, setAxis2, GenRandomUnitVector3());
        atomicCall(e, setRotateSpeed1, 0.1f);
        atomicCall(e, setRotateSpeed2, 0.1f);
        atomicCall(e, setRoutine, ROUTINE_SHOOT);
        atomicCall(e, setLightRadius, 1.4f);
        atomicCall(e, setExplosionSE, SE_EXPLOSION5);
        atomicCall(e, setExplosionChannel, SE_CHANNEL5);
        m_large_enemies.push_back(e->getHandle());
        return e;
    }

    void update(float32 dt)
    {
        ++m_frame;
        updateCamera();

        if(m_state==ST_BEGIN) {
            level0();
            m_state = ST_NORMAL;
            m_frame = 0;
        }
        else if(m_state==ST_NORMAL) {
            switch(m_level) {
            case 1: level1(); break;
            case 2: level2(); break;
            case 3: level3(); break;
            case 4: level4(); break;
            case 5: level5(); break;
            case 6: level6(); break;
            default: ++m_loop; m_level=1; break;
            }
            if(m_level > 0 && !isPlayerAlive()) {
                m_frame = 0;
                m_state = ST_GAME_OVER;
                atomicGetFader()->setFade(vec4(0.0f, 0.0f, 0.0f, 1.0f), 300.0f);
            }
        }
        else if(m_state==ST_GAME_OVER) {
            if(m_frame > 300) {
                atomicGetApplication()->requestExit();
            }
        }
    }

    void updateCamera()
    {
        PerspectiveCamera *pcam = atomicGetCamera();
        if(IEntity *player = atomicGetEntity(m_player)) {
            vec4 player_pos = atomicQuery(player, getPosition, vec4);
            vec4 cpos       = pcam->getPosition();
            vec4 tpos       = pcam->getTarget();
            vec4 cpos2      = cpos + (player_pos-cpos)*0.03f;
            vec4 tpos2      = tpos + (player_pos-tpos)*0.03f;
            cpos2.z = cpos.z;
            tpos2.z = tpos.z;
            pcam->setPosition(cpos2);
            pcam->setTarget(tpos2);

            atomicSetListenerPosition(cpos2);
        }
    }

    bool isPlayerAlive()
    {
        if(!atomicGetEntity(m_player)) {
            return false;
        }
        return true;
    }


    bool isAllDead(stl::vector<EntityHandle> &ev)
    {
        for(uint32 i=0; i<ev.size(); ++i) {
            if(atomicGetEntity(ev[i])) { return false; }
        }
        return true;
    }

    bool isAllDead()
    {
        return isAllDead(m_small_enemies) && isAllDead(m_medium_enemies) && isAllDead(m_large_enemies);
    }

    void goNextLevel()
    {
        ++m_level;
        m_frame = 0;
        m_small_enemies.clear();
        m_medium_enemies.clear();
        m_large_enemies.clear();
    }

    void level0()
    {
        {
            //SPHPutParticles(30000);
            IEntity *e = atomicCreateEntity(Player);
            m_player = e->getHandle();
            atomicCall(e, setPosition, vec4(0.0f, 0.0f, 0.0f, 1.0f));
            //atomicCall(e, setHealth, 100000000.0f);
        }
        {
            atomicGetFader()->setColor(vec4(0.0f, 0.0f, 0.0f, 1.0f));
            atomicGetFader()->setFade(vec4(0.0f, 0.0f, 0.0f, 0.0f), 60.0f);
        }
        goNextLevel();
    }

    void level1()
    {
        if(m_frame < 1200) {
            //if(m_frame % 1 == 0) {
            //    putPinballEnemy();
            //    putPinballEnemy();
            //}
            if(m_frame % 50 == 0) {
                IEntity *e = putSmallEnemy();
            }
            if(m_frame==1) {
                //IEntity *e = putLargeEnemy();
            }
        }
        else if(isAllDead()) {
            goNextLevel();
        }
    }

    void level2()
    {
        if(m_frame < 1200) {
            if(m_frame % 60 == 0) {
                IEntity *e = putSmallEnemy();
            }
            if(m_frame % 200 == 0) {
                IEntity *e = putMediumEnemy();
            }
        }
        else if(isAllDead()) {
            goNextLevel();
        }
    }

    void level3()
    {
        if(m_frame < 1200) {
            if(m_frame % 30 == 0) {
                IEntity *e = putSmallEnemy();
            }
            if(m_frame % 220 == 0) {
                IEntity *e = putMediumEnemy();
            }
        }
        else if(isAllDead()) {
            goNextLevel();
        }
    }

    void level4()
    {
        if(m_frame < 1200) {
            if(m_frame % 20 == 0) {
                IEntity *e = putSmallEnemy();
            }
            if(m_frame % 150 == 0) {
                IEntity *e = putMediumEnemy();
            }
        }
        else if(isAllDead()) {
            goNextLevel();
        }
    }

    void level5()
    {
        if(m_frame < 1200) {
            if(m_frame % 15 == 0) {
                IEntity *e = putSmallEnemy();
            }
            if(m_frame % 100 == 0) {
                IEntity *e = putMediumEnemy();
            }
        }
        else if(isAllDead()) {
            goNextLevel();
        }
    }

    void level6()
    {
        if(m_frame < 1200) {
            if(m_frame % 500 == 0) {
                IEntity *e = putLargeEnemy();
            }
            if(m_frame % 50 == 0) {
                IEntity *e = putSmallEnemy();
            }
        }
        else if(isAllDead()) {
            goNextLevel();
        }
    }

    void draw()
    {
        {
            DirectionalLight dl;
            dl.setDirection(glm::normalize(vec4(1.0f, -1.0f, -0.5f, 0.0f)));
            dl.setDiffuse(vec4(0.3f, 0.3f, 0.3f, 1.0f));
            dl.setAmbient(vec4(0.0f, 0.0f, 0.0f, 0.0f));
            atomicGetLights()->addLight(dl);
        }


        float32 health = 0.0f;
        if(IEntity *e = atomicGetEntity(m_player)) {
            health = atomicQuery(e, getHealth, float32);
        }

        char buf[64];
        istsprintf(buf, "life: %.0f", health);
        atomicGetSystemTextRenderer()->addText(vec2(5.0f, 60.0f), buf);
    }
};

atomicImplementEntity(Level_Test, ECID_Level);

} // namespace atomic
