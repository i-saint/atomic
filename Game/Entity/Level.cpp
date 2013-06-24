#include "stdafx.h"
#include "Game/Entity/EntityCommon.h"
#include "Game/Entity/Routine.h"

namespace atm {


class dpPatch Level_Test : public IEntity
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
    ist::vector<EntityHandle> m_small_enemies;
    ist::vector<EntityHandle> m_medium_enemies;
    ist::vector<EntityHandle> m_large_enemies;

    int32 m_level;
    int32 m_loop;
    STATE m_state;

    istSerializeBlock(
        istSerializeBase(super)
        istSerialize(m_frame)
        istSerialize(m_planes)
        istSerialize(m_player)
        istSerialize(m_small_enemies)
        istSerialize(m_medium_enemies)
        istSerialize(m_large_enemies)
        istSerialize(m_level)
        istSerialize(m_loop)
        istSerialize(m_state)
    )

public:
    Level_Test() : m_frame(0), m_player(0), m_level(0), m_loop(0), m_state(ST_BEGIN)
    {
        stl::fill_n(m_planes, _countof(m_planes), 0);
    }

    void initialize()
    {
        super::initialize();

        const vec4 field_size = vec4(PSYM_GRID_SIZE*0.5f);
        atmGetWorld()->setFieldSize(field_size);

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
            CollisionPlane *p = atmCreateCollision(CollisionPlane);
            p->setEntityHandle(getHandle());
            p->setFlags(CF_Sender|CF_SPH_Sender);
            p->bb = bboxes[i];
            p->plane = planes[i];
            m_planes[i] = p->getCollisionHandle();
        }
    }

    void finalize()
    {
        for(uint32 i=0; i<_countof(m_planes); ++i) {
            atmDeleteCollision(m_planes[i]);
        }
        super::finalize();
    }

    float32 getLoopBoost() const { return 1.0f+(0.2f * m_loop); }

    IEntity* putSmallEnemy()
    {
        IEntity *e = NULL;
        e = atmCreateEntity(Enemy_Test);
        atmCall(e, setCollisionShape, CS_Sphere);
        atmCall(e, setModel, PSET_SPHERE_SMALL);
        atmCall(e, setPosition, vec3(GenRandomVector2()*2.2f, 0.0f));
        atmCall(e, setLife, 15.0f * getLoopBoost());
        atmCall(e, setAxis1, GenRandomUnitVector3());
        atmCall(e, setAxis2, GenRandomUnitVector3());
        atmCall(e, setRotateSpeed1, 2.4f);
        atmCall(e, setRotateSpeed2, 2.4f);
        atmCall(e, setRoutine, RCID_Routine_HomingPlayer);
        atmCall(e, setLightRadius, 0.5f);
        m_small_enemies.push_back(e->getHandle());
        return e;
    }

    IEntity* putPinballEnemy()
    {
        IEntity *e = NULL;
        e = atmCreateEntity(Enemy_Test);
        atmCall(e, setCollisionShape, CS_Sphere);
        atmCall(e, setCollisionFlags, CF_Receiver|CF_Sender);
        atmCall(e, setModel, PSET_SPHERE_BULLET);
        atmCall(e, setPosition, vec3(GenRandomVector2(),0.0f) * 0.5f + vec3(1.5f, 1.5f, 0.0f));
        atmCall(e, setLife, 5.0f * getLoopBoost());
        atmCall(e, setAxis1, GenRandomUnitVector3());
        atmCall(e, setAxis2, GenRandomUnitVector3());
        atmCall(e, setRotateSpeed1, 2.4f);
        atmCall(e, setRotateSpeed2, 2.4f);
        atmCall(e, setRoutine, RCID_Routine_Pinball);
        atmCall(e, setLightRadius, 0.0f);
        atmCall(e, setVelocity, (vec4(0.0f, -0.99f, 0.0f, 0.0f))*0.005f);
        //atmCall(e, setAccel, GenRandomUnitVector2()*0.00005f);
        atmCall(e, setAccel, vec4(0.0f, -1.0f, 0.0f, 0.0f)*0.00005f);
        return e;
    }

    IEntity* putMediumEnemy()
    {
        IEntity *e = atmCreateEntity(Enemy_Test);
        switch(atmGetRandom()->genInt32() % 2) {
        case 0: atmCall(e, setModel, PSET_CUBE_MEDIUM);  atmCall(e, setCollisionShape, CS_Box); break;
        case 1: atmCall(e, setModel, PSET_SPHERE_MEDIUM);atmCall(e, setCollisionShape, CS_Sphere); break;
        }
        atmCall(e, setPosition, vec3(GenRandomVector2()*2.1f, 0.0f));
        atmCall(e, setLife, 100.0f * getLoopBoost());
        atmCall(e, setAxis1, GenRandomUnitVector3());
        atmCall(e, setAxis2, GenRandomUnitVector3());
        atmCall(e, setRotateSpeed1, 0.4f);
        atmCall(e, setRotateSpeed2, 0.4f);
        atmCall(e, setRoutine, RCID_Routine_SingleShoot);
        atmCall(e, setLightRadius, 0.8f);
        atmCall(e, setExplosionSE, SE_EXPLOSION4);
        atmCall(e, setExplosionChannel, SE_CHANNEL4);
        m_medium_enemies.push_back(e->getHandle());
        return e;
    }

    IEntity* putLargeEnemy()
    {
        IEntity *e = atmCreateEntity(Enemy_Test);
        //atmCall(e, setModel, PSET_CUBE_LARGE); atmCall(e, setCollisionShape, CS_BOX);
        switch(atmGetRandom()->genInt32() % 2) {
        case 0: atmCall(e, setModel, PSET_CUBE_LARGE);   atmCall(e, setCollisionShape, CS_Box); break;
        case 1: atmCall(e, setModel, PSET_SPHERE_LARGE); atmCall(e, setCollisionShape, CS_Sphere); break;
        }
        atmCall(e, setPosition, vec3(GenRandomVector2()*1.5f, 0.0f));
        atmCall(e, setLife, 1200.0f * getLoopBoost());
        atmCall(e, setAxis1, GenRandomUnitVector3());
        atmCall(e, setAxis2, GenRandomUnitVector3());
        atmCall(e, setRotateSpeed1, 0.1f);
        atmCall(e, setRotateSpeed2, 0.1f);
        atmCall(e, setRoutine, RCID_Routine_SingleShoot);
        atmCall(e, setLightRadius, 1.4f);
        atmCall(e, setExplosionSE, SE_EXPLOSION5);
        atmCall(e, setExplosionChannel, SE_CHANNEL5);
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
                atmGetFader()->setFade(vec4(0.0f, 0.0f, 0.0f, 1.0f), 300.0f);
            }
        }
        else if(m_state==ST_GAME_OVER) {
            if(m_frame > 300) {
                atmGetApplication()->requestExit();
            }
        }
    }

    void updateCamera()
    {
        PerspectiveCamera *pcam = atmGetGameCamera();
        if(IEntity *player = atmGetEntity(m_player)) {
            vec3 player_pos;
            atmQuery(player, getPosition, player_pos);
            vec3 cpos       = pcam->getPosition();
            vec3 tpos       = pcam->getTarget();
            vec3 cpos2      = cpos + (player_pos-cpos)*0.03f;
            vec3 tpos2      = tpos + (player_pos-tpos)*0.03f;
            cpos2.z = cpos.z;
            tpos2.z = tpos.z;
            pcam->setPosition(cpos2);
            pcam->setTarget(tpos2);

            PerspectiveCamera *bgcam = atmGetBGCamera();
            *bgcam = *pcam;

            atmSetListenerPosition(cpos2);
        }
    }

    bool isPlayerAlive()
    {
        if(!atmGetEntity(m_player)) {
            return false;
        }
        return true;
    }


    bool isAllDead(ist::vector<EntityHandle> &ev)
    {
        for(uint32 i=0; i<ev.size(); ++i) {
            if(atmGetEntity(ev[i])) { return false; }
        }
        return true;
    }

    bool isAllDead()
    {
        return isAllDead(m_small_enemies) && isAllDead(m_medium_enemies) && isAllDead(m_large_enemies);
    }

    void goNextLevel()
    {
        {
            atmDebugLog("goNextLevel(): %d\n", (uint32)atmGetFrame());
        }
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
            IEntity *e = atmCreateEntity(Player);
            m_player = e->getHandle();
            atmCall(e, setPosition, vec4(0.0f, 0.0f, 0.0f, 1.0f));
            atmCall(e, setLife, 100000000.0f);
        }
        {
            atmGetFader()->setColor(vec4(0.0f, 0.0f, 0.0f, 1.0f));
            atmGetFader()->setFade(vec4(0.0f, 0.0f, 0.0f, 0.0f), 60.0f);
        }
        goNextLevel();
    }

    void level1()
    {
        // 敵テスト用
        if(m_frame==1) {
            {
                IEntity *e = putLargeEnemy();
                atmCall(e, setRoutine, RCID_Routine_CircularShoot);
                atmCall(e, setLife, 10000.0f);
            }
            {
                IEntity *e = atmCreateEntity(GroundBlock);
                atmCall(e, setPosition, vec3(0.5f, 0.5f, 0.0f));
            }
            {
                IEntity *e = atmCreateEntity(GearSmall);
                atmCall(e, setPosition, vec3(-1.0f, -1.0f, 0.0f));
            }
        }

        if(m_frame < 1200) {
            //if(m_frame % 1 == 0) {
            //    putPinballEnemy();
            //    putPinballEnemy();
            //}
            if(m_frame % 50 == 0) {
                IEntity *e = putSmallEnemy();
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
            dl.setDirection(glm::normalize(vec3(1.0f, -1.0f, -0.5f)));
            dl.setDiffuse(vec4(0.3f, 0.3f, 0.3f, 1.0f));
            dl.setAmbient(vec4(0.0f, 0.0f, 0.0f, 0.0f));
            atmGetLightPass()->addLight(dl);
        }


        float32 health = 0.0f;
        if(IEntity *e = atmGetEntity(m_player)) {
            atmQuery(e, getLife, health);
        }

        char buf[64];
        istSPrintf(buf, "life: %.0f", health);
        atmGetTextRenderer()->addText(vec2(5.0f, 60.0f), buf);
    }
};
atmImplementEntity(Level_Test);
atmExportClass(Level_Test);


} // namespace atm
