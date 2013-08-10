#include "stdafx.h"
#include "Entity/EntityCommon.h"
#include "Entity/Routine.h"
#include "Entity/Level.h"

namespace atm {

class dpPatch Level1 : public EntityWithPosition
{
typedef EntityWithPosition super;
private:
    enum State {
        St_Begin,
        St_Scene1,
        St_Scene2,
        St_Scene3,
        St_Scene4,
        St_Wait,
        St_End,
        St_GameOver,
    };

    CollisionHandle m_planes[4];
    EntityHandle m_player;
    EntityHandle m_boss;
    EntityHandle m_layer;
    EntityHandle m_guards[3];
    State m_state;
    int m_frame_total;
    int m_frame_scene;

    istSerializeBlock(
        istSerializeBase(super)
        istSerialize(m_planes)
        istSerialize(m_player)
        istSerialize(m_boss)
        istSerialize(m_layer)
        istSerialize(m_guards)
        istSerialize(m_state)
        istSerialize(m_frame_total)
        istSerialize(m_frame_scene)
    )
    atmJsonizeBlock(
        atmJsonizeSuper(super)
    )

public:
    Level1()
        : m_player(), m_boss(), m_layer()
        , m_state(St_Begin)
        , m_frame_total(0), m_frame_scene(0)
    {
        clear(m_planes);
        clear(m_guards);

        wdmScope(
        wdmString path = wdmFormat("Level/Level1/0x%p", this);
        super::addDebugNodes(path);
        wdmAddNode(path+"/m_state", (int32*)&m_state);
        wdmAddNode(path+"/m_frame_total", &m_frame_total);
        wdmAddNode(path+"/m_frame_scene", &m_frame_scene);
        )
    }

    ~Level1()
    {
        wdmEraseNode(wdmFormat("Level/Level1/0x%p", this));
    }

    void initialize()
    {
        super::initialize();
        atmGetBackgroundPass()->setBGShader(SH_BG6);

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
            p->setFlags(CF_SPH_Sender);
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


    State getState() const  { return m_state; }
    void  setState(State v) { m_state=v; m_frame_scene=0; }

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
        if(m_player!=0 && !atmGetEntity(m_player)) {
            return false;
        }
        return true;
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


    void update(float32 dt)
    {
        updateCamera();
        if(dt==0.0f) { return; }

        ++m_frame_total;
        ++m_frame_scene;
        switch(getState()) {
        case St_Begin:  sceneBegin(dt); break;
        case St_Scene1: scene1(dt); break;
        case St_Scene2: scene2(dt); break;
        case St_Scene3: scene3(dt); break;
        case St_Scene4: scene4(dt); break;
        case St_Wait:   sceneWait(dt); break;
        case St_End:    break;
        }
        if(getState()==St_GameOver) {
            if(m_frame_scene > 300) {
                atmGetApplication()->requestReturnToTitleScreen();
            }
        }
        else {
            if(!isPlayerAlive()) {
                setState(St_GameOver);
                atmGetFader()->setFade(vec4(0.0f, 0.0f, 0.0f, 1.0f), 300.0f);
            }
        }
    }


    IEntity* putElectron()
    {
        IEntity *e = nullptr;
        e = atmCreateEntityT(Electron);
        atmCall(e, setPosition, vec3(GenRandomVector2()*2.2f, 0.0f));
        return e;
    }

    IEntity* putProton()
    {
        IEntity *e = nullptr;
        e = atmCreateEntityT(Proton);
        atmCall(e, setPosition, vec3(GenRandomVector2()*2.2f, 0.0f));
        return e;
    }


    void sceneBegin(float32 dt)
    {
        IEntity *e = atmCreateEntityT(Player);
        m_player = e->getHandle();
        atmCall(e, setPosition, vec4(0.0f, 0.0f, 0.0f, 1.0f));

        setState(St_Scene3);
    }

    void scene1(float32 dt)
    {
        int32 f = m_frame_scene;
        if(f < 500) {
            if(f % 40 == 0) {
                IEntity *e = putElectron();
            }
        }
        else if(f < 1400) {
            if(f % 60 == 0) {
                IEntity *e = putElectron();
            }
            if(f % 200 == 0) {
                IEntity *e = putProton();
            }
        }
        if(f>1000) {
            setState(St_Scene2);
        }
    }

    IEntity* _PutChildEntity(EntityClassID ecid, IEntity *parent, const vec3 &pos)
    {
        IEntity *e = atmCreateEntity(ecid);
        atmCall(e, setPosition, pos);
        atmCall(e, setParent, parent->getHandle());
        return e;
    }
    #define PutChildEntity(Class, Parent, Pos) _PutChildEntity(EC_##Class, Parent, Pos)

    void scene2(float32 dt)
    {
        int32 f = m_frame_scene;
        if(f==1) {
            float32 x = 4.0f;
            float32 scroll = 0.003f;
            float32 timespan = 6000.0f;
            IEntity *layer = atmCreateEntityT(LevelLayer);
            atmCall(layer, addPositionXCP, ControlPoint(    0.0f, x,  0.0f, 0.0f, ControlPoint::Linear));
            atmCall(layer, addPositionXCP, ControlPoint(timespan, x-scroll*timespan,  0.0f, 0.0f));
            atmCall(layer, setLifeTime, timespan);

            const vec4 light_color(0.5f, 0.7f, 1.0f, 1.0f);
            IEntity *e = nullptr;
            e = PutChildEntity(PointLightEntity, layer, vec3(1.75f, 0.5f, 0.3f));
            atmCall(e, setRadius, 1.0f);
            atmCall(e, setDiffuse, light_color);
            e = PutChildEntity(PointLightEntity, layer, vec3(3.75f, -1.1f, 0.3f));
            atmCall(e, setRadius, 1.0f);
            atmCall(e, setDiffuse, light_color);

            e = PutChildEntity(GearSmall, layer, vec3(0.0f, 0.7f, 0.0f));
            e = PutChildEntity(GearMedium, layer, vec3(0.85f, -0.9f, 0.0f));
            e = PutChildEntity(GearLarge, layer, vec3(1.95f, 0.5f, 0.0f));
            e = PutChildEntity(GearSmall, layer, vec3(2.3f, -1.4f, 0.0f));
            e = PutChildEntity(GearLarge, layer, vec3(3.95f, -1.1f, 0.0f));
            e = PutChildEntity(GearMedium, layer, vec3(4.1f, 1.1f, 0.0f));
      }

        {
            if(f % 40 == 0) {
                IEntity *e = putElectron();
            }
            if(f % 300 == 0) {
                IEntity *e = putProton();
            }
        }

        if(f>2600) {
            setState(St_Scene3);
        }
    }

    void scene3(float32 dt)
    {
        int32 f = m_frame_scene;
        if(f==1) {
            float32 x = 4.0f;
            float32 scroll = 0.003f;
            float32 timespan = 6000.0f;
            IEntity *layer = atmCreateEntityT(LevelLayer);
            atmCall(layer, addPositionXCP, ControlPoint(    0.0f, x,  0.0f, 0.0f, ControlPoint::Linear));
            atmCall(layer, addPositionXCP, ControlPoint(timespan, x-scroll*timespan,  0.0f, 0.0f));
            atmCall(layer, setLifeTime, timespan);

            IEntity *e = nullptr;
            const vec4 light_color(0.5f, 0.7f, 1.0f, 1.0f);
            e = PutChildEntity(PointLightEntity, layer, vec3(0.0f, 0.0f, 0.3f));
            atmCall(e, setRadius, 1.0f);
            atmCall(e, setDiffuse, light_color);
            e = PutChildEntity(PointLightEntity, layer, vec3(2.5f, 0.0f, 0.3f));
            atmCall(e, setRadius, 1.0f);
            atmCall(e, setDiffuse, light_color);

            e = PutChildEntity(Antiproton, layer, vec3(-0.7f, 0.0f, 0.0f));

            PutGroundBlockByBox(layer, 1, vec3(1.0f, 0.5f, -0.1f), vec3(1.5f, 2.5f, 0.1f));
            PutGroundBlockByBox(layer, 1, vec3(1.0f,-0.5f, -0.1f), vec3(1.5f,-2.5f, 0.1f));
            {
                IEntity *block = PutGroundBlockByBox(layer, 1, vec3(1.5f, 1.0f, -0.1f), vec3(1.85f, 0.0f, 0.15f));
                IEntity *gear = PutChildEntity(GearSmall, layer, vec3(0.0f, 0.5f, 0.0f));
                IEntity *linkage = atmCreateEntityT(GateLinkage);
                atmCall(gear, setSpinMinAngle,   0.0f);
                atmCall(gear, setSpinMaxAngle, 720.0f);
                atmCall(gear, setSpinReturnSpeed, 0.02f);
                atmCall(gear, setSpinOneWay, 1.0f);
                atmCall(linkage, setBlock, block->getHandle());
                atmCall(linkage, setGear, gear->getHandle());
                atmCall(linkage, setSlideDir, vec3(0.0f,1.0f,0.0f));
                atmCall(linkage, setLinkSpeed, 0.5f/720.0f);
            }
            {
                IEntity *block = PutGroundBlockByBox(layer, 1, vec3(1.5f,-1.0f, -0.1f), vec3(1.85f, 0.0f, 0.15f));
                IEntity *gear = PutChildEntity(GearSmall, layer, vec3(0.0f, -0.5f, 0.0f));
                IEntity *linkage = atmCreateEntityT(GateLinkage);
                atmCall(gear, setSpinMinAngle,-720.0f);
                atmCall(gear, setSpinMaxAngle,   0.0f);
                atmCall(gear, setSpinReturnSpeed, 0.02f);
                atmCall(gear, setSpinOneWay, -1.0f);
                atmCall(linkage, setBlock, block->getHandle());
                atmCall(linkage, setGear, gear->getHandle());
                atmCall(linkage, setSlideDir, vec3(0.0f,1.0f,0.0f));
                atmCall(linkage, setLinkSpeed, 0.5f/720.0f);
            }
            PutGroundBlockByBox(layer, 1, vec3(1.5f,-2.0f, -0.1f), vec3(4.0f,-2.5f, 0.15f));
            PutGroundBlockByBox(layer, 1, vec3(3.5f,-2.0f, -0.1f), vec3(4.0f,-0.5f, 0.15f));
            PutFluidFilterByBox(layer, 1, vec3(3.5f,-0.5f, -0.1f), vec3(4.0f, 0.5f, 0.15f));
            PutGroundBlockByBox(layer, 1, vec3(3.5f, 0.5f, -0.1f), vec3(4.0f, 1.5f, 0.15f));
            PutGroundBlockByBox(layer, 1, vec3(4.0f, 1.5f, -0.1f), vec3(5.0f, 1.0f, 0.15f));
            PutRigidFilterByBox(layer, 1, vec3(3.5f, 1.5f, -0.1f), vec3(4.0f, 2.5f, 0.15f));
            e = PutChildEntity(Antiproton, layer, vec3( 5.9f,-2.0f, 0.0f));
            e = PutChildEntity(Antiproton, layer, vec3( 6.5f,-2.0f, 0.0f));
        }

        {
            if(f % 25 == 0) {
                IEntity *e = putElectron();
            }
            if(f % 500 == 0) {
                IEntity *e = putProton();
            }
        }

        if(f>3000) {
            setState(St_Scene4);
        }
    }

    void scene4(float32 dt)
    {
        int32 f = m_frame_scene;
        if(f==1) {
            float32 x = 4.0f;
            float32 scroll = 0.003f;
            float32 timespan = 1500.0f;
            IEntity *layer = atmCreateEntityT(LevelLayer);
            m_layer = layer->getHandle();
            atmCall(layer, addPositionXCP, ControlPoint(    0.0f, x,  0.0f, 0.0f, ControlPoint::Linear));
            atmCall(layer, addPositionXCP, ControlPoint(timespan, x-scroll*timespan,  0.0f, 0.0f));
            atmCall(layer, setLifeTime, 0.0f);

            IEntity *e = nullptr;
            {
                IEntity *gear = PutChildEntity(GearSmall, layer, vec3(-0.5f, 1.9f, 0.0f));
                atmCall(gear, setSpinMinAngle,   0.0f);
                atmCall(gear, setSpinMaxAngle, 720.0f);
                atmCall(gear, setSpinReturnSpeed, 0.01f);
                atmCall(gear, setSpinOneWay, 1.0f);
                IEntity *l = PutChildEntity(LevelLayer, layer, vec3(0.0f, 1.25f, 0.0f));
                PutGroundBlockByBox(l, 1, vec3(-0.75f,-0.1f, -0.1f), vec3(0.0f,0.1f, 0.15f));
                PutGroundBlockByBox(l, 1, vec3(-0.2f,-0.6f, -0.1f), vec3(0.0f,0.1f, 0.15f));
                IEntity *linkage = atmCreateEntityT(HingeLinkage);
                atmCall(linkage, setBlock, l->getHandle());
                atmCall(linkage, setGear, gear->getHandle());
                atmCall(linkage, setLinkSpeed, 150.0f/720.0f);
            }
            {
                IEntity *gear = PutChildEntity(GearSmall, layer, vec3(-0.5f,-1.9f, 0.0f));
                atmCall(gear, setSpinMinAngle,-720.0f);
                atmCall(gear, setSpinMaxAngle,   0.0f);
                atmCall(gear, setSpinReturnSpeed, 0.01f);
                atmCall(gear, setSpinOneWay, -1.0f);
                IEntity *l = PutChildEntity(LevelLayer, layer, vec3(0.0f,-1.25f, 0.0f));
                PutGroundBlockByBox(l, 1, vec3(-0.75f,-0.1f, -0.1f), vec3(0.0f,0.1f, 0.15f));
                PutGroundBlockByBox(l, 1, vec3(-0.2f, 0.6f, -0.1f), vec3(0.0f,-0.1f, 0.15f));
                IEntity *linkage = atmCreateEntityT(HingeLinkage);
                atmCall(linkage, setBlock, l->getHandle());
                atmCall(linkage, setGear, gear->getHandle());
                atmCall(linkage, setLinkSpeed, 150.0f/720.0f);
            }
        }
        if(f==1500) {
            IEntity *layer = atmGetEntity(m_layer);
            IEntity *core = PutChildEntity(Core, layer, vec3(1.9f, 0.0f, 0.0f));
            IEntity *barrier = PutFluidFilter(core, 1, vec3(0.0f, 0.0f, 0.0f), vec3(1.4f));
            m_boss = core->getHandle();
        }
        if(f > 1750) {
            if(f % 60 == 0) {
                IEntity *e = putElectron();
            }
            if(f % 1200 == 0) {
                IEntity *e = putProton();
            }
        }
        if(f > 1900) {
            SweepDeadEntities(m_guards);
            IEntity *layer = atmCreateEntityT(LevelLayer);
            IEntity *e = nullptr;
            if(m_guards[0]==0 && f%900==0) {
                e = PutChildEntity(Antiproton, layer, vec3(1.2f, 1.4f, 0.0f));
                m_guards[0] = e->getHandle();
            }
            if(m_guards[1]==0 && f%960==0) {
                e = PutChildEntity(Antiproton, layer, vec3(1.2f,-1.4f, 0.0f));
                m_guards[1] = e->getHandle();
            }
            if(m_guards[2]==0 && f%1020==0) {
                e = PutChildEntity(Antiproton, layer, vec3(-2.0f, 0.0f, 0.0f));
                atmCall(e, setRoutine, RCID_Routine_FixedLaser);
                m_guards[2] = e->getHandle();
            }
        }


        if(f>1500) {
            SweepDeadEntities(m_boss);
            if(!m_boss) {
                setState(St_Wait);
            }
        }
    }

    void sceneWait(float32 dt)
    {
        int32 f = m_frame_scene;
        if(f==1) {
            atmGetFader()->setFade(vec4(0.0f, 0.0f, 0.0f, 1.0f), 300.0f);
        }
        if(f==300) {
            atmGetApplication()->requestReturnToTitleScreen();
        }
    }
};
atmImplementEntity(Level1);
atmExportClass(Level1);


} // namespace atm
