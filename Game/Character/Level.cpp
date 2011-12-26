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


class Level_Test : public IEntity
{
private:
    int m_frame;

public:
    Level_Test() : m_frame(0)
    {
    }

    void update(float32 dt)
    {
        ++m_frame;
        updateCamera();

        if(m_frame==1) {
            {
                IEntity *e =  atomicGetEntitySet()->createEntity<Player>();
                atomicCall(e, setPosition, vec4(0.0f, 0.0f, 0.0f, 1.0f));
            }
            {
                IEntity *e =  atomicGetEntitySet()->createEntity<Enemy_Cube>();
                atomicCall(e, setPosition, vec4(0.5f, 0.0f, 0.0f, 1.0f));
                atomicCall(e, setAxis1, GenRotateAxis());
                atomicCall(e, setAxis2, GenRotateAxis());
                atomicCall(e, setRotateSpeed1, 0.4f);
                atomicCall(e, setRotateSpeed2, 0.4f);
            }
            {
                IEntity *e =  atomicGetEntitySet()->createEntity<Enemy_Sphere>();
                atomicCall(e, setPosition, vec4(-0.5f, 0.0f, 0.0f, 1.0f));
                atomicCall(e, setAxis1, GenRotateAxis());
                atomicCall(e, setAxis2, GenRotateAxis());
                atomicCall(e, setRotateSpeed1, 0.4f);
                atomicCall(e, setRotateSpeed2, 0.4f);
            }
        }
    }

    void updateCamera()
    {
        PerspectiveCamera *pcam = atomicGetCamera();
        if(IEntity *player = atomicGetEntity( EntityCreateHandle(ECID_PLAYER, ESID_PLAYER, 0) )) {
            variant v;
            if(atomicQuery(player, getPosition, v)) {
                vec4 player_pos = v.cast<vec4>();
                vec4 cpos = pcam->getPosition();
                vec4 tpos = pcam->getTarget();
                vec4 cpos2 = cpos + (player_pos-cpos)*0.02f;
                vec4 tpos2 = tpos + (player_pos-tpos)*0.02f;
                cpos2.z = cpos.z;
                tpos2.z = tpos.z;
                pcam->setPosition(cpos2);
                pcam->setTarget(tpos2);
            }
        }
    }
};

atomicImplementEntity(Level_Test, ECID_LEVEL, ESID_LEVEL_TEST);

} // namespace atomic
