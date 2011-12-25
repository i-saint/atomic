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
    };

    atomicImplementEntity(Level_Test, ECID_LEVEL, ESID_LEVEL_TEST);

} // namespace atomic
