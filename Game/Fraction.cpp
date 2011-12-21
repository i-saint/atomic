#include "stdafx.h"
#include "ist/ist.h"
#include "types.h"
#include "AtomicApplication.h"
#include "Graphics/Renderer.h"
#include "Game/Message.h"
#include "Game/AtomicGame.h"
#include "Game/World.h"
#include "Game/Fraction.h"

namespace atomic {




FractionSet::FractionSet()
{
    m_task_asyncupdate = IST_NEW(AsyncUpdateTask)(this);
}

FractionSet::~FractionSet()
{
    sync();
    IST_SAFE_DELETE(m_task_asyncupdate);
}


void FractionSet::initialize()
{
    {
        SPHSphericalGravity h_sg;
        h_sg.position = make_float4(0.0f);
        h_sg.is_active = 1;
        h_sg.inner_radus = 0.25f;
        h_sg.range_radus = 5.12f;
        h_sg.strength = 1.0f;
        for(uint32 i=0; i<_countof(m_sgravity); ++i) {
            m_sgravity[i] = h_sg;
        }

    }
}


void FractionSet::update()
{
    m_task_asyncupdate->kick();

    m_rigids.clear();

    //updateAsync();
}

void FractionSet::sync() const
{
    m_task_asyncupdate->join();
}


void FractionSet::updateAsync()
{
    processMessage();
    updateSPH();
}

void FractionSet::updateSPH()
{
    // todo: copy character data to device


    vec2 move = atomicGetInputs()->getMove()*0.01f;
    m_sgravity[0].position.x += move.x;
    m_sgravity[0].position.y += move.y;

    SPHUpdateGravity(m_sgravity);
    SPHUpdateRigids(m_rigids);
    SPHUpdateFluid();

    //PerformanceCounter counter;
    //CUDA_SAFE_CALL( cudaMemcpyFromSymbol(m_particles, "d_particles", sizeof(m_particles), 0, cudaMemcpyDeviceToHost ) );
    //IST_PRINT("copying d_particles took %f ms.\n", counter.getElapsedMillisecond());
}


void FractionSet::processMessage()
{
}

void FractionSet::draw() const
{
}


} // namespace atomic
