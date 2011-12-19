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

    for(uint32 i=0; i<_countof(m_matrices); ++i) { m_matrices[i].clear(); }

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
    SPHUpdate();

    vec2 move = atomicGetInputs()->getMove()*0.01f;
    m_sgravity[0].position.x += move.x;
    m_sgravity[0].position.y += move.y;
    SPHUpdateSphericalGravityData(m_sgravity);

    //PerformanceCounter counter;
    //CUDA_SAFE_CALL( cudaMemcpyFromSymbol(m_particles, "d_particles", sizeof(m_particles), 0, cudaMemcpyDeviceToHost ) );
    //IST_PRINT("copying d_particles took %f ms.\n", counter.getElapsedMillisecond());
}


void FractionSet::processMessage()
{
}



void FractionSet::draw() const
{
    PassGBuffer_Fluid *cube = atomicGetFluidRenderer();
    PassDeferredShading_PointLights *light = atomicGetPointLights();

    //cube->pushFractionInstance(make_float4(0.0f));
    //light->pushInstance(make_float4(0.2f, 0.2f, 0.2f, 0.2f));

    size_t num_data = _countof(m_particles);
    for(uint32 i=0; i<num_data; ++i) {
        //cube->pushFractionInstance(m_particles[i].position);
    }
    for(uint32 i=0; i<num_data; ++i) {
        if(m_particles[i].id % (SPH_MAX_FLUID_PARTICLES>>6)==0) {
            //light->pushInstance(m_particles[i].position);
        }
    }
}


} // namespace atomic
