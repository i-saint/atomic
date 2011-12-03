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
: m_idgen(0)
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
    static bool s_init = false;
    if(!s_init) {
        s_init = true;
        float32 xv[4] = {2.5f, -2.5f, 0.0f, 0.0f};
        float32 yv[4] = {0.0f, 0.0f, 2.5f, -2.5f};
        for(uint32 i=0; i<4; ++i) {
            Message_GenerateFraction mes;
            mes.gen_type = Message_GenerateFraction::GEN_SPHERE;
            mes.num = 3000;

            ist::Sphere sphere;
            sphere.x = xv[i];
            sphere.y = yv[i];
            sphere.z = 0.0f;
            sphere.r = 1.2f;
            mes.assignData<ist::Sphere>(sphere);
            atomicPushMessage(MR_FRACTION, 0, mes);
        }
    }

    {
        SPHSphericalGravity h_sg;
        h_sg.position = make_float4(0.0f);
        h_sg.is_active = 1;
        h_sg.inner_radus = 0.5f;
        h_sg.range_radus = 5.12f;
        h_sg.strength = 1.5f;
        for(uint32 i=0; i<_countof(m_sgravity); ++i) {
            m_sgravity[i] = h_sg;
        }

    }
}


void FractionSet::update()
{
    m_task_asyncupdate->kick();

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
    SPHUpdateGrid();
    SPHComputeDensity();
    SPHComputeForce();
    SPHIntegrate();

    float2 move = atomicGetInputs()->getMove()*0.01f;
    m_sgravity[0].position.x += move.x;
    m_sgravity[0].position.y += move.y;
    SPHUpdateSphericalGravityData(m_sgravity);

    //PerformanceCounter counter;
    //CUDA_SAFE_CALL( cudaMemcpyFromSymbol(m_particles, "d_particles", sizeof(m_particles), 0, cudaMemcpyDeviceToHost ) );
    //IST_PRINT("copying d_particles took %f ms.\n", counter.getElapsedMillisecond());
}


void FractionSet::processMessage()
{
    MessageIterator<Message_GenerateFraction> mes_gf_iter;
    while(mes_gf_iter.hasNext()) {
        const Message_GenerateFraction& mes = mes_gf_iter.iterate();
        for(uint32 n=0; n<mes.num; ++n) {
            if(mes.gen_type==Message_GenerateFraction::GEN_SPHERE) {
                ist::Sphere& sphere = (ist::Sphere&)(*mes.shape_data);
                SPHParticle fd;
                fd.id = ++m_idgen;
                fd.lifetime = 0xFFFFFFFF;
                fd.velocity = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                fd.position = make_float4(sphere.x, sphere.y, sphere.z, 0.0f);

                float4 r = make_float4(atomicGenRandFloat(),atomicGenRandFloat(),atomicGenRandFloat(), 0.0f);
                r = (r - make_float4(0.5f)) * make_float4(2.0f) * make_float4(sphere.r);
                fd.position += r;
                fd.velocity = make_float4(atomicGenRandFloat(),atomicGenRandFloat(),atomicGenRandFloat(),0.0f) * make_float4(0.2f);

                m_spawn.push_back(fd);
            }
            else if(mes.gen_type==Message_GenerateFraction::GEN_BOX) {
                IST_ASSERT("Message_GenerateFraction::GEN_BOX is not implemented yet");
            }
        }
    }

    //const uint32 num_data = m_data.size();
}



void FractionSet::draw() const
{
    PassGBuffer_Cube *cube = atomicGetCubeRenderer();
    PassDeferred_PointLight *light = atomicGetSphereLightRenderer();

    //cube->pushFractionInstance(make_float4(0.0f));
    //light->pushInstance(make_float4(0.2f, 0.2f, 0.2f, 0.2f));

    size_t num_data = _countof(m_particles);
    for(uint32 i=0; i<num_data; ++i) {
        //cube->pushFractionInstance(m_particles[i].position);
    }
    for(uint32 i=0; i<num_data; ++i) {
        if(m_particles[i].id % (SPH_MAX_PARTICLE_NUM>>6)==0) {
            //light->pushInstance(m_particles[i].position);
        }
    }
}


} // namespace atomic
