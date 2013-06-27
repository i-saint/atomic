#include "stdafx.h"
#include "types.h"
#include "Graphics/ResourceManager.h"
#include "Graphics/Renderer.h"
#include "Game/VFXModule.h"
#include "Util.h"

namespace atm {

    template<class T>
    struct VFXData_IsDead
    {
        bool operator()(const T &v) const
        {
            return v.size <= 0.0f;
        }
    };


void VFXScintilla::frameBegin()
{
}

void VFXScintilla::update( float32 dt )
{
}

void VFXScintilla::asyncupdate( float32 dt )
{
    uint32 num_data = m_particles.size();
    for(uint32 i=0; i<num_data; ++i) {
        ParticleData &data = m_particles[i];
        simdvec4 pos = simdvec4(data.position);
        simdvec4 vel = simdvec4(data.velosity);
        data.position = glm::vec4_cast(pos + vel);
        data.frame += dt;
        if(data.frame > 60.0f) {
            data.size = data.size -= 0.0003f;
        }
    }
    m_particles.erase(
        stl::remove_if(m_particles.begin(), m_particles.end(), VFXData_IsDead<ParticleData>()),
        m_particles.end());
}

void VFXScintilla::draw()
{
    uint32 num_data = m_particles.size();
    for(uint32 i=0; i<num_data; ++i) {
        ParticleData &data = m_particles[i];
        SingleParticle particles;
        particles.position = data.position;
        particles.color = data.color;
        particles.glow = data.glow;
        atmGetParticlePass()->addParticle(&particles, 1);
    }
}

void VFXScintilla::frameEnd()
{
}

void VFXScintilla::addData( const VFXScintillaSpawnData &spawn )
{
    ParticleData *particles = (ParticleData*)_alloca(sizeof(ParticleData)*spawn.num_particles);
    for(uint32 i=0; i<spawn.num_particles; ++i) {
        particles[i].position = spawn.position + (vec4(GenRandomUnitVector3(),0.0f) * spawn.scatter_radius);
        particles[i].color = spawn.color;
        particles[i].glow = spawn.glow;
        particles[i].size = spawn.size;
        particles[i].velosity = spawn.velosity + (vec4(GenRandomUnitVector3(),0.0f) * spawn.diffuse_strength);
    }
    m_particles.insert(m_particles.end(), particles, particles+spawn.num_particles);
}

atmExportClass(VFXModule);

VFXModule::VFXModule()
    : m_scintilla(nullptr)
    , m_components()
{
}

VFXModule::~VFXModule()
{
    for(uint32 i=0; i<m_components.size(); ++i) {
        istDelete(m_components[i]);
    }
}

void VFXModule::initialize()
{
    m_scintilla = istNew(VFXScintilla)();
    m_components.push_back(m_scintilla);
}

void VFXModule::frameBegin()
{
    for(uint32 i=0; i<m_components.size(); ++i) {
        m_components[i]->frameBegin();
    }
}

void VFXModule::update( float32 dt )
{
    for(uint32 i=0; i<m_components.size(); ++i) {
        m_components[i]->update(dt);
    }
}

void VFXModule::asyncupdate( float32 dt )
{
    for(uint32 i=0; i<m_components.size(); ++i) {
        m_components[i]->asyncupdate(dt);
    }
}

void VFXModule::draw()
{
    for(uint32 i=0; i<m_components.size(); ++i) {
        m_components[i]->draw();
    }
}

void VFXModule::frameEnd()
{
    for(uint32 i=0; i<m_components.size(); ++i) {
        m_components[i]->frameEnd();
    }
}

} // namespace atm
