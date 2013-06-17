#include "stdafx.h"
#include "ist/ist.h"
#include "types.h"
#include "Graphics/ResourceManager.h"
#include "Graphics/Renderer.h"
#include "Game/VFX.h"
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
        IndivisualParticle particles;
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
        particles[i].position = spawn.position + (GenRandomUnitVector3() * spawn.scatter_radius);
        particles[i].color = spawn.color;
        particles[i].glow = spawn.glow;
        particles[i].size = spawn.size;
        particles[i].velosity = spawn.velosity + (GenRandomUnitVector3() * spawn.diffuse_strength);
    }
    m_particles.insert(m_particles.end(), particles, particles+spawn.num_particles);
}

atmExportClass(atm::VFXSet);

VFXSet::VFXSet()
    : m_scintilla(nullptr)
    , m_components()
{
}

VFXSet::~VFXSet()
{
    for(uint32 i=0; i<m_components.size(); ++i) {
        istDelete(m_components[i]);
    }
}

void VFXSet::initialize()
{
    m_scintilla = istNew(VFXScintilla)();
    m_components.push_back(m_scintilla);
}

void VFXSet::frameBegin()
{
    for(uint32 i=0; i<m_components.size(); ++i) {
        m_components[i]->frameBegin();
    }
}

void VFXSet::update( float32 dt )
{
    for(uint32 i=0; i<m_components.size(); ++i) {
        m_components[i]->update(dt);
    }
}

void VFXSet::asyncupdate( float32 dt )
{
    for(uint32 i=0; i<m_components.size(); ++i) {
        m_components[i]->asyncupdate(dt);
    }
}

void VFXSet::draw()
{
    for(uint32 i=0; i<m_components.size(); ++i) {
        m_components[i]->draw();
    }
}

void VFXSet::frameEnd()
{
    for(uint32 i=0; i<m_components.size(); ++i) {
        m_components[i]->frameEnd();
    }
}

} // namespace atm
