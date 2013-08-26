#include "atmPCH.h"
#include "VFXCommon.h"
#include "VFXParticle.h"

namespace atm {

void VFXScintilla::frameBegin()
{
}

void VFXScintilla::update( float32 dt )
{
}

void VFXScintilla::asyncupdate( float32 dt )
{
    uint32 num_data = m_vfxdata.size();
    for(uint32 i=0; i<num_data; ++i) {
        VFXData &data = m_vfxdata[i];
        data.position += data.velosity;
        data.time += dt;
        if(data.time > 60.0f) {
            data.scale -= 0.01f;
        }
    }
    erase(m_vfxdata, [](VFXData &v){ return v.time <= 0.0f; });
}

void VFXScintilla::draw()
{
    uint32 num_data = m_vfxdata.size();
    m_drawdata.resize(num_data);
    for(uint32 i=0; i<num_data; ++i) {
        VFXData &data = m_vfxdata[i];
        SingleParticle &drawdata = m_drawdata[i];;
        drawdata.position = (vec4&)data.position;
        drawdata.color = (vec4&)data.color;
        drawdata.glow = (vec4&)data.glow;
        drawdata.scale = data.scale;
    }
    atmGetParticlePass()->addParticle(&m_drawdata[0], num_data);
}

void VFXScintilla::frameEnd()
{
}

void VFXScintilla::addData( const VFXScintillaSpawnData &spawn )
{
    VFXData *particles = (VFXData*)_alloca(sizeof(VFXData)*spawn.num_particles);
    for(uint32 i=0; i<spawn.num_particles; ++i) {
        particles[i].position = simdvec4(vec4(spawn.position + (GenRandomUnitVector3()*spawn.scatter_radius), 0.0f));
        particles[i].velosity = simdvec4(vec4(spawn.velosity + (GenRandomUnitVector3()*spawn.diffuse_strength), 0.0f));
        particles[i].color = simdvec4(spawn.color);
        particles[i].glow = simdvec4(spawn.glow);
        particles[i].scale = spawn.scale;
    }
    m_vfxdata.insert(m_vfxdata.end(), particles, particles+spawn.num_particles);
}

atmExportClass(VFXScintilla);
IVFXComponent* VFXScintillaCreate() { return istNew(VFXScintilla)(); }
void VFXScintillaSpawn(const VFXScintillaSpawnData &v) { atmGetVFXModule()->getScintilla()->addData(v); }

} // namespace atm
