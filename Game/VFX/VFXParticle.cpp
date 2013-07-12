#include "stdafx.h"
#include "VFXCommon.h"

namespace atm {

struct istAlign(16) VFXScintillaParticleData
{
    simdvec4 position;
    simdvec4 velosity;
    simdvec4 color;
    simdvec4 glow;
    float32 scale;
    float32 time;
    float32 pad[2];
};
atmSerializeRaw(VFXScintillaParticleData);
istStaticAssert(sizeof(VFXScintillaParticleData)%16==0);

class VFXScintilla : public IVFXComponent
{
typedef IVFXComponent super;
private:
    typedef VFXScintillaParticleData ParticleData;
    typedef ist::raw_vector<VFXScintillaParticleData> ParticleCont;
    typedef ist::raw_vector<SingleParticle> DrawDataCont;

    ParticleCont m_particles;
    DrawDataCont m_drawdata; // serialize 不要

    istSerializeBlock(
        istSerializeBase(super)
        istSerialize(m_particles)
    )

public:
    void frameBegin() override
    {
    }

    void update( float32 dt ) override
    {
    }

    void asyncupdate( float32 dt ) override
    {
        uint32 num_data = m_particles.size();
        for(uint32 i=0; i<num_data; ++i) {
            ParticleData &data = m_particles[i];
            data.position += data.velosity;
            data.time += dt;
            if(data.time > 60.0f) {
                data.scale -= 0.01f;
            }
        }
        erase(m_particles, [](ParticleData &v){ return v.time <= 0.0f; });
    }

    void draw() override
    {
        uint32 num_data = m_particles.size();
        m_drawdata.resize(num_data);
        for(uint32 i=0; i<num_data; ++i) {
            ParticleData &data = m_particles[i];
            SingleParticle &drawdata = m_drawdata[i];;
            drawdata.position = (vec4&)data.position;
            drawdata.color = (vec4&)data.color;
            drawdata.glow = (vec4&)data.glow;
            drawdata.scale = data.scale;
        }
        atmGetParticlePass()->addParticle(&m_drawdata[0], num_data);
    }

    void frameEnd() override
    {
    }

    void addData( const VFXScintillaSpawnData &spawn )
    {
        ParticleData *particles = (ParticleData*)_alloca(sizeof(ParticleData)*spawn.num_particles);
        for(uint32 i=0; i<spawn.num_particles; ++i) {
            particles[i].position = simdvec4(vec4(spawn.position + (GenRandomUnitVector3()*spawn.scatter_radius), 0.0f));
            particles[i].velosity = simdvec4(vec4(spawn.velosity + (GenRandomUnitVector3()*spawn.diffuse_strength), 0.0f));
            particles[i].color = simdvec4(spawn.color);
            particles[i].glow = simdvec4(spawn.glow);
            particles[i].scale = spawn.scale;
        }
        m_particles.insert(m_particles.end(), particles, particles+spawn.num_particles);
    }
};
atmExportClass(VFXScintilla);

IVFXComponent* VFXScintillaCreate() { return istNew(VFXScintilla)(); }
void VFXScintillaSpawn(const VFXScintillaSpawnData &v) {
    atmGetVFXModule()->getScintilla()->addData(v);
}


} // namespace atm
