#include "stdafx.h"
#include "ist/ist.h"
#include "types.h"
#include "Game/VFX.h"

namespace atomic {

    template<class T>
    struct VFXData_IsDead
    {
        bool operator()(const T &v) const
        {
            return v.scale <= 0.0f;
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
    uint32 num_data = m_data.size();
    for(uint32 i=0; i<num_data; ++i) {
        Data &data = m_data[i];
        simdvec4 pos = simdvec4(data.position);
        simdvec4 vel = simdvec4(data.velosity);
        data.position = glm::vec4_cast(pos + vel);
        data.frame += dt;
        if(data.frame > 60.0f) {
            data.scale = data.scale -= 0.0003f;
        }
    }
    m_data.erase(
        stl::remove_if(m_data.begin(), m_data.end(), VFXData_IsDead<Data>()),
        m_data.end());
}

void VFXScintilla::draw()
{
}

void VFXScintilla::frameEnd()
{
}

void VFXScintilla::addData( const VFXScintillaData *data, uint32 data_num )
{
    m_data.insert(m_data.end(), data, data+data_num);
}



void VFXExplosion::frameBegin()
{
}

void VFXExplosion::update( float32 dt )
{
}

void VFXExplosion::asyncupdate( float32 dt )
{
}

void VFXExplosion::draw()
{
}

void VFXExplosion::frameEnd()
{
}

void VFXExplosion::addData( const Data *data, uint32 data_num )
{
    m_data.insert(m_data.end(), data, data+data_num);
}



void VFXDebris::frameBegin()
{
}

void VFXDebris::update( float32 dt )
{
}

void VFXDebris::asyncupdate( float32 dt )
{
}

void VFXDebris::draw()
{
}

void VFXDebris::frameEnd()
{
}

void VFXDebris::addData( const Data *data, uint32 data_num )
{
    m_data.insert(m_data.end(), data, data+data_num);
}



VFXSet::VFXSet()
{
    m_scintilla = istNew(VFXScintilla)();
    m_components.push_back(m_scintilla);

    m_explosion = istNew(VFXExplosion)();
    m_components.push_back(m_explosion);

    m_debris    = istNew(VFXDebris)();
    m_components.push_back(m_debris);
}

VFXSet::~VFXSet()
{
    for(uint32 i=0; i<m_components.size(); ++i) {
        istDelete(m_components[i]);
    }
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

} // namespace atomic
