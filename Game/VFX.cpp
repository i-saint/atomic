#include "stdafx.h"
#include "ist/ist.h"
#include "types.h"
#include "Game/VFX.h"

namespace atomic {


void VFXScintilla::frameBegin()
{
}

void VFXScintilla::update( float32 dt )
{
}

void VFXScintilla::updateEnd()
{
}

void VFXScintilla::asyncupdate( float32 dt )
{
}

void VFXScintilla::draw()
{
}

void VFXScintilla::frameEnd()
{
}

void VFXScintilla::addData( const VFXScintillaData *data, uint32 data_num )
{
}

void VFXExplosion::frameBegin()
{
}



void VFXExplosion::update( float32 dt )
{
}

void VFXExplosion::updateEnd()
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

void VFXExplosion::addData( const VFXScintillaData *data, uint32 data_num )
{
}



void VFXDebris::frameBegin()
{
}

void VFXDebris::update( float32 dt )
{
}

void VFXDebris::updateEnd()
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

void VFXDebris::addData( const VFXScintillaData *data, uint32 data_num )
{
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
