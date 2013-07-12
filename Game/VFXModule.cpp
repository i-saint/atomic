#include "stdafx.h"
#include "types.h"
#include "Game/VFXModule.h"
#include "Game/VFX/VFXInterfaces.h"
#include "Util.h"

namespace atm {



istSerializeBlockImpl(VFXModule,
    istSerializeBase(super)
    istSerialize(m_components)
    istSerialize(m_scintilla)
    istSerialize(m_light)
    istSerialize(m_shockwave)
    istSerialize(m_feedbackblur)
)
atmExportClass(VFXModule);

VFXModule::VFXModule()
    : m_scintilla(), m_light(), m_shockwave(), m_feedbackblur()
{
}

VFXModule::~VFXModule()
{
    for(uint32 i=0; i<m_components.size(); ++i) {
        m_components[i]->release();
    }
    m_components.clear();
}

void VFXModule::initialize()
{
    m_scintilla = VFXScintillaCreate();
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
