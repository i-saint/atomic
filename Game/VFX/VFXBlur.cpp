#include "stdafx.h"
#include "VFXCommon.h"
#include "VFXBlur.h"

namespace atm {

void VFXFeedbackBlur::frameBegin()
{
}

void VFXFeedbackBlur::update( float32 dt )
{
}

void VFXFeedbackBlur::asyncupdate( float32 dt )
{
}

void VFXFeedbackBlur::draw()
{
}

void VFXFeedbackBlur::frameEnd()
{
}

void VFXFeedbackBlur::addData( const VFXFeedbackBlurSpawnData &spawn )
{
}

atmExportClass(VFXFeedbackBlur);
IVFXComponent* VFXFeedbackBlurCreate() { return istNew(VFXFeedbackBlur)(); }
void VFXFeedbackBlurSpawn(const VFXFeedbackBlurSpawnData &v) { atmGetVFXModule()->getFeedbackBlur()->addData(v); }

} // namespace atm
