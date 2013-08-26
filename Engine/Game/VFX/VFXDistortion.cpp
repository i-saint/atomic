#include "atmPCH.h"
#include "VFXCommon.h"
#include "VFXDistortion.h"

namespace atm {


void VFXShockwave::frameBegin()
{
}

void VFXShockwave::update( float32 dt )
{
}

void VFXShockwave::asyncupdate( float32 dt )
{
}

void VFXShockwave::draw()
{
}

void VFXShockwave::frameEnd()
{
}

void VFXShockwave::addData( const VFXShockwaveSpawnData &spawn )
{
}

atmExportClass(VFXShockwave);
IVFXComponent* VFXShockwaveCreate() { return istNew(VFXShockwave)(); }
void VFXShockwaveSpawn(const VFXShockwaveSpawnData &v) { atmGetVFXModule()->getShockwave()->addData(v); }

} // namespace atm
