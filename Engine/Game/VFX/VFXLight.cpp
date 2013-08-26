#include "atmPCH.h"
#include "VFXCommon.h"
#include "VFXLight.h"

namespace atm {

void VFXLight::frameBegin()
{
}

void VFXLight::update( float32 dt )
{
}

void VFXLight::asyncupdate( float32 dt )
{
}

void VFXLight::draw()
{
}

void VFXLight::frameEnd()
{
}

void VFXLight::addData( const SpawnData &spawn )
{
}

atmExportClass(VFXLight);
IVFXComponent* VFXLightCreate() { return istNew(VFXLight)(); }
void VFXLightSpawn(const VFXLightSpawnData &v) { atmGetVFXModule()->getLight()->addData(v); }

} // namespace atm
