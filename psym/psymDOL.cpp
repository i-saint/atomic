#include "psym.h"

DOL_Module

namespace psym {

DOL_Export void impIntegrateDOL(ispc::Particle * all_particles, ispc::GridData * grid, int32_t xi, int32_t yi)
{
    ispc::impIntegrate(all_particles, grid, xi, yi);
}

DOL_Export void impUpdateVelocityDOL(ispc::Particle * all_particles, ispc::GridData * grid, int32_t xi, int32_t yi)
{
    ispc::impUpdateVelocity(all_particles, grid, xi, yi);
}

DOL_Export void sphInitializeConstantsDOL()
{
    ispc::sphInitializeConstants();
}

DOL_Export void sphIntegrateDOL(ispc::Particle * all_particles, ispc::GridData * grid, int32_t xi, int32_t yi)
{
    ispc::sphIntegrate(all_particles, grid, xi, yi);
}

DOL_Export void sphProcessCollisionDOL(
    ispc::Particle * all_particles, ispc::GridData * grid, int32_t xi, int32_t yi,
    ispc::RigidSphere * spheres, int32_t num_spheres,
    ispc::RigidPlane * planes, int32_t num_planes,
    ispc::RigidBox * boxes, int32_t num_boxes )
{
    ispc::sphProcessCollision(all_particles, grid, xi, yi, spheres, num_spheres, planes, num_planes, boxes, num_boxes);
}

DOL_Export void sphProcessExternalForceDOL(
    ispc::Particle * all_particles, ispc::GridData * grid, int32_t xi, int32_t yi,
    ispc::PointForce * pforce, int32_t num_pforce,
    ispc::DirectionalForce * dforce, int32_t num_dforce,
    ispc::BoxForce * bforce, int32_t num_bforce )
{
    ispc::sphProcessExternalForce(all_particles, grid, xi, yi, pforce, num_pforce, dforce, num_dforce, bforce, num_bforce);
}

DOL_Export void sphUpdateDensityDOL(ispc::Particle * all_particles, ispc::GridData * grid, int32_t xi, int32_t yi)
{
    ispc::sphUpdateDensity(all_particles, grid, xi, yi);
}

DOL_Export void sphUpdateForceDOL(ispc::Particle * all_particles, ispc::GridData * grid, int32_t xi, int32_t yi)
{
    ispc::sphUpdateForce(all_particles, grid, xi, yi);
}

} // namespace psym


DOL_OnLoad({
})
    
DOL_OnUnload({
})
