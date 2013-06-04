#include "psym.h"


namespace psym {

void impIntegrateDOL(ispc::Particle * all_particles, ispc::GridData * grid, int32_t xi, int32_t yi)
{
    ispc::impIntegrate(all_particles, grid, xi, yi);
}

void impUpdateVelocityDOL(ispc::Particle * all_particles, ispc::GridData * grid, int32_t xi, int32_t yi)
{
    ispc::impUpdateVelocity(all_particles, grid, xi, yi);
}

void sphInitializeConstantsDOL()
{
    ispc::sphInitializeConstants();
}

void sphIntegrateDOL(ispc::Particle * all_particles, ispc::GridData * grid, int32_t xi, int32_t yi)
{
    ispc::sphIntegrate(all_particles, grid, xi, yi);
}

void sphProcessCollisionDOL(
    ispc::Particle * all_particles, ispc::GridData * grid, int32_t xi, int32_t yi,
    ispc::RigidSphere * spheres, int32_t num_spheres,
    ispc::RigidPlane * planes, int32_t num_planes,
    ispc::RigidBox * boxes, int32_t num_boxes )
{
    ispc::sphProcessCollision(all_particles, grid, xi, yi, spheres, num_spheres, planes, num_planes, boxes, num_boxes);
}

void sphProcessExternalForceDOL(
    ispc::Particle * all_particles, ispc::GridData * grid, int32_t xi, int32_t yi,
    ispc::PointForce * pforce, int32_t num_pforce,
    ispc::DirectionalForce * dforce, int32_t num_dforce,
    ispc::BoxForce * bforce, int32_t num_bforce )
{
    ispc::sphProcessExternalForce(all_particles, grid, xi, yi, pforce, num_pforce, dforce, num_dforce, bforce, num_bforce);
}

void sphUpdateDensityDOL(ispc::Particle * all_particles, ispc::GridData * grid, int32_t xi, int32_t yi)
{
    ispc::sphUpdateDensity(all_particles, grid, xi, yi);
}

#ifdef psym_enable_neighbor_density_estimation
void sphUpdateDensity2DOL(ispc::Particle * all_particles, ispc::GridData * grid, int32_t xi, int32_t yi)
{
    ispc::sphUpdateDensity2(all_particles, grid, xi, yi);
}
#endif // psym_enable_neighbor_density_estimation

void sphUpdateForceDOL(ispc::Particle * all_particles, ispc::GridData * grid, int32_t xi, int32_t yi)
{
    ispc::sphUpdateForce(all_particles, grid, xi, yi);
}

} // namespace psym

