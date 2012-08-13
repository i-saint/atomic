#ifndef __atomic_Graphics_CreateModelData__
#define __atomic_Graphics_CreateModelData__

#include "GPGPU/SPH.cuh"

namespace atomic {

    class CudaBuffer;
    class ParticleSet;

    void CreateFloorQuad(VertexArray *va, Buffer *&vbo, vec4 pos, vec4 size);
    void CreateSphere(VertexArray *va, Buffer *&vbo, Buffer *&ibo, float32 radius, uint32 div_xz, uint32 div_y);
    void CreateScreenQuad(VertexArray *va, Buffer *&vbo);
    void CreateBloomLuminanceQuads(VertexArray *va, Buffer *&vbo);
    void CreateBloomBlurQuads(VertexArray *va, Buffer *&vbo);
    void CreateBloomCompositeQuad(VertexArray *va, Buffer *&vbo);
    void CreateCube(VertexArray *va, Buffer *&vbo, float32 len);
    void CreateFieldGridLines(VertexArray *va, Buffer *&vbo);
    void CreateDistanceFieldQuads(VertexArray *va, Buffer *&quad_model, Buffer *&quad_pos, Buffer *&quad_dist);

    bool CreateCubeParticleSet(ParticleSet &pset, RigidInfo &ri, float32 half_len);
    bool CreateSphereParticleSet(ParticleSet &pset, RigidInfo &ri, float32 radius);
    bool CreateBulletParticleSet(ParticleSet &pset, RigidInfo &ri);

} // namespace atomic
#endif // __atomic_Graphics_ModelData_h__
