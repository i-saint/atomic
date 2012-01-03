#ifndef __atomic_Graphics_CreateModelData__
#define __atomic_Graphics_CreateModelData__

#include "GPGPU/SPH.cuh"

namespace atomic {

    class CudaBuffer;
    class ParticleSet;

    void CreateSphere(VertexArray& va, VertexBuffer& vbo, IndexBuffer& ibo, float32 radius, uint32 div_xz, uint32 div_y);
    void CreateScreenQuad(VertexArray& va, VertexBuffer& vbo);
    void CreateBloomLuminanceQuads(VertexArray& va, VertexBuffer& vbo);
    void CreateBloomBlurQuads(VertexArray& va, VertexBuffer& vbo);
    void CreateBloomCompositeQuad(VertexArray& va, VertexBuffer& vbo);
    void CreateCube(VertexArray& va, VertexBuffer& vbo, float32 len);
    void CreateFieldGridLines(VertexArray& va, VertexBuffer& vbo);
    void CreateDistanceFieldQuads(VertexArray& va, VertexBuffer& quad_model, VertexBuffer& quad_pos, VertexBuffer& quad_dist);

    bool CreateCubeParticleSet(ParticleSet &pset, RigidInfo &ri, float32 half_len);
    bool CreateSphereParticleSet(ParticleSet &pset, RigidInfo &ri, float32 radius);

} // namespace atomic
#endif // __atomic_Graphics_ModelData_h__
