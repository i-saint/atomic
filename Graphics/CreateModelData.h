#ifndef __atomic_Graphics_CreateModelData__
#define __atomic_Graphics_CreateModelData__

#include "GPGPU/SPH.cuh"

namespace atomic {

    class CudaBuffer;

    void CreateSphere(VertexArray& va, VertexBufferObject& vbo, IndexBufferObject& ibo, float32 radius, uint32 div_xz, uint32 div_y);
    void CreateScreenQuad(VertexArray& va, VertexBufferObject& vbo);
    void CreateBloomLuminanceQuads(VertexArray& va, VertexBufferObject& vbo);
    void CreateBloomBlurQuads(VertexArray& va, VertexBufferObject& vbo);
    void CreateBloomCompositeQuad(VertexArray& va, VertexBufferObject& vbo);
    void CreateCube(VertexArray& va, VertexBufferObject& vbo, float32 len);

    bool CreateCubeParticleSet(CudaBuffer& ps, SPHCharacterClass &sphcc, float32 len);
    bool CreateSphereParticleSet(CudaBuffer& ps, SPHCharacterClass &sphcc, float32 radius);

} // namespace atomic
#endif // __atomic_Graphics_ModelData_h__
