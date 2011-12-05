#ifndef __atomic_Graphics_CreateModelData_h__
#define __atomic_Graphics_CreateModelData_h__
namespace atomic {

    void CreateQuadModel(ModelData& model, float32 len);
    void CreateCubeModel(ModelData& model, float32 len);
    void CreateOctahedronModel(ModelData& model, float32 len_xz, float32 len_y);
    void CreateSphereModel(ModelData& model, float32 radius, uint32 div_xz=16, uint32 div_y=16);
    void CreateCylinderModel(ModelData& model, float32 len_y, float32 radius);

    void CreateScreenQuad(VertexArray& va, VertexBufferObject& vbo);
    void CreateBloomLuminanceQuads(VertexArray& va, VertexBufferObject& vbo);
    void CreateBloomBlurQuads(VertexArray& va, VertexBufferObject& vbo);
    void CreateBloomCompositeQuad(VertexArray& va, VertexBufferObject& vbo);

} // namespace atomic
#endif // __atomic_Graphics_ModelData_h__
