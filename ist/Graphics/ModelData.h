#ifndef __ist_Graphics_ModelData__
#define __ist_Graphics_ModelData__

#include "BufferObject.h"
#include "GraphicsAssert.h"

namespace ist {
namespace graphics {


class ModelData
{
public:
    enum VERTEX_FORMAT
    {
        VTX_FLOAT2 = 2,
        VTX_FLOAT3 = 3,
        VTX_FLOAT4 = 4,
    };
    enum INDEX_FORMAT
    {
        IDX_INT16 = GL_UNSIGNED_SHORT,
        IDX_INT32 = GL_UNSIGNED_INT,
    };
    enum PRIMITIVE_TYPE
    {
        PRM_POINTS      = GL_POINTS,
        PRM_LINES       = GL_LINES,
        PRM_TRIANGLES   = GL_TRIANGLES,
        PRM_QUADS       = GL_QUADS,
    };
    enum USAGE
    {
        USAGE_STATIC  = VertexBufferObject::USAGE_STATIC,
        USAGE_DYNAMIC = VertexBufferObject::USAGE_DYNAMIC,
        USAGE_STREAM  = VertexBufferObject::USAGE_STREAM,
    };

private:
    VertexBufferObject m_vbo;
    VertexBufferObject m_nbo;
    IndexBufferObject m_ibo;
    int m_vertex_format;
    int m_index_format;
    int m_primitive_type;
    size_t m_num_index;

public:
    ModelData();
    bool initialize();
    void finalize();
    void setVertex(void *data, size_t num_vertex, VERTEX_FORMAT fmt, USAGE usage);
    void setNormal(void *data, size_t num_vertex, USAGE usage);
    void setIndex(void *data, size_t num_index, INDEX_FORMAT fmt, PRIMITIVE_TYPE prm, USAGE usage);

    void draw() const;
    void drawInstanced(GLuint num_intance) const;
};

} // namespace graphics
} // namespace ist
#endif // __ist_Graphics_ModelData__
