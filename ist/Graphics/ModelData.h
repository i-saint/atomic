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
    ModelData()
    {
        m_vbo.initialize();
        m_nbo.initialize();
        m_ibo.initialize();
    }

    void setVertex(void *data, size_t num_vertex, VERTEX_FORMAT fmt, USAGE usage)
    {
        m_vertex_format = fmt;
        m_vbo.allocate(sizeof(float)*m_vertex_format*num_vertex, (VertexBufferObject::USAGE)usage, data);
    }

    void setNormal(void *data, size_t num_vertex, USAGE usage)
    {
        m_nbo.allocate(sizeof(float)*3*num_vertex, (VertexBufferObject::USAGE)usage, data);
    }

    void setIndex(void *data, size_t num_index, INDEX_FORMAT fmt, PRIMITIVE_TYPE prm, USAGE usage)
    {
        m_num_index = num_index;
        m_index_format = fmt;
        m_primitive_type = prm;
        int e = fmt==IDX_INT16 ? sizeof(short) : sizeof(int);
        m_ibo.allocate(e*num_index, (IndexBufferObject::USAGE)usage, data);
    }

    void draw() const
    {
        m_vbo.bind();
        glVertexPointer(m_vertex_format, GL_FLOAT, 0, 0);
        m_nbo.bind();
        glNormalPointer(GL_FLOAT, 0, 0);
        m_ibo.bind();
        glEnableClientState(GL_VERTEX_ARRAY);
        glEnableClientState(GL_NORMAL_ARRAY);
        glDrawElements(m_primitive_type, m_num_index, m_index_format, 0);
        glDisableClientState(GL_NORMAL_ARRAY);
        glDisableClientState(GL_VERTEX_ARRAY);
    }

    void drawInstanced(GLuint num_intance) const
    {
        m_vbo.bind();
        glVertexPointer(m_vertex_format, GL_FLOAT, 0, 0);
        m_nbo.bind();
        glNormalPointer(GL_FLOAT, 0, 0);
        m_ibo.bind();
        glEnableClientState(GL_VERTEX_ARRAY);
        glEnableClientState(GL_NORMAL_ARRAY);
        glDrawElementsInstanced(m_primitive_type, m_num_index, m_index_format, 0, num_intance);
        glDisableClientState(GL_NORMAL_ARRAY);
        glDisableClientState(GL_VERTEX_ARRAY);
    }
};

} // namespace graphics
} // namespace ist
#endif // __ist_Graphics_ModelData__
