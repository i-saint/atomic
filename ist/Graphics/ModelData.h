#ifndef __ist_Graphics_ModelData__
#define __ist_Graphics_ModelData__

#include "BufferObject.h"
#include "GraphicsAssert.h"

namespace ist {
namespace graphics {


class ModelData : public VertexArray
{
typedef VertexArray super;
public:
    enum INDEX_FORMAT
    {
        IDX_INT16 = GL_UNSIGNED_SHORT,
        IDX_INT32 = GL_UNSIGNED_INT,
    };
    enum PRIMITIVE_TYPE
    {
        PRM_POINTS          = GL_POINTS,
        PRM_LINES           = GL_LINES,
        PRM_TRIANGLES       = GL_TRIANGLES,
        PRM_TRIANGLE_STRIP  = GL_TRIANGLE_STRIP,
        PRM_TRIANGLE_FAN    = GL_TRIANGLE_FAN,
        PRM_QUADS           = GL_QUADS,
    };
    enum USAGE
    {
        USAGE_STATIC  = VertexBufferObject::USAGE_STATIC,
        USAGE_DYNAMIC = VertexBufferObject::USAGE_DYNAMIC,
        USAGE_STREAM  = VertexBufferObject::USAGE_STREAM,
    };
    enum {
        MAX_ATTRIBUTES = 4,
    };

private:
    VertexBufferObject m_data[MAX_ATTRIBUTES];
    IndexBufferObject m_ibo;
    int m_index_format;
    int m_primitive_type;
    size_t m_num_index;

public:
    ModelData();
    ~ModelData();
    bool initialize();
    void finalize();
    // num_elements: 1,2,3,4
    void setData(int index, void *data, size_t num_vertex, size_t num_elements, USAGE usage=USAGE_STATIC);
    void setInstanceData(int index, size_t num_elements, VertexBufferObject &data);
    void setIndex(void *data, size_t num_index, INDEX_FORMAT fmt, PRIMITIVE_TYPE prm, USAGE usage=USAGE_STATIC);

    void draw() const;
    void drawInstanced(GLuint num_intance) const;
};

} // namespace graphics
} // namespace ist
#endif // __ist_Graphics_ModelData__
