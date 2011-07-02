#include "stdafx.h"
#include "ModelData.h"

namespace ist {
namespace graphics {

ModelData::ModelData()
{
}

bool ModelData::initialize()
{
    m_vbo.initialize();
    m_nbo.initialize();
    m_ibo.initialize();
    return true;
}

void ModelData::finalize()
{
    m_vbo.finalize();
    m_nbo.finalize();
    m_ibo.finalize();
}


void ModelData::setVertex(void *data, size_t num_vertex, VERTEX_FORMAT fmt, USAGE usage)
{
    m_vertex_format = fmt;
    m_vbo.allocate(sizeof(float)*m_vertex_format*num_vertex, (VertexBufferObject::USAGE)usage, data);
}

void ModelData::setNormal(void *data, size_t num_vertex, USAGE usage)
{
    m_nbo.allocate(sizeof(float)*3*num_vertex, (VertexBufferObject::USAGE)usage, data);
}

void ModelData::setIndex(void *data, size_t num_index, INDEX_FORMAT fmt, PRIMITIVE_TYPE prm, USAGE usage)
{
    m_num_index = num_index;
    m_index_format = fmt;
    m_primitive_type = prm;
    int e = fmt==IDX_INT16 ? sizeof(short) : sizeof(int);
    m_ibo.allocate(e*num_index, (IndexBufferObject::USAGE)usage, data);
}

void ModelData::draw() const
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

void ModelData::drawInstanced(GLuint num_intance) const
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

} // namespace graphics
} // namespace ist
