#include "stdafx.h"
#include "../Base/Assert.h"
#include "ModelData.h"

namespace ist {
namespace graphics {

ModelData::ModelData()
: m_index_format(0)
, m_primitive_type(0)
, m_num_index(0)
{
}

ModelData::~ModelData()
{
    finalize();
}

bool ModelData::initialize()
{
    super::initialize();
    for(size_t i=0; i<_countof(m_data); ++i) {
        m_data[i].initialize();
    }
    m_ibo.initialize();
    return true;
}

void ModelData::finalize()
{
    m_ibo.finalize();
    for(size_t i=0; i<_countof(m_data); ++i) {
        m_data[i].finalize();
    }
    super::finalize();
}


void ModelData::setData(int index, void *data, size_t num_vertex, size_t num_elements, USAGE usage)
{
    if(index>=MAX_ATTRIBUTES) {
        IST_ASSERT("");
    }
    m_data[index].allocate(sizeof(float)*num_vertex*num_elements, (VertexBufferObject::USAGE)usage, data);
    m_data[index].bind();
    super::setAttribute(index, num_elements, m_data[index]);
    m_data[index].unbind();
}

void ModelData::setInstanceData(int index, size_t num_elements, VertexBufferObject &data)
{
    super::setInstanceAttribute(index, num_elements, data);
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
    bind();
    m_ibo.bind();
    glDrawElements(m_primitive_type, m_num_index, m_index_format, 0);
    m_ibo.unbind();
    unbind();
}

void ModelData::drawInstanced(GLuint num_intance) const
{
    super::bind();
    m_ibo.bind();
    glDrawElementsInstanced(m_primitive_type, m_num_index, m_index_format, 0, num_intance);
    m_ibo.unbind();
    super::unbind();
}

} // namespace graphics
} // namespace ist
