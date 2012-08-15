#include "istPCH.h"
#ifdef __ist_with_OpenGL__
#include "ist/Base.h"
#include "i3dglDevice.h"
#include "i3dglDeviceContext.h"

namespace ist {
namespace i3dgl {

DeviceContext::DeviceContext( Device *dev )
    : m_device(dev)
    , m_vertex_array(NULL)
    , m_render_target(NULL)

    , m_index_buffer(NULL)
    , m_index_format(I3D_UINT)

    , m_shader(NULL)
{
}

DeviceContext::~DeviceContext()
{
}

void DeviceContext::setVertexArray( VertexArray *va )
{
    m_vertex_array = va;
    if(m_vertex_array) {
        m_vertex_array->bind();
    }
    else {
        m_vertex_array->unbind();
    }
}

void DeviceContext::setIndexBuffer( Buffer *v, I3D_TYPE format )
{
    m_index_buffer = v;
    m_index_format = format;

    if(m_index_buffer != NULL) {
        m_index_buffer->bind();
    }
    else {
        m_index_buffer->unbind();
    }
}

void DeviceContext::setShader( ShaderProgram *v )
{
    m_shader = v;
    if(m_shader != NULL) {
        m_shader->bind();
    }
    else {
        m_shader->unbind();
    }
}

void DeviceContext::setRenderTarget( RenderTarget *rt )
{
    m_render_target = rt;
    if(m_render_target != NULL) {
        m_render_target->bind();
    }
    else {
        m_render_target->unbind();
    }
}

void DeviceContext::setTexture( uint32 i, Texture *tex )
{
    if(tex!=NULL) {
        tex->bind(i);
    }
    else {
        glBindTexture(GL_TEXTURE_2D, 0);
    }
}


void DeviceContext::draw( I3D_TOPOLOGY topology, uint32 first_vertex, uint32 num_vertices )
{
    applyRenderStates();
    glDrawArrays(topology, first_vertex, num_vertices);
}

void DeviceContext::drawIndexed( I3D_TOPOLOGY topology, uint32 first_vertex, uint32 num_indices )
{
    applyRenderStates();
    glDrawElements(topology, first_vertex, num_indices, NULL);
}

void DeviceContext::drawInstanced( I3D_TOPOLOGY topology, uint32 first_vertex, uint32 num_vertices, uint32 num_instances )
{
    applyRenderStates();
    glDrawArraysInstanced(topology, first_vertex, num_vertices, num_instances);
}

void DeviceContext::drawIndexedInstanced( I3D_TOPOLOGY topology, uint32 first_vertex, uint32 num_indices, uint32 num_instances )
{
    applyRenderStates();
    glDrawElementsInstanced(topology, num_indices, m_index_format, NULL, num_instances);
    //glDrawElementsInstancedBaseVertex(topology, num_indices, m_index_format, NULL, num_instances, first_vertex);
}

void DeviceContext::applyRenderStates()
{
}

void DeviceContext::clearColor( RenderTarget *rt, vec4 color )
{
    rt->bind();
    glClearColor(color.x, color.y, color.z, color.w);
    glClear(GL_COLOR_BUFFER_BIT);

}

void DeviceContext::clearDepth( RenderTarget *rt, float32 depth )
{
    rt->bind();
    glClearDepth(depth);
    glClear(GL_DEPTH_BUFFER_BIT);
}

void DeviceContext::clearStencil( RenderTarget *rt, int32 stencil )
{
    rt->bind();
    glClearStencil(stencil);
    glClear(GL_STENCIL_BUFFER_BIT);
}

} // namespace i3dgl
} // namespace ist
#endif // __ist_with_OpenGL__
