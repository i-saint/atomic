#include "istPCH.h"
#ifdef ist_with_OpenGL
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
    setRef(1);
    istSafeAddRef(m_device);
}

DeviceContext::~DeviceContext()
{
    istSafeRelease(m_device);
}

void DeviceContext::setViewport( const Viewport &vp )
{
    vp.bind();
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
        glUseProgram(0);
    }
}

void DeviceContext::setRenderTarget( RenderTarget *rt )
{
    m_render_target = rt;
    if(m_render_target != NULL) {
        m_render_target->bind();
    }
    else {
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }
}

void DeviceContext::setSampler( uint32 i, Sampler *smp )
{
    if(smp!=NULL) {
        smp->bind(i);
    }
    else {
        glBindSampler(i, 0);
    }
}

void DeviceContext::setTexture( uint32 i, Texture *tex )
{
    if(tex!=NULL) {
        tex->bind(i);
    }
    else {
        glActiveTexture(GL_TEXTURE0+i);
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
    glDrawElementsInstancedBaseVertex(topology, num_indices, m_index_format, NULL, num_instances, first_vertex);
}

void DeviceContext::applyRenderStates()
{
    // todo: render state の変更要求はバッファリングして実際の変更はここでやる
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
#endif // ist_with_OpenGL
