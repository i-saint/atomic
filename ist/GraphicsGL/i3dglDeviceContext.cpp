#include "istPCH.h"
#ifdef ist_with_OpenGL
#include "ist/Base.h"
#include "i3dglDevice.h"
#include "i3dglDeviceContext.h"

namespace ist {
namespace i3dgl {

DeviceContext::DeviceContext( Device *dev )
    : m_device(dev)
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
    const ivec2 &pos = vp.getPosition();
    const uvec2 &size = vp.getSize();
    glViewport(pos.x, pos.y, size.x, size.y);
}

void DeviceContext::setVertexArray( VertexArray *va )
{
    m_current.vertex_array = va;
    if(m_current.vertex_array) {
        m_current.vertex_array->bind();
    }
    else {
        m_current.vertex_array->unbind();
    }
}

void DeviceContext::setIndexBuffer( Buffer *v, I3D_TYPE format )
{
    m_current.index_buffer = v;
    m_current.index_format = format;

    if(m_current.index_buffer != NULL) {
        m_current.index_buffer->bind();
    }
    else {
        m_current.index_buffer->unbind();
    }
}

void DeviceContext::setShader( ShaderProgram *v )
{
    m_current.shader = v;
    if(m_current.shader != NULL) {
        m_current.shader->bind();
    }
    else {
        glUseProgram(0);
    }
}

void DeviceContext::setRenderTarget( RenderTarget *rt )
{
    m_current.render_target = rt;
    if(m_current.render_target != NULL) {
        m_current.render_target->bind();
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

void DeviceContext::setBlendState( BlendState *state )
{
    m_current.blend_state = state;
    state->apply();
}

void DeviceContext::setDepthStencilState( DepthStencilState *state )
{
    m_current.depthstencil_state = state;
    state->apply();
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
    glDrawElementsInstancedBaseVertex(topology, num_indices, m_current.index_format, NULL, num_instances, first_vertex);
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

void DeviceContext::clearDepthStencil( RenderTarget *rt, float32 depth, int32 stencil )
{
    rt->bind();
    glClearDepth(depth);
    glClearStencil(stencil);
    glClear(GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
}

} // namespace i3dgl
} // namespace ist
#endif // ist_with_OpenGL
