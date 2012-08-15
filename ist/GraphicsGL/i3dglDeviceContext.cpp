#include "istPCH.h"
#ifdef __ist_with_OpenGL__
#include "ist/Base.h"
#include "i3dglDevice.h"
#include "i3dglDeviceContext.h"

namespace ist {
namespace i3dgl {

DeviceContext::DeviceContext( Device *dev )
    : m_device(dev)
    , m_ia_num_vertex_buffers(0)
    , m_ia_num_vertex_descs(0)
    , m_ia_index_buffer(NULL)
    , m_ia_index_format(I3D_UINT)
    , m_ia_vertex_array(NULL)
    , m_ia_topology(I3D_TRIANGLES)
    , m_shader(NULL)
    , m_vs_shader(NULL)
    , m_ps_shader(NULL)
{
    m_va = m_device->createVertexArray();
    m_shader = m_device->createShaderProgram();
    m_ia_render_target = m_device->createRenderTarget();
}

DeviceContext::~DeviceContext()
{
    istSafeRelease(m_ia_render_target);
    istSafeRelease(m_shader);
    istSafeRelease(m_va);
}

void DeviceContext::setRenderTarget( uint32 num, RenderBuffer **rt, RenderBuffer *depthstencil )
{
    istAssert(num < I3D_MAX_RENDER_TARGETS, "");
    m_ia_num_render_targets = num;
    std::copy(rt, rt+num, m_ia_frame_buffers);
    m_ia_depth_stencil_buffer = depthstencil;
}

void DeviceContext::IAsetVertexBuffers( uint32 num, Buffer **vb, uint32 *strides )
{
    istAssert(num < I3D_MAX_VERTEX_BUFFERS, "");
    m_ia_num_vertex_buffers = num;
    std::copy(vb, vb+num, m_ia_vertex_buffers);
    std::copy(strides, strides+num, m_ia_vertex_strides);
}

void DeviceContext::IAsetInputLayout(uint32 num_descs, const VertexDesc *descs)
{
    m_ia_num_vertex_descs = num_descs;
    std::copy(descs, descs+num_descs, m_ia_vertex_descs);
}

void DeviceContext::IAsetIndexBuffer( Buffer *v, I3D_TYPE format )
{
    m_ia_index_buffer = v;
    m_ia_index_format = format;
}

void DeviceContext::IAsetPrimitiveTopology( I3D_TOPOLOGY v )
{
    m_ia_topology = v;
}

void DeviceContext::VSsetShader( VertexShader *v )
{
    m_vs_shader = v;
}

void DeviceContext::PSsetShader( PixelShader *v )
{
    m_ps_shader = v;
}


void DeviceContext::draw( uint32 num_vertices, uint32 first_vertex )
{
    prepareDraw();
    glDrawArrays(m_ia_topology, first_vertex, num_vertices);
}

void DeviceContext::drawIndexed( uint32 num_indices, uint32 first_vertex )
{
    prepareDraw();
    glDrawElements(m_ia_topology, first_vertex, num_indices, NULL);
}

void DeviceContext::drawInstanced( uint32 num_vertices, uint32 first_vertex, uint32 num_instances )
{
    prepareDraw();
    glDrawArraysInstanced(m_ia_topology, first_vertex, num_vertices, num_instances);
}

void DeviceContext::drawIndexedInstanced( uint32 num_indices, uint32 first_vertex, uint32 num_instances )
{
    prepareDraw();
    glDrawElementsInstancedBaseVertex(m_ia_topology, num_indices, m_ia_index_format, NULL, num_instances, first_vertex);
}

void DeviceContext::prepareDraw()
{
    for(size_t i=0; i<m_ia_num_vertex_descs; ++i) {
        const VertexDesc &desc = m_ia_vertex_descs[i];
        size_t vboi = desc.vbo_index;
        m_ia_vertex_buffers[vboi]->bind();
        m_va->setAttribute(m_ia_vertex_strides[vboi], desc);
    }

    for(size_t i=0; i<m_ia_num_render_targets; ++i) {
        m_ia_render_target->setColorBuffer(i, m_ia_frame_buffers[i]);
    }
    for(size_t i=m_ia_num_render_targets; i<I3D_MAX_RENDER_TARGETS; ++i) {
        m_ia_render_target->setColorBuffer(i, NULL);
    }
    m_ia_render_target->setDepthStencilBuffer(m_ia_depth_stencil_buffer);
    m_ia_render_target->bind();

    m_shader->link(m_vs_shader, NULL, m_ps_shader);
    m_shader->bind();
}

} // namespace i3dgl
} // namespace ist
#endif // __ist_with_OpenGL__
