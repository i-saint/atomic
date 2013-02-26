#include "istPCH.h"
#include "ist/Base.h"
#ifdef ist_with_OpenGL
#include "i3dglDevice.h"
#include "i3dglDeviceContext.h"

namespace ist {
namespace i3dgl {

DeviceContext::DeviceContext( Device *dev )
    : m_device(dev)
{
    istSafeAddRef(m_device);

    m_dirty.flags = 0;
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
    m_dirty.vertex_array = 1;
    m_current.vertex_array = va;
}

void DeviceContext::setIndexBuffer( Buffer *v, size_t offset, I3D_TYPE format )
{
    m_dirty.index = 1;
    m_current.index.buffer = v;
    m_current.index.offset = offset;
    m_current.index.format = format;
}

void DeviceContext::setUniformBuffer( int32 loc, int32 bind, Buffer *buf )
{
    istAssert(loc<_countof(m_current.uniform));
    m_dirty.uniform = 1;
    m_current.uniform[loc].buffer = buf;
    m_current.uniform[loc].bind = bind;
    m_current.uniform[loc].dirty = true;
}

void DeviceContext::setShader( ShaderProgram *v )
{
    m_dirty.shader = 1;
    m_current.shader = v;
}

void DeviceContext::setRenderTarget( RenderTarget *rt )
{
    m_dirty.render_target = 1;
    m_current.render_target = rt;
}

void DeviceContext::setSampler( uint32 slot, Sampler *smp )
{
    m_dirty.samplers = 1;
    m_current.samplers[slot] = smp;
}

void DeviceContext::setTexture( uint32 slot, Texture *tex )
{
    istAssert(slot<_countof(m_current.textures));
    m_dirty.textures = 1;
    m_current.textures[slot] = tex;
}

void DeviceContext::setBlendState( BlendState *state )
{
    m_dirty.blend_state = 1;
    m_current.blend_state = state;
}

void DeviceContext::setDepthStencilState( DepthStencilState *state )
{
    m_dirty.depthstencil_state = 1;
    m_current.depthstencil_state = state;
}


void DeviceContext::draw( I3D_TOPOLOGY topology, uint32 first_vertex, uint32 num_vertices )
{
    applyRenderStates();
    glDrawArrays(topology, first_vertex, num_vertices);
}

void DeviceContext::drawIndexed( I3D_TOPOLOGY topology, uint32 num_indices )
{
    applyRenderStates();
    glDrawElements(topology, num_indices, m_current.index.format, (void*)m_current.index.offset);
}

void DeviceContext::drawInstanced( I3D_TOPOLOGY topology, uint32 first_vertex, uint32 num_vertices, uint32 num_instances )
{
    applyRenderStates();
    glDrawArraysInstanced(topology, first_vertex, num_vertices, num_instances);
}

void DeviceContext::drawIndexedInstanced( I3D_TOPOLOGY topology, uint32 first_vertex, uint32 num_indices, uint32 num_instances )
{
    applyRenderStates();
    glDrawElementsInstancedBaseVertex(topology, num_indices, m_current.index.format, (void*)m_current.index.offset, num_instances, first_vertex);
}

void* DeviceContext::map(Buffer *buffer, I3D_MAP_MODE mode)
{
    if(buffer) {
        applyRenderStates();
        return buffer->map(mode);
    }
    return NULL;
}

void DeviceContext::unmap(Buffer *buffer)
{
    if(buffer) {
        applyRenderStates();
        buffer->unmap();
    }
}

void DeviceContext::updateResource(Texture1D *tex, uint32 mip, uint32 pos, uint32 size, void *data)
{
    if(tex) {
        applyRenderStates();
        tex->update(mip, pos, size, data);
    }
}

void DeviceContext::updateResource(Texture2D *tex, uint32 mip, uvec2 pos, uvec2 size, void *data)
{
    if(tex) {
        applyRenderStates();
        tex->update(mip, pos, size, data);
    }
}

void DeviceContext::updateResource(Texture3D *tex, uint32 mip, uvec3 pos, uvec3 size, void *data)
{
    if(tex) {
        applyRenderStates();
        tex->update(mip, pos, size, data);
    }
}

void DeviceContext::generateMips( Texture *tex )
{
    if(tex) {
        applyRenderStates();
        tex->generateMips();
    }
}

void DeviceContext::clearColor( RenderTarget *rt, vec4 color )
{
    applyRenderStates();
    rt->bind();
    glClearColor(color.x, color.y, color.z, color.w);
    glClear(GL_COLOR_BUFFER_BIT);

    if(m_current.render_target) { m_current.render_target->bind(); }
    else { m_current.render_target->unbind(); }

}

void DeviceContext::clearDepthStencil( RenderTarget *rt, float32 depth, int32 stencil )
{
    applyRenderStates();
    rt->bind();
    glClearDepth(depth);
    glClearStencil(stencil);
    glClear(GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

    if(m_current.render_target) { m_current.render_target->bind(); }
    else { m_current.render_target->unbind(); }
}


// template 関数だと private メソッドにアクセスできないのでマクロで…
#define IsDirty(Obj) (Obj && Obj->isDirty())

void DeviceContext::applyRenderStates()
{
    if(m_dirty.flags==0) { return; }

    if( m_dirty.shader && 
        m_current.shader!=m_prev.shader)
    {
        if(m_current.shader != NULL) {
            m_current.shader->bind();
        }
        else {
            glUseProgram(0);
        }
    }

    // render target 更新と被らないよう、先にテクスチャの解除だけ行う
    if( m_dirty.textures &&
        (m_dirty.render_target || IsDirty(m_current.render_target)) )
    {
        for(size_t i=0; i<_countof(m_current.textures); ++i) {
            glActiveTexture(GL_TEXTURE0+i);
            glBindTexture(GL_TEXTURE_2D, 0);
            m_prev.textures[i] = NULL;
        }
    }

    if( m_dirty.render_target || IsDirty(m_current.render_target) )
    {
        if(m_current.render_target != NULL) {
            m_current.render_target->bind();
        }
        else {
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
        }
    }

    if( m_dirty.vertex_array || IsDirty(m_current.vertex_array) )
    {
        if(m_current.vertex_array) {
            m_current.vertex_array->bind();
        }
        else {
            glBindVertexArray(0);
        }
    }

    if( m_dirty.index )
    {
        if(m_current.index.buffer != NULL) {
            m_current.index.buffer->bind();
        }
        else {
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
        }
    }

    if( m_dirty.uniform ) {
        for(size_t i=0; i<_countof(m_current.uniform); ++i) {
            if(!m_current.uniform[i].dirty) { continue; }
            m_current.uniform[i].dirty = false;
            if(m_current.uniform[i].buffer!=NULL) {
                m_current.shader->setUniformBlock(i, m_current.uniform[i].bind, m_current.uniform[i].buffer->getHandle());
            }
            else {
                m_current.shader->setUniformBlock(i, m_current.uniform[i].bind, 0);
            }
        }
    }

    if( m_dirty.samplers ) {
        for(size_t i=0; i<_countof(m_current.samplers); ++i) {
            if(m_prev.samplers[i]==m_current.samplers[i]) { continue; }
            if(m_current.samplers[i]!=NULL) {
                m_current.samplers[i]->bind(i);
            }
            else {
                glBindSampler(i, 0);
            }
        }
    }

    if( m_dirty.textures ) {
        for(size_t i=0; i<_countof(m_current.textures); ++i) {
            if(m_prev.textures[i]==m_current.textures[i]) { continue; }
            if(m_current.textures[i]!=NULL) {
                m_current.textures[i]->bind(i);
            }
            else {
                glActiveTexture(GL_TEXTURE0+i);
                glBindTexture(GL_TEXTURE_2D, 0);
            }
        }
    }

    if( m_dirty.blend_state && m_current.blend_state!=m_prev.blend_state )
    {
        m_current.blend_state->apply();
    }

    if(m_dirty.depthstencil_state && m_current.depthstencil_state!=m_prev.depthstencil_state )
    {
        m_current.depthstencil_state->apply();
    }

    m_prev = m_current;
    m_dirty.flags = 0;
}

} // namespace i3dgl
} // namespace ist
#endif // ist_with_OpenGL
