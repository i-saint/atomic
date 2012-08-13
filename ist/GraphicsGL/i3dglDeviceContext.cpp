#include "stdafx.h"
#include "../Base.h"
#include "i3dglDevice.h"
#include "i3dglDeviceContext.h"

namespace ist {
namespace i3dgl {

DeviceContext::DeviceContext( Device *dev )
    : m_device(dev)
    , m_ia_index_type(I3D_UINT)
{

}

DeviceContext::~DeviceContext()
{

}

void DeviceContext::setRenderTarget( RenderBuffer **rt, uint32 num, RenderBuffer *depthstencil )
{

}

void DeviceContext::IAsetVertexBuffer( Buffer **vb, uint32 *strides, uint32 num )
{

}

void DeviceContext::IAsetInputLayout()
{

}

void DeviceContext::IAsetIndexBuffer( Buffer *v, I3D_TYPE format )
{

}

void DeviceContext::VSsetVertexShader( VertexShader *v )
{

}

void DeviceContext::PSsetPixelShader( PixelShader *v )
{

}


void DeviceContext::draw( uint32 num_vertices )
{
    glDrawArrays(m_ia_topology, 0, num_vertices);
}

void DeviceContext::drawIndexed( uint32 num_indices )
{
    glDrawElements(m_ia_topology, 0, num_indices, NULL);
}

void DeviceContext::drawInstanced( uint32 num_vertices, uint32 num_instances )
{
    glDrawArraysInstanced(m_ia_topology, 0, num_vertices, num_instances);
}

void DeviceContext::drawIndexedInstanced( uint32 num_indices, uint32 num_instances )
{
    glDrawElementsInstanced(m_ia_topology, num_indices, m_ia_index_type, NULL, num_instances);
}

} // namespace i3dgl
} // namespace ist
