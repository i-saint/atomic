#include "istPCH.h"
#include "i3dglRenderStates.h"
#ifdef ist_with_OpenGL

namespace ist {
namespace i3dgl {

BlendState::BlendState( Device *dev, const BlendStateDesc &desc )
    : super(dev), m_desc(desc)
{
}

void BlendState::apply()
{
    if(m_desc.enable_blend) { glEnable(GL_BLEND); }
    else                    { glDisable(GL_BLEND); }
    for(uint32 i=0; i<sizeof(m_desc.masks); ++i) {
        glColorMaski(i, m_desc.masks[i].red, m_desc.masks[i].green, m_desc.masks[i].blue, m_desc.masks[i].alpha);
    }
    glBlendEquationSeparate(m_desc.equation_rgb, m_desc.equation_a);
    glBlendFuncSeparate(m_desc.func_src_rgb, m_desc.func_dst_rgb, m_desc.func_src_a, m_desc.func_dst_a);
}



DepthStencilState::DepthStencilState( Device *dev, const DepthStencilStateDesc &desc )
    : super(dev)
    , m_desc(desc)
{
}

void DepthStencilState::apply()
{
    if(m_desc.depth_enable) { glEnable(GL_DEPTH_TEST); }
    else                    { glDisable(GL_DEPTH_TEST); }
    if(m_desc.depth_write)  { glDepthMask(GL_TRUE); }
    else                    { glDepthMask(GL_FALSE); }
    glDepthFunc(m_desc.depth_func);

    if(m_desc.stencil_enable)   { glEnable(GL_STENCIL_TEST); }
    else                        { glDisable(GL_STENCIL_TEST); }
    glStencilFunc(m_desc.stencil_func, m_desc.stencil_ref, m_desc.stencil_mask);
    glStencilOp(m_desc.stencil_op_onsfail, m_desc.stencil_op_ondfail, m_desc.stencil_op_onpass);
}


} // namespace i3dgl
} // namespace ist

#endif // ist_with_OpenGL
