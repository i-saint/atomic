#include "iuiPCH.h"
#include "iuiRenderer.h"
namespace ist {
namespace iui {

IRenderCommand::IRenderCommand()
{
}

IRenderCommand::~IRenderCommand()
{
}

RCRect::RCRect( const Rect &s, const Color &c, FillMode f )
    : shape(s), color(c), fillmode(f)
{}

RCRect* RCRect::create( const Rect &s, const Color &c, FillMode f )
{
     return istNew(RCRect)(s, c, f);
}



UIRenderer::UIRenderer( i3d::Device *dev, i3d::DeviceContext *dc )
    : m_dev(dev)
    , m_dc(dc)
    , m_vbo(NULL)
{
}

UIRenderer::~UIRenderer()
{
    releaseAllCommands();
}

void UIRenderer::addCommand( IRenderCommand *command )
{
    m_commands.push_back(command);
}

void UIRenderer::flush()
{

}

void UIRenderer::releaseAllCommands()
{
    for(size_t i=0; i<m_commands.size(); ++i) {
        istDelete(m_commands[i]);
    }
    m_commands.clear();
}


} // namespace iui
} // namespace ist
