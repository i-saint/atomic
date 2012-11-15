#include "iuiPCH.h"
#include "iuiRenderer.h"
namespace ist {
namespace iui {


void RCRect::create( RCRect &out, const Rect &shape, const Color &color )
{
    out.ctype = RC_Rect;
    out.shape = shape;
    out.color = color;
}


void RCLine::create( RCLine &out, Line &shape, const Color &color )
{
    out.ctype = RC_Line;
    out.shape = shape;
    out.color = color;
}


void RCText::create( RCText &out, const char *text, size_t textlen, const Color &color, Float size, Float spacing, bool monospace )
{
    out.ctype = RC_Text;
    out.text = text;
    out.textlen = textlen;
    out.color = color;
    out.size = size;
    out.spacing = spacing;
    out.monospace = monospace;
}



UIRenderer::UIRenderer( i3d::Device *dev, i3d::DeviceContext *dc )
    : m_dev(dev)
    , m_dc(dc)
    , m_vbo(NULL)
{
    m_text.reserve(1024*4);
}

UIRenderer::~UIRenderer()
{
}

void UIRenderer::begin()
{

}

void UIRenderer::end()
{
    for(size_t i=0; i<m_commands.size(); /**/) {
        RCType type = *reinterpret_cast<RCType*>(&m_commands[i]);
        switch(type) {
        case RC_Rect:
            {
                draw( *reinterpret_cast<const RCRect*>(&m_commands[i]) );
            }
            i += sizeof(RCRect);
            break;

        case RC_Line:
            {
                draw( *reinterpret_cast<const RCLine*>(&m_commands[i]) );
            }
            i += sizeof(RCLine);
            break;

        case RC_Text:
            {
                draw( *reinterpret_cast<const RCText*>(&m_commands[i]) );
            }
            i += sizeof(RCText);
            break;
        }
    }

    m_commands.clear();
    m_text.clear();
}

void UIRenderer::addCommand( const RCRect &command )
{
    const char *data = reinterpret_cast<const char*>(&command);
    m_commands.insert(m_commands.end(), data, data+sizeof(RCRect));
}

void UIRenderer::addCommand( const RCLine &command )
{
    const char *data = reinterpret_cast<const char*>(&command);
    m_commands.insert(m_commands.end(), data, data+sizeof(RCLine));
}

void UIRenderer::addCommand( const RCText &_command )
{
    // 文字列のコピーを挟む必要がある
    // ここで一旦 RCText::text を m_text[n] の n の値に置き換える
    // (ポインタだと m_text に追加があった時無効になる可能性があるため)

    RCText command = _command;
    command.text = reinterpret_cast<char*>(m_text.size());
    m_text.insert(m_text.end(), _command.text, _command.text+_command.textlen);

    const char *data = reinterpret_cast<const char*>(&command);
    m_commands.insert(m_commands.end(), data, data+sizeof(RCText));
}


void UIRenderer::draw( const RCRect &command )
{

}

void UIRenderer::draw( const RCLine &command )
{

}

void UIRenderer::draw( const RCText &command )
{

}



} // namespace iui
} // namespace ist
