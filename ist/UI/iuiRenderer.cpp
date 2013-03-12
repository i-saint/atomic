#include "iuiPCH.h"
#include "iuiRenderer.h"
#include "ist/GraphicsCommon/EasyDrawer.h"

namespace iui {
// とりあえず
namespace i3d = ist::i3dgl;

class UIRendererImpl : public UIRenderer
{
public:
    UIRendererImpl()
        : m_drawer(NULL)
    {
    }

    virtual ~UIRendererImpl()
    {
    }

    virtual void release()
    {
        istDelete(this);
    }

    virtual void initialize(ist::i3dgl::EasyDrawer *drawer=NULL, ist::i3dgl::IFontRenderer *font=NULL)
    {
        m_drawer = drawer ? drawer : i3d::CreateEasyDrawer();
        m_font = font;
        istSafeAddRef(drawer);
        istSafeAddRef(font);
    }

    virtual void finalize()
    {
        istSafeRelease(m_font);
        istSafeRelease(m_drawer);
    }

    virtual void setScreen(float32 width, float32 height)
    {
        m_drawer->setScreen(width, height);
    }


    virtual void drawLine(const Line &line, const Color &color)
    {
        if(color.a<=0.0f) { return; }
        i3d::DrawLine(*m_drawer,line.begin, line.end, color);
    }

    virtual void drawRect(const Rect &rect, const Color &color)
    {
        if(color.a<=0.0f) { return; }
        i3d::DrawRectT(*m_drawer, rect.pos, rect.pos+rect.size, vec2(1.0f), vec2(0.0f), color);
    }

    virtual void drawOutlineRect(const Rect &rect, const Color &color)
    {
        if(color.a<=0.0f) { return; }
        i3d::DrawOutlineRect(*m_drawer, rect.pos, rect.pos+rect.size, color);
    }

    virtual void drawFont(const TextPosition &tp, const Color &color, const wchar_t *text, uint32 len)
    {
        if(m_font==NULL) { return; }
        vec2 pos  = tp.rect.getPosition();
        vec2 size = tp.rect.getSize();
        vec2 fs   = m_font->computeTextSize(text, len);
        if     (tp.halign==TA_HRight)  { pos.x += size.x-fs.x; }
        else if(tp.halign==TA_HCenter) { pos.x += size.x*0.5f - fs.x*0.5f; }
        if     (tp.valign==TA_VBottom) { pos.y += size.y-fs.y; }
        else if(tp.valign==TA_VCenter) { pos.y += size.y*0.5f - fs.y*0.5f; }
        m_font->setSpacing(tp.hspace);
        m_font->setColor(color);
        m_font->addText(pos, text, len);
    }

    virtual void begin()
    {
        m_prev_state = m_drawer->getRenderStates();
        if(m_font) {
            m_font->setSize(20.0f);
            m_font->setMonospace(true);
            m_font->setSpacing(1.0f);
        }
    }

    virtual void end()
    {
        m_drawer->forceSetRenderStates(m_prev_state);
    }

    virtual void flush()
    {
        m_drawer->flush(i3d::GetDevice()->getImmediateContext());
    }

private:
    i3d::EasyDrawer *m_drawer;
    i3d::IFontRenderer *m_font;
    i3d::EasyDrawState m_state;
    i3d::EasyDrawState m_prev_state;
};

UIRenderer* CreateUIRenderer()
{
    return istNew(UIRendererImpl)();
}



} // namespace iui
