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

    virtual void initialize(i3d::EasyDrawer *drawer=NULL)
    {
        m_drawer = drawer ? drawer : i3d::CreateEasyDrawer();
        istSafeAddRef(drawer);
    }

    virtual void finalize()
    {
        istSafeRelease(m_drawer);
    }

    virtual void setScreen(float32 width, float32 height)
    {
        m_state.setScreen(width, height);
    }

    virtual void setScreen(float32 left, float32 right, float32 bottom, float32 top)
    {
        m_state.setScreen(left, right, bottom, top);
    }


    virtual void drawLine(const Line &line, const Color &color)
    {
        if(color.a<=0.0f) { return; }
        i3d::DrawLine(*m_drawer,line.begin, line.end, color);
    }

    virtual void drawRect(const Rect &rect, const Color &color)
    {
        if(color.a<=0.0f) { return; }
        i3d::DrawRectT(*m_drawer, rect.pos+rect.size, rect.pos, vec2(1.0f), vec2(0.0f), color);
    }

    virtual void drawOutlineRect(const Rect &rect, const Color &color)
    {
        if(color.a<=0.0f) { return; }
        i3d::DrawOutlineRect(*m_drawer, rect.pos+rect.size, rect.pos, color);
    }

    //virtual void drawCircle(const Circle &circle, const Color &color)=0;
    //virtual void drawOutlineCircle(const Circle &circle, const Color &color)=0;

    virtual void drawFont(const TextPosition &pos, const Color &color, const wchar_t *text, uint32 len)
    {

    }

    virtual void begin()
    {
        m_prev_state = m_drawer->getRenderStates();
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
    i3d::EasyDrawState m_state;
    i3d::EasyDrawState m_prev_state;
};

UIRenderer* CreateUIRenderer()
{
    return istNew(UIRendererImpl)();
}



} // namespace iui
