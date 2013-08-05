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
        : m_drawer(nullptr)
    {
    }

    virtual ~UIRendererImpl()
    {
    }

    void release() override
    {
        iuiDelete(this);
    }

    void initialize(ist::i3dgl::EasyDrawer *drawer=nullptr, ist::i3dgl::IFontRenderer *font=nullptr) override
    {
        m_drawer = drawer ? drawer : i3d::CreateEasyDrawer();
        m_font = font;
        istSafeAddRef(drawer);
        istSafeAddRef(font);
    }

    void finalize() override
    {
        istSafeRelease(m_font);
        istSafeRelease(m_drawer);
    }

    void setViewport(int32 x, int32 y, int32 width, int32 height) override
    {
        m_font->setViewport(x,y,width,height);
        m_drawer->setViewport(x,y,width,height);
    }

    void setScreen(float32 x, float32 y, float32 width, float32 height) override
    {
        m_font->setScreen(x,x+width, y+height,y);
        m_drawer->setScreen(x,x+width, y+height,y);
    }
    void setTranslate(Position pos) override
    {
        mat4 trans = glm::translate(mat4(), vec3(pos, 0.0f));
        m_drawer->setWorldMatrix(trans);
    }


    void drawLine(const Line &line, const Color &color) override
    {
        if(color.a<=0.0f) { return; }
        i3d::DrawLine(*m_drawer,line.begin, line.end, color);
    }

    void drawRect(const Rect &rect, const Color &color) override
    {
        if(color.a<=0.0f) { return; }
        i3d::DrawRectT(*m_drawer, rect.pos, rect.pos+rect.size, vec2(1.0f), vec2(0.0f), color);
    }

    void drawOutlineRect(const Rect &rect, const Color &color) override
    {
        if(color.a<=0.0f) { return; }
        i3d::DrawOutlineRect(*m_drawer, rect.pos, rect.pos+rect.size, color);
    }

    void drawFont(const TextPosition &tp, const Color &color, const wchar_t *text, uint32 len) override
    {
        if(m_font==nullptr) { return; }
        vec2 pos  = tp.rect.getPosition();
        vec2 size = tp.rect.getSize();
        m_font->setSpacing(tp.hspace);
        m_font->setColor(color);

        vec2 fs   = m_font->computeTextSize(text, len);
        if     (tp.halign==TA_HRight)  { pos.x += size.x-fs.x; }
        else if(tp.halign==TA_HCenter) { pos.x += size.x*0.5f - fs.x*0.5f; }
        if     (tp.valign==TA_VBottom) { pos.y += size.y-fs.y; }
        else if(tp.valign==TA_VCenter) { pos.y += size.y*0.5f - fs.y*0.5f; }
        m_font->addText(pos, text, len);
        m_font->draw();
    }

    void begin() override
    {
        m_prev_state = m_drawer->getRenderStates();
        if(m_font) {
            m_font->setSize(20.0f);
            m_font->setMonospace(true);
            m_font->setSpacing(1.0f);
        }
    }

    void end() override
    {
        m_drawer->forceSetRenderStates(m_prev_state);
    }

    void flush() override
    {
        m_drawer->flush(i3d::GetDevice()->getImmediateContext());
    }


    Size computeTextSize( const wchar_t *text, uint32 len/*=0*/ ) override
    {
        if(m_font==nullptr) { return Size(); }
        return Size(m_font->computeTextSize(text,len));
    }


    ist::i3dgl::EasyDrawer*     getDrawer() const override { return m_drawer; }
    ist::i3dgl::IFontRenderer*  getFont() const override  { return m_font; }

private:
    i3d::EasyDrawer *m_drawer;
    i3d::IFontRenderer *m_font;
    i3d::EasyDrawState m_state;
    i3d::EasyDrawState m_prev_state;
};

UIRenderer* CreateUIRenderer()
{
    return iuiNew(UIRendererImpl)();
}



} // namespace iui
