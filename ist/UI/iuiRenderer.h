#ifndef iui_Renderer_h
#define iui_Renderer_h
#include "iuiCommon.h"

namespace ist {
namespace i3dgl {
    class EasyDrawer;
    class IFontRenderer;
} // i3dgl
namespace i3dgles {
    class EasyDrawer;
    class IFontRenderer;
} // i3dgles
} // namespace ist

namespace iui {

struct TextPosition
{
    Rect rect;
    TextHAlign halign;
    TextVAlign valign;
    Float hspace;
    Float vspace;
    TextPosition() : halign(TA_HLeft), valign(TA_VTop), hspace(0.75f), vspace(1.0f) {}
    TextPosition(const Rect &rc, TextHAlign ha=TA_HLeft, TextVAlign va=TA_VTop, Float hs=0.75f, Float vs=1.0f)
        : rect(rc), halign(ha), valign(va), hspace(hs), vspace(vs) {}
};

class iuiInterModule UIRenderer
{
public:
    virtual void release()=0;
    virtual void initialize(ist::i3dgl::EasyDrawer *drawer=NULL, ist::i3dgl::IFontRenderer *font=NULL)=0;
    virtual void finalize()=0;

    virtual void setScreen(float32 width, float32 height)=0;
    virtual void setTranslate(Position pos)=0;

    virtual void drawLine(const Line &line, const Color &color)=0;
    virtual void drawRect(const Rect &rect, const Color &color)=0;
    virtual void drawOutlineRect(const Rect &rect, const Color &color)=0;
    virtual void drawFont(const TextPosition &pos, const Color &color, const wchar_t *text, uint32 len=0)=0;

    virtual void begin()=0;
    virtual void end()=0;
    virtual void flush()=0;

protected:
    virtual ~UIRenderer() {}
};
UIRenderer* CreateUIRenderer();

} // namespace iui
#endif // iui_Renderer_h
