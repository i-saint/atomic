#ifndef iui_Renderer_h
#define iui_Renderer_h
#include "iuiCommon.h"

namespace ist {
namespace i3dgl {
    class EasyDrawer;
} // i3dgl
namespace i3dgles {
    class EasyDrawer;
} // i3dgles
} // namespace ist

namespace iui {

struct TextPosition
{
    Rect rect;
    TextHAlign halign;
    TextVAlign valign;
    TextPosition() : halign(TA_HLeft), valign(TA_VTop) {}
    TextPosition(const Position &pos, const Size &size=Size(), TextHAlign ha=TA_HLeft, TextVAlign va=TA_VTop) : halign(ha), valign(va), rect(pos, size) {}
};

class iuiInterModule UIRenderer
{
public:
    virtual void release()=0;
    virtual void initialize(ist::i3dgl::EasyDrawer *drawer=NULL)=0;
    virtual void finalize()=0;

    virtual void setScreen(float32 width, float32 height)=0;
    virtual void setScreen(float32 left, float32 right, float32 bottom, float32 top)=0;

    virtual void drawLine(const Line &line, const Color &color)=0;
    virtual void drawRect(const Rect &rect, const Color &color)=0;
    virtual void drawOutlineRect(const Rect &rect, const Color &color)=0;
    //virtual void drawCircle(const Circle &circle, const Color &color)=0;
    //virtual void drawOutlineCircle(const Circle &circle, const Color &color)=0;
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
