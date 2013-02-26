#ifndef iui_Renderer_h
#define iui_Renderer_h
#include "iuiCommon.h"

namespace ist {
namespace i3dgl {
    class EasyDrawer;
} // i3dgl
} // namespace ist

namespace iui {



class istInterModule UIRenderer : public ist::SharedObject
{
public:
    virtual ~UIRenderer() {}
    virtual void setScreen(float32 width, float32 height);
    virtual void setScreen(float32 left, float32 right, float32 bottom, float32 top);

    virtual void drawLine(const Line &line, const Color &color)=0;
    virtual void drawRect(const Rect &rect, const Color &color)=0;
    virtual void drawOutlineRect(const Rect &rect, const Color &color)=0;
    //virtual void drawCircle(const Circle &circle, const Color &color)=0;
    //virtual void drawOutlineCircle(const Circle &circle, const Color &color)=0;
    virtual void flush()=0;
};
UIRenderer* CreateUIRenderer();
UIRenderer* CreateUIRenderer(ist::i3dgl::EasyDrawer *drawer);


} // namespace iui
#endif // iui_Renderer_h
