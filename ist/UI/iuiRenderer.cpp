#include "iuiPCH.h"
#include "iuiRenderer.h"
#include "ist/GraphicsCommon/EasyDrawer.h"

namespace ist {
namespace iui {
// とりあえず
namespace i3d = i3dgl;

class UIRendererGL : public UIRenderer
{
public:
    UIRendererGL() {}
    virtual ~UIRendererGL() {}

    virtual void drawPoint(const Position &point)
    {

    }

    virtual void drawLine(const Line &line)
    {

    }

    virtual void drawRect(const Rect &rect)
    {

    }

    virtual void drawOutlineRect(const Rect &rect)
    {

    }

    //virtual void drawCircle(const Circle &circle)=0;
    //virtual void drawOutlineCircle(const Circle &circle)=0;

    virtual void flush()
    {

    }

private:
    i3d::EasyDrawer *m_drawer;
};

UIRenderer* CreateUIRenderer()
{
    return istNew(UIRendererGL)();
}


} // namespace iui
} // namespace ist
