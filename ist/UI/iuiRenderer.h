#ifndef ist_UI_iuiRenderer_h
#define ist_UI_iuiRenderer_h
#include "iuiCommon.h"

namespace ist {
namespace iui {



class istInterModule UIRenderer : public SharedObject
{
public:
    virtual ~UIRenderer() {}
    virtual void drawPoint(const Position &point)=0;
    virtual void drawLine(const Line &line)=0;
    virtual void drawRect(const Rect &rect)=0;
    virtual void drawOutlineRect(const Rect &rect)=0;
    //virtual void drawCircle(const Circle &circle)=0;
    //virtual void drawOutlineCircle(const Circle &circle)=0;
    virtual void flush()=0;
};
UIRenderer* CreateUIRenderer();


} // namespace iui
} // namespace ist
#endif // ist_UI_iuiRenderer_h
