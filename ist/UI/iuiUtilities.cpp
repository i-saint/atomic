#include "iuiPCH.h"
#include "iuiUtilities.h"
#include "iuiWidget.h"
#include "iuiSystem.h"
#include "iuiRenderer.h"

namespace iui {

iuiInterModule bool IsInside( const Rect &rect, const Position &pos )
{
    Position rel = pos - rect.getPosition();
    const Size &size = rect.getSize();
    return (rel.x>=0 && rel.x<=size.x) && (rel.y>=0 && rel.y<=size.y);
}

iuiInterModule bool IsInside( const Circle &circle, const Position &pos )
{
    Float r = circle.getRadius();
    vec2 d = circle.getPosition() - pos;
    return glm::dot(d,d) < r*r;
}

iuiInterModule bool IsOverlaped( const Rect &r1, const Rect &r2 )
{
    Position rel = r1.getPosition() - r2.getPosition();
    Size size = r1.getSize();
    if( (r2.size.x < rel.x || 0.0f > rel.x+size.x) ||
        (r2.size.y < rel.y || 0.0f > rel.y+size.y) )
    {
        return false;
    }
    return true;
}


iuiInterModule WidgetHit MouseHit(const Rect &rect, const WM_Base &wm)
{
    switch(wm.type) {
    case WMT_MouseDown:     // 
    case WMT_MouseUp:       // 
    case WMT_MouseMove:     // 
    case WMT_MouseWheelDown:// 
    case WMT_MouseWheelUp:  // fall through
        auto &mes = WM_Mouse::cast(wm);
        if(IsInside(rect, mes.mouse_pos)) {
            switch(wm.type) {
            case WMT_MouseDown:
                if(mes.button.left)     return WH_HitMouseLeftDown;
                if(mes.button.right)    return WH_HitMouseRightDown;
                if(mes.button.middle)   return WH_HitMouseMiddleDown;
            case WMT_MouseUp:
                if(mes.button.left)     return WH_HitMouseLeftUp;
                if(mes.button.right)    return WH_HitMouseRightUp;
                if(mes.button.middle)   return WH_HitMouseMiddleUp;
            case WMT_MouseWheelDown:    return WH_HitMouseWheelDown;
            case WMT_MouseWheelUp:      return WH_HitMouseWheelUp;
            case WMT_MouseMove:         return WH_MouseInside;
            }
        }
        else {
            switch(wm.type) {
            case WMT_MouseDown:
                if(mes.button.left)     return WH_MissMouseLeftDown;
                if(mes.button.right)    return WH_MissMouseRightDown;
                if(mes.button.middle)   return WH_MissMouseMiddleDown;
            case WMT_MouseUp:
                if(mes.button.left)     return WH_MissMouseLeftUp;
                if(mes.button.right)    return WH_MissMouseRightUp;
                if(mes.button.middle)   return WH_MissMouseMiddleUp;
            case WMT_MouseWheelDown:    return WH_MissMouseWheelDown;
            case WMT_MouseWheelUp:      return WH_MissMouseWheelUp;
            case WMT_MouseMove:         return WH_MouseOutside;
            }
        }
        break;
    }
    return WH_Nothing;
}

iuiInterModule WidgetHit MouseHit(Widget *w, const WM_Base &wm)
{
    return MouseHit(Rect(w->getPositionAbs(), w->getSize()), wm);
}


iuiInterModule void HandleMouseHover(const Rect &rect, bool &hovered)
{
    if(IsInside(rect, iuiGetMousePos())) {
        if(!hovered) {
            hovered = true;
        }
    }
    else {
        if(hovered) {
            hovered = false;
        }
    }
}

iuiInterModule void HandleMouseHover(Widget *w, bool &hovered)
{
    return HandleMouseHover(Rect(w->getPositionAbs(), w->getSize()), hovered);
}


iuiInterModule void SetupScreen( const Rect &rect )
{
    const Position &pos = rect.getPosition();
    const Size &size    = rect.getSize();
    const Rect &screen  = iuiGetSystem()->getScreen();
    const Size viewport = Size(istGetAplication()->getWindowSize().x, istGetAplication()->getWindowSize().y);
    Size r = viewport/screen.size;
    iuiGetRenderer()->setViewport( (int32)(pos.x*r.x-0.5f), (int32)(viewport.y-(pos.y+size.y)*r.y-0.5f), (int32)(size.x*r.x+1.0f), (int32)(size.y*r.y+1.0f) );
    iuiGetRenderer()->setScreen(-0.5f, -0.5f, size.x+1.0f, size.y+1.0f);
}

iuiInterModule void SetupScreen( Widget *w )
{
    SetupScreen(Rect(w->getPositionAbs(), w->getSize()));
}

} // namespace iui
