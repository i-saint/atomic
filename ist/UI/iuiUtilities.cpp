#include "iuiPCH.h"
#include "iuiUtilities.h"
#include "iuiWidget.h"
#include "iuiSystem.h"

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


iuiInterModule WidgetHit MouseHitWidget(Widget *w, const WM_Base &wm)
{
    const Rect rect(w->getPositionAbs(), w->getSize());
    switch(wm.type) {
    case WMT_MouseDown: // 
    case WMT_MouseUp:   // 
    case WMT_MouseMove: // fall through
        auto &mes = WM_Mouse::cast(wm);
        if(IsInside(rect, mes.mouse_pos)) {
            switch(wm.type) {
            case WMT_MouseDown:
                if(mes.button.left)   return WH_HitMouseLeftDown;
                if(mes.button.right)  return WH_HitMouseRightDown;
                if(mes.button.middle) return WH_HitMouseMiddleDown;
            case WMT_MouseUp:
                if(mes.button.left)   return WH_HitMouseLeftUp;
                if(mes.button.right)  return WH_HitMouseRightUp;
                if(mes.button.middle) return WH_HitMouseMiddleUp;
            case WMT_MouseMove:
                return WH_MouseInside;
            }
        }
        else {
            switch(wm.type) {
            case WMT_MouseDown:
                if(mes.button.left)   return WH_MissMouseLeftDown;
                if(mes.button.right)  return WH_MissMouseRightDown;
                if(mes.button.middle) return WH_MissMouseMiddleDown;
            case WMT_MouseUp:
                if(mes.button.left)   return WH_MissMouseLeftUp;
                if(mes.button.right)  return WH_MissMouseRightUp;
                if(mes.button.middle) return WH_MissMouseMiddleUp;
            case WMT_MouseMove:
                return WH_MouseOutside;
            }
        }
        break;
    }
    return WH_Nothing;
}

iuiInterModule void HandleMouseHover(Widget *w, bool &hovered)
{
    const Rect rect(w->getPositionAbs(), w->getSize());
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

} // namespace iui
