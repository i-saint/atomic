#ifndef iui_Utilities_h
#define iui_Utilities_h
#include "iuiCommon.h"
#include "iuiEvent.h"

namespace iui {

iuiAPI bool IsInside( const Rect &rect, const Position &pos );
iuiAPI bool IsInside( const Circle &circle, const Position &pos );
iuiAPI bool IsOverlaped( const Rect &r1, const Rect &r2 );

enum WidgetHit {
    WH_Nothing,
    WH_HitMouseLeftDown,
    WH_HitMouseRightDown,
    WH_HitMouseMiddleDown,
    WH_HitMouseLeftUp,
    WH_HitMouseRightUp,
    WH_HitMouseMiddleUp,
    WH_HitMouseWheelUp,
    WH_HitMouseWheelDown,
    WH_MissMouseLeftDown,
    WH_MissMouseRightDown,
    WH_MissMouseMiddleDown,
    WH_MissMouseLeftUp,
    WH_MissMouseRightUp,
    WH_MissMouseMiddleUp,
    WH_MissMouseWheelUp,
    WH_MissMouseWheelDown,
    WH_MouseInside,
    WH_MouseOutside,
};
iuiAPI WidgetHit MouseHit(const Rect &rect, const WM_Base &wm);
iuiAPI WidgetHit MouseHit(Widget *w, const WM_Base &wm);

iuiAPI void HandleMouseHover(const Rect &rect, bool &hovered);
iuiAPI void HandleMouseHover(Widget *w, bool &hovered);

iuiAPI void SetupScreen( const Rect &rect );
iuiAPI void SetupScreen(Widget *w);

} // namespace iui
#endif // iui_Utilities_h
