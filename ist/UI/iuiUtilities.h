#ifndef iui_Utilities_h
#define iui_Utilities_h
#include "iuiCommon.h"
#include "iuiEvent.h"

namespace iui {

iuiInterModule bool IsInside( const Rect &rect, const Position &pos );
iuiInterModule bool IsInside( const Circle &circle, const Position &pos );

enum WidgetHit {
    WH_Nothing,
    WH_HitMouseLeftDown,
    WH_HitMouseRightDown,
    WH_HitMouseMiddleDown,
    WH_HitMouseLeftUp,
    WH_HitMouseRightUp,
    WH_HitMouseMiddleUp,
    WH_MissMouseLeftDown,
    WH_MissMouseRightDown,
    WH_MissMouseMiddleDown,
    WH_MissMouseLeftUp,
    WH_MissMouseRightUp,
    WH_MissMouseMiddleUp,
    WH_MouseEnter,
    WH_MouseLeave,
};
iuiInterModule WidgetHit MouseHitWidget(Widget *w, const WM_Base &wm);

} // namespace iui
#endif // iui_Utilities_h
