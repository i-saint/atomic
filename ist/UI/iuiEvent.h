﻿#ifndef iui_Event_h
#define iui_Event_h
namespace iui {

using ist::WMT_Unknown;

using ist::WMT_WindowOpen;
using ist::WMT_WindowClose;
using ist::WMT_WindowFocus;
using ist::WMT_WindowDefocus;
using ist::WMT_WindowSize;
using ist::WMT_WindowMove;

using ist::WMT_KeyDown;
using ist::WMT_KeyUp;
using ist::WMT_KeyChar;

using ist::WMT_MouseDown;
using ist::WMT_MouseUp;
using ist::WMT_MouseMove;
using ist::WMT_MouseWheelDown;
using ist::WMT_MouseWheelUp;
using ist::WMT_MouseWheelLeft;
using ist::WMT_MouseWheelRight;

using ist::WMT_IMEBegin;
using ist::WMT_IMEEnd;
using ist::WMT_IMECandidateOpen;
using ist::WMT_IMECandidateClose;
using ist::WMT_IMECandidateChange;
using ist::WMT_IMECursorMove;
using ist::WMT_IMENotify;
using ist::WMT_IMEChar;
using ist::WMT_IMEResult;

enum WindowMessageType
{
    WMT_WidgetCretate = 256,
    WMT_WidgetDelete,
};

typedef ist::WM_Base        WM_Base;
typedef ist::WM_Mouse       WM_Mouse;
typedef ist::WM_Keyboard    WM_Keyboard;
typedef ist::WM_IME         WM_IME;

class Widget;

struct WM_Widget : public WM_Base
{
    Widget *from;
};


} // namespace iui
#endif // iui_Event_h