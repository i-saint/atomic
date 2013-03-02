#ifndef iui_Types_h
#define iui_Types_h

namespace iui {

enum WidgetTypeID
{
    WT_Unknown,
    WT_Widget,
    WT_RootWindow,
    WT_Panel,
    WT_Window,
    WT_Button,
    WT_ToggleButton,
    WT_Checkbox,
    WT_Editbox,
    WT_EditboxMultiline,
    WT_Listbox,
    WT_Combobox,
    WT_VSlider,
    WT_HSlider,
    WT_VScrollbar,
    WT_HScrollbar,
    WT_End,
};

class Widget;
class RootWindow;
class Panel;
class Window;
class Button;
class ToggleButton;
class Checkbox;
class Editbox;
class EditboxMultiline;
class Listbox;
class Combobox;
class VSlider;
class HSlider;
class VScrollbar;
class HScrollbar;

template<class T> struct GetWidgetTypeID            { static const WidgetTypeID result=WT_Unknown; };
template<> struct GetWidgetTypeID<Widget>           { static const WidgetTypeID result=WT_Widget; };
template<> struct GetWidgetTypeID<RootWindow>       { static const WidgetTypeID result=WT_RootWindow; };
template<> struct GetWidgetTypeID<Panel>            { static const WidgetTypeID result=WT_Panel; };
template<> struct GetWidgetTypeID<Window>           { static const WidgetTypeID result=WT_Window; };
template<> struct GetWidgetTypeID<Button>           { static const WidgetTypeID result=WT_Button; };
template<> struct GetWidgetTypeID<ToggleButton>     { static const WidgetTypeID result=WT_ToggleButton; };
template<> struct GetWidgetTypeID<Checkbox>         { static const WidgetTypeID result=WT_Checkbox; };
template<> struct GetWidgetTypeID<Editbox>          { static const WidgetTypeID result=WT_Editbox; };
template<> struct GetWidgetTypeID<EditboxMultiline> { static const WidgetTypeID result=WT_EditboxMultiline; };
template<> struct GetWidgetTypeID<Listbox>          { static const WidgetTypeID result=WT_Listbox; };
template<> struct GetWidgetTypeID<Combobox>         { static const WidgetTypeID result=WT_Combobox; };
template<> struct GetWidgetTypeID<VSlider>          { static const WidgetTypeID result=WT_VSlider; };
template<> struct GetWidgetTypeID<HSlider>          { static const WidgetTypeID result=WT_HSlider; };
template<> struct GetWidgetTypeID<VScrollbar>       { static const WidgetTypeID result=WT_VScrollbar; };
template<> struct GetWidgetTypeID<HScrollbar>       { static const WidgetTypeID result=WT_HScrollbar; };

} // namespace iui
#endif // iui_Types_h
