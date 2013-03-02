#ifndef iui_Window_h
#define iui_Window_h
#include "iuiCommon.h"
#include "iuiWidget.h"
namespace iui {


class RootWindowStyle : public Style
{
public:
    virtual void draw();
};

class RootWindow : public Widget
{
public:
    iuiImplWidget(RootWindow)
private:
};



class PanelStyle : public Style
{
public:
    virtual void draw();
};

class Panel : public Widget
{
public:
    iuiImplWidget(Panel)
private:
};



class WindowStyle : public Style
{
public:
    virtual void draw();
};

class Window : public Widget
{
public:
    iuiImplWidget(Window)
private:
};


} // namespace iui
#endif // iui_Window_h
