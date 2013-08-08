#ifndef iui_Window_h
#define iui_Window_h
#include "iuiCommon.h"
#include "iuiWidget.h"
namespace iui {


class RootWindowStyle : public Style
{
public:
    void draw() override;
};

class RootWindow : public Widget
{
public:
    iuiImplWidget(RootWindow)
    RootWindow();
private:
};



class PanelStyle : public Style
{
public:
    void draw() override;
};

class Panel : public Widget
{
public:
    iuiImplWidget(Panel)
    Panel();
private:
};



class WindowStyle : public Style
{
public:
    void draw() override;
};

class Window : public Widget
{
public:
    iuiImplWidget(Window)
private:
};


} // namespace iui
#endif // iui_Window_h
