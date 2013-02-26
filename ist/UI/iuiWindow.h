#ifndef iui_Window_h
#define iui_Window_h
#include "iuiCommon.h"
#include "iuiWidget.h"
namespace iui {


class RootWindow : public Widget
{
public:
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
private:
};


} // namespace iui
#endif // iui_Window_h
