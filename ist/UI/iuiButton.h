#ifndef iui_Button_h
#define iui_Button_h
#include "iuiCommon.h"
#include "iuiWidget.h"
namespace iui {

class iuiInterModule ButtonStyle : public Style
{
public:
    ButtonStyle();
    virtual void draw();
};

class iuiInterModule Button : public Widget
{
typedef Widget super;
public:
    iuiImplWidget(Button)
    Button(Widget *parent, const wchar_t *text=L"", const Rect &pos=Rect(), const WidgetCallback &on_press=WidgetCallback());
    void update(Float dt);
    bool isPressing() const;
    bool isHovered() const;

protected:
    virtual bool handleEvent(const WM_Base &wm);
private:
    istMemberPtrDecl(Members) m;
};


class iuiInterModule ToggleButtonStyle : public Style
{
public:
    virtual void draw();
};

class iuiInterModule ToggleButton : public Widget
{
typedef Widget super;
public:
    iuiImplWidget(ToggleButton)
    ToggleButton(const wchar_t *text=L"", const WidgetCallback &on_toggle=WidgetCallback());
    void update(Float dt);
    bool isPressed() const;
    bool isPressing() const;
    bool isHovered() const;

protected:
    virtual bool handleEvent(const WM_Base &wm);
private:
    istMemberPtrDecl(Members) m;
};


class iuiInterModule CheckboxStyle : public Style
{
public:
    virtual void draw();
};

class iuiInterModule Checkbox : public Widget
{
typedef Widget super;
public:
    iuiImplWidget(Checkbox)
    Checkbox(const wchar_t *text=L"", const WidgetCallback &on_toggle=WidgetCallback());
    bool isChecked() const;
    bool isPressing() const;

protected:
    virtual bool handleEvent(const WM_Base &wm);
private:
    istMemberPtrDecl(Members) m;
};


} // namespace iui
#endif // iui_Button_h
