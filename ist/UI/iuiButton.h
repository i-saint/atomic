#ifndef iui_Button_h
#define iui_Button_h
#include "iuiCommon.h"
#include "iuiWidget.h"
namespace iui {

class iuiInterModule ButtonStyle : public Style
{
public:
    virtual void draw();
};

class iuiInterModule Button : public Widget
{
public:
    iuiImplWidget(Button)
    Button(const wchar_t *text=L"", const WidgetCallback &on_press=WidgetCallback());
    bool isPressing() const;

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
public:
    iuiImplWidget(ToggleButton)
    ToggleButton(const wchar_t *text=L"", const WidgetCallback &on_toggle=WidgetCallback());
    bool isPressed() const;
    bool isPressing() const;

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
public:
    iuiImplWidget(Checkbox)
    Checkbox(const wchar_t *text=L"", const WidgetCallback &on_toggle=WidgetCallback());
    bool isChecked() const;
    bool isPressing() const;

private:
    istMemberPtrDecl(Members) m;
};


} // namespace iui
#endif // iui_Button_h
