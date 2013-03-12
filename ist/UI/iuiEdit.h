#ifndef iui_Edit_h
#define iui_Edit_h
#include "iuiCommon.h"
#include "iuiWidget.h"
namespace iui {

class iuiInterModule LabelStyle : public Style
{
public:
    virtual void draw();
};

class iuiInterModule Label : public Widget
{
typedef Widget super;
public:
    iuiImplWidget(Label)
    Label(Widget *parent, const wchar_t *text=L"", const Rect &rect=Rect());

protected:
    virtual bool handleEvent(const WM_Base &wm);
};


class iuiInterModule EditboxStyle : public Style
{
public:
    virtual void draw();
};

class iuiInterModule Editbox : public Widget
{
typedef Widget super;
public:
    iuiImplWidget(Editbox)
    Editbox(Widget *parent, const wchar_t *text=L"", const Rect &rect=Rect(), const WidgetCallback &on_edit=WidgetCallback());
    bool    isHovered() const;
    bool    isReadOnly() const;
    int32   getCursor() const;
    void    setReadOnly(bool ro);
    void    setCursor(int32 cursor);

protected:
    virtual bool handleEvent(const WM_Base &wm);
private:
    istMemberPtrDecl(Members) m;
};



class iuiInterModule EditboxMultilineStyle : public Style
{
public:
    virtual void draw();
};

class iuiInterModule EditboxMultiline : public Widget
{
typedef Widget super;
public:
    EditboxMultiline(const wchar_t *text=L"", const WidgetCallback &on_change=WidgetCallback());
    bool            isReadOnly() const;
    const ivec2&    getCursor() const;
    void            setReadOnly(bool ro);
    void            setCursor(const ivec2& cursor);

protected:
    virtual Style* createDefaultStyle() const;
private:
    istMemberPtrDecl(Members) m;
};

} // namespace iui
#endif // iui_Edit_h
