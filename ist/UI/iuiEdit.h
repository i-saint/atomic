#ifndef iui_Edit_h
#define iui_Edit_h
#include "iuiCommon.h"
#include "iuiWidget.h"
namespace iui {

class iuiAPI LabelStyle : public Style
{
public:
    LabelStyle();
    virtual void draw();
};

class iuiAPI Label : public Widget
{
typedef Widget super;
public:
    iuiImplWidget(Label)
    Label(Widget *parent, const wchar_t *text=L"", const Rect &rect=Rect());
    bool handleEvent(const WM_Base &wm) override;
};


class iuiAPI EditboxStyle : public Style
{
public:
    virtual void draw();
};

class iuiAPI Editbox : public Widget
{
typedef Widget super;
public:
    iuiImplWidget(Editbox)
    Editbox(Widget *parent, const wchar_t *text=L"", const Rect &rect=Rect(), WidgetCallback on_edit=WidgetCallback());
    void    update(Float dt);
    bool    isHovered() const;
    bool    isReadOnly() const;
    int32   getCursor() const;
    void    setReadOnly(bool ro);
    void    setCursor(int32 cursor);
    bool    handleEvent(const WM_Base &wm) override;
    void    setText(const String &v, bool e=true) override;

private:
    bool m_readonly;
    bool m_hovered;
    bool m_ime_on;
    int32 m_cursor;
};



class iuiAPI EditboxMultilineStyle : public Style
{
public:
    virtual void draw();
};

class iuiAPI EditboxMultiline : public Widget
{
typedef Widget super;
public:
    iuiImplWidget(EditboxMultiline)
    EditboxMultiline(const wchar_t *text=L"", WidgetCallback on_change=WidgetCallback());
    bool            isReadOnly() const;
    const ivec2&    getCursor() const;
    void            setReadOnly(bool ro);
    void            setCursor(const ivec2& cursor);

protected:
    virtual Style* createDefaultStyle() const;
private:
    bool m_readonly;
    ivec2 m_cursor;
};

} // namespace iui
#endif // iui_Edit_h
