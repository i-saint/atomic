#ifndef iui_Edit_h
#define iui_Edit_h
#include "iuiCommon.h"
#include "iuiWidget.h"
namespace iui {

class iuiInterModule EditboxStyle : public Style
{
public:
    virtual void draw();
};

class iuiInterModule Editbox : public Widget
{
public:
    Editbox(const wchar_t *text=L"", const WidgetCallback &on_change=WidgetCallback(), Style *style=NULL);
    bool isReadOnly() const;
    int32 getCursor() const;

protected:
    virtual Style* createDefaultStyle() const;
private:
    struct Members;
    istMemberPtrDecl(Members) m;
};



class iuiInterModule EditboxMultilineStyle : public Style
{
public:
    virtual void draw();
};

class iuiInterModule EditboxMultiline : public Widget
{
public:
    EditboxMultiline(const wchar_t *text=L"", const WidgetCallback &on_change=WidgetCallback(), Style *style=NULL);
    bool isReadOnly() const;
    const ivec2& getCursor() const;

protected:
    virtual Style* createDefaultStyle() const;
private:
    struct Members;
    istMemberPtrDecl(Members) m;
};

} // namespace iui
#endif // iui_Edit_h
