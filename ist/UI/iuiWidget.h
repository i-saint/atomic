#ifndef __ist_UI_iuiWidget_h__
#define __ist_UI_iuiWidget_h__
#include "iuiCommon.h"
namespace ist {
namespace iui {

    class Widget;
    typedef stl::vector<Widget*> Widgets;
    class Style;


    class istInterModule Widget : public SharedObject
    {
    public:
        Widget();
        ~Widget();

        virtual void update(Float dt);
        virtual bool handleEvent();

        Widgets&        getChildren()       { return m_children; }
        const Widgets&  getChildren() const { return m_children; }
        Style*          getStyle() const    { return m_style; }
        const Position& getPosition() const { return m_pos; }
        const Size&     getSize() const     { return m_size; }
        const String&   getText() const     { return m_text; }

    private:
        Widgets m_children;
        Style *m_style;
        Position m_pos;
        Size m_size;
        String m_text;
    };


    class istInterModule Style : public SharedObject
    {
    public:
        Style();
        ~Style();

        virtual void render();

    private:
    };

} // namespace iui
} // namespace ist
#endif // __ist_UI_iuiWidget_h__
