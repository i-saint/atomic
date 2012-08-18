#ifndef __ist_UI_iuiSystem_h__
#define __ist_UI_iuiSystem_h__
#include "iuiCommon.h"
namespace ist {
namespace iui {

    class UIRenderer;


    class istInterModule UISystem : public SharedObject
    {
    public:
        UISystem();
        ~UISystem();

        void update(Float dt);
        void draw();

        UIRenderer* getRenderer() const { return m_renderer; }
        Widget* getWidget() const { return m_widget; }

    private:
        UIRenderer *m_renderer;
        Widget *m_widget;
    };

} // namespace iui
} // namespace ist
#endif // __ist_UI_iuiSystem_h__
