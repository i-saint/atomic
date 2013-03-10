#ifndef atomic_UI_Title_h
#define atomic_UI_Title_h
#include "ist/iui.h"

namespace atomic {

class TitleWindow;
class StartWindow;
class RecordWindow;
class ConfigWindow;
class LogWindow;

class RootWindow : public iui::RootWindow
{
public:
    RootWindow();

private:
    TitleWindow     *m_title;
    ConfigWindow    *m_config;
    LogWindow       *m_log;
};

} // namespace atomic
#endif // atomic_UI_Title_h
