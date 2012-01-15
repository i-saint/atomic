#include "stdafx.h"
#include <wingdi.h>
#include "../Window.h"
#include "i3duglFont.h"


namespace ist {
namespace i3dgl {
#ifdef _WIN32

static const int g_list_base = 0;

SystemFont::SystemFont(HDC m_hdc)
    : m_hdc(m_hdc)
    , m_window_height(0)
    , m_font_height(0)
{
    SelectObject(m_hdc, GetStockObject(SYSTEM_FONT));
    wglUseFontBitmapsW( m_hdc, 0, 256*32, g_list_base );

    TEXTMETRIC metric;
    GetTextMetrics(m_hdc, &metric);
    m_font_height = metric.tmHeight;
    m_window_height = istGetAplication()->getWindowSize().y;
}

SystemFont::~SystemFont()
{
    m_hdc = NULL;
    m_window_height = 0;
    m_font_height = 0;
}

void SystemFont::draw(int x, int y, const char *text)
{
    int len = strlen(text);
    glWindowPos2i(x, m_window_height-m_font_height-y);
    glCallLists (len, GL_UNSIGNED_BYTE, text);
}

void SystemFont::draw(int x, int y, const wchar_t *text)
{
    int len = wcslen(text);
    glWindowPos2i(x, m_window_height-m_font_height-y);
    glCallLists (len, GL_UNSIGNED_SHORT, text);
}
#endif // _WIN32

} // namespace i3d
} // namespace ist
