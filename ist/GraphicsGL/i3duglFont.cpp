﻿#include "istPCH.h"
#ifdef __ist_with_OpenGL__
#include <wingdi.h>
#include "ist/Base.h"
#include "ist/Window.h"
#include "i3duglFont.h"


namespace ist {
namespace i3dgl {
#ifdef istWindows


class SystemFont : public IFontRenderer
{
private:
    HDC m_hdc;
    int m_window_height;
    int m_font_height;

public:
    SystemFont(HDC m_hdc);
    ~SystemFont();

    void draw(const vec2 &pos, const char *text);
    void draw(const vec2 &pos, const wchar_t *text);
    void flush();

    float32 getFontHeight() const { return (float32)m_font_height; }
};


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

void SystemFont::draw(const vec2 &pos, const char *text)
{
    int len = strlen(text);
    glWindowPos2i((int32)pos.x, m_window_height-m_font_height-(int32)pos.y);
    glCallLists(len, GL_UNSIGNED_BYTE, text);
}

void SystemFont::draw(const vec2 &pos, const wchar_t *text)
{
    int len = wcslen(text);
    glWindowPos2i((int32)pos.x, m_window_height-m_font_height-(int32)pos.y);
    glCallLists(len, GL_UNSIGNED_SHORT, text);
}

void SystemFont::flush()
{
}

IFontRenderer* CreateSystemFont(void *hdc)
{
    return istNew(SystemFont)((HDC)hdc);
}






#define UCS2_CODE_MAX			65536

struct SFF_HEAD
{
    uint32 Guid;
    uint32 Version;
    int32 FontSize;
    int32 FontWidth;
    int32 FontHeight;
    int32 SheetMax;
    int32 FontMax;
    wchar_t SheetName[64];
    struct {
        uint32 IsVertical	: 1;
        uint32 Pad		: 31;
    } Flags;
    uint16 IndexTbl[UCS2_CODE_MAX];
};
struct SFF_DATA
{
    uint16 u;
    uint16 v;
    uint8 w;
    uint8 h;
    uint8 No;
    uint8 Offset;
    uint8 Width;
};

struct FontQuad
{
    vec2 pos;
    vec2 size;
};

class FSS
{
public:
    FSS() : m_header(NULL), m_data(NULL) {}

    bool load(const char *path)
    {
        {
            FILE *lfd = fopen(path, "rb");
            if(lfd==NULL) { istPrint("file open failed: %s\n", path); return false; }
            fseek(lfd, 0, SEEK_END);
            istPrint("filesize: %d\n", ftell(lfd));
            m_buf.resize(ftell(lfd));
            fseek(lfd, 0, SEEK_SET);
            fread(&m_buf[0], 1, m_buf.size(), lfd);
            fclose(lfd);
        }
        m_header = (const SFF_HEAD*)&m_buf[0];
        m_data = (const SFF_DATA*)(&m_buf[0]+sizeof(SFF_HEAD));
        return true;
    }

    vec2 getFontRect(const wchar_t *str, size_t len) const
    {
        ivec2 ret = ivec2(0,0);
        for(size_t i=0; i<len; ++i) {
            uint32 di = m_header->IndexTbl[str[i]];
            if(di==0xffff) {
                istPrint(L"FSS::getFontRect() %c not found\n", str[i]);
                continue;
            }
            ret.x += m_data[di].w;
            ret.y = stl::max<int32>(ret.y, m_data[di].h);
        }
        return vec2((float32)ret.x, (float32)ret.y);
    }

private:
    stl::vector<char> m_buf;
    const SFF_HEAD *m_header;
    const SFF_DATA *m_data;
};

class SpriteFontRenderer : public SharedObject
{
public:
    virtual void draw(const vec2 &pos, const char *text)
    {

    }

    virtual void draw(const vec2 &pos, const wchar_t *text)
    {

    }

    virtual void flush()
    {

    }

    virtual float32 getFontHeight() const
    {

    }
};



#endif // istWindows

} // namespace i3d
} // namespace ist
#endif // __ist_with_OpenGL__
