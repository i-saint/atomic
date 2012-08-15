#include "istPCH.h"
#ifdef __ist_with_OpenGL__
#include <wingdi.h>
#include "ist/Base.h"
#include "ist/Window.h"
#include "ist/GraphicsGL/i3dglDevice.h"
#include "ist/GraphicsGL/i3dglDeviceContext.h"
#include "ist/GraphicsGL/i3duglFont.h"
#include "ist/GraphicsGL/i3dglUtil.h"


namespace ist {
namespace i3dgl {
#ifdef istWindows

static const int g_list_base = 0;

class SystemFont : public IFontRenderer
{
private:
    HDC m_hdc;
    int m_window_height;
    int m_font_height;

public:
    SystemFont(HDC m_hdc)
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

    ~SystemFont()
    {
        m_hdc = NULL;
        m_window_height = 0;
        m_font_height = 0;
    }

    virtual void setScreen(float32 left, float32 right, float32 bottom, float32 top) {}
    virtual void setColor(const vec4 &v) {}
    virtual void setSize(float32 v) {}

    void addText(const vec2 &pos, const char *text, size_t len)
    {
        glUseProgram(0);
        glWindowPos2i((int32)pos.x, m_window_height-(int32)pos.y);
        glCallLists(len, GL_UNSIGNED_BYTE, text);
    }

    void addText(const vec2 &pos, const wchar_t *text, size_t len)
    {
        glUseProgram(0);
        glWindowPos2i((int32)pos.x, m_window_height-(int32)pos.y);
        glCallLists(len, GL_UNSIGNED_SHORT, text);
    }

    void flush(DeviceContext *dc)
    {
    }
};

IFontRenderer* CreateSystemFont(Device *device, void *hdc)
{
    return istNew(SystemFont)((HDC)hdc);
}

#endif // istWindows





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
    vec2 uv_pos;
    vec2 uv_size;
};

class FSS
{
public:
    FSS() : m_header(NULL), m_data(NULL) {}

    void setTextureSize(const vec2 &v)
    {
        m_tex_size = v;
        m_rcp_tex_size = vec2(1.0f, 1.0f) / m_tex_size;
    }

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

    void makeQuads(const vec2 &pos, const wchar_t *text, size_t len, float32 size, stl::vector<FontQuad> &quads) const
    {
        if(m_header==NULL) { return; }

        const float scale = (float32)m_header->FontHeight / size;
        vec2 base = pos;
        vec2 texsize = vec2(float32(m_header->FontWidth), float32(m_header->FontHeight));
        for(size_t i=0; i<len; ++i) {
            uint32 di = m_header->IndexTbl[text[i]];
            if(di==0xffff) {
                istPrint(L"FSS::getFontRect() %c not found\n", text[i]);
                continue;
            }
            const SFF_DATA &cdata = m_data[di];
            vec2 uv = vec2(cdata.u, cdata.v);
            vec2 wh = vec2(cdata.w, cdata.h);
            vec2 scaled_wh = wh * scale;
            FontQuad q = {base, scaled_wh, uv*m_rcp_tex_size, wh*m_rcp_tex_size};
            quads.push_back(q);
            base.x += scaled_wh.x;
        }
    }

private:
    stl::vector<char> m_buf;
    const SFF_HEAD *m_header;
    const SFF_DATA *m_data;
    vec2 m_tex_size;
    vec2 m_rcp_tex_size;
};

namespace {
    const char *g_font_vssrc = "\
struct RenderStates\
{\
    mat4 ViewProjectionMatrix;\
    vec4 Color;\
};\
layout(std140) uniform render_states\
{\
    RenderStates u_RS;\
};\
layout(location=0) in vec2 ia_VertexPosition;\
layout(location=1) in vec2 ia_VertexTexcoord0;\
out vec2 vs_Texcoord;\
\
void main(void)\
{\
    vs_Texcoord = ia_VertexTexcoord0;\
    gl_Position = u_RS.ViewProjectionMatrix * vec4(ia_VertexPosition, 0.0, 1.0);\
}\
";

    const char *g_font_pssrc = "\
struct RenderStates\
{\
    mat4 ViewProjectionMatrix;\
    vec4 Color;\
};\
layout(std140) uniform render_states\
{\
    RenderStates u_RS;\
};\
in vec2 vs_Texcoord;\
layout(location=0) out vec4 ps_FragColor;\
\
void main()\
{\
    vec4 color = u_RS.Color;\
    color.a = texture(u_ColorBuffer, vs_Texcoord).r;\
    ps_FragColor = color;\
}\
";
} // namespace

class SpriteFontRenderer : public IFontRenderer
{
public:
    struct VertexT
    {
        vec2 pos;
        vec2 texcoord;

        VertexT() {}
        VertexT(const vec2 &p, const vec2 &t) : pos(p), texcoord(t) {}
    };
    struct RenderState
    {
        mat4 matrix;
        vec4 color;
    };

    static const size_t MaxCharsPerDraw = 1024;

public:
    SpriteFontRenderer(Device *dev, const char *path_to_fss, const char *path_to_png)
        : m_texture(NULL)
        , m_vbo(NULL)
        , m_ubo(NULL)
        , m_vs(NULL)
        , m_ps(NULL)
        , m_shader(NULL)
        , m_size(20.0f)
    {
        if(m_texture = CreateTexture2DFromFile(dev, path_to_png)) {
            m_fss.load(path_to_fss);
            m_fss.setTextureSize(vec2(m_texture->getDesc().size));
        }
        m_vbo = CreateVertexBuffer(dev, sizeof(VertexT)*4*MaxCharsPerDraw, I3D_USAGE_DYNAMIC);
        m_ubo = CreateUniformBuffer(dev, sizeof(RenderState), I3D_USAGE_DYNAMIC);
        {
            m_va = dev->createVertexArray();
            const VertexDesc descs[] = {
                {0, I3D_FLOAT, 2, 0, false, 0},
                {1, I3D_FLOAT, 2, 8, false, 0},
            };
            m_va->setAttributes(*m_vbo, sizeof(VertexT), descs, _countof(descs));
        }

        m_vs = CreateVertexShaderFromString(dev, g_font_vssrc);
        m_ps = CreatePixelShaderFromString(dev, g_font_pssrc);
        m_shader = dev->createShaderProgram(ShaderProgramDesc(m_vs, m_ps));
    }

    ~SpriteFontRenderer()
    {
        istSafeRelease(m_shader);
        istSafeRelease(m_ps);
        istSafeRelease(m_vs);
        istSafeRelease(m_va);
        istSafeRelease(m_ubo);
        istSafeRelease(m_vbo);
        istSafeRelease(m_texture);
    }

    virtual void setScreen(float32 left, float32 right, float32 bottom, float32 top)
    {
        m_renderstate.matrix = glm::ortho(left, right, bottom, top);
    }
    virtual void setColor(const vec4 &v) { m_renderstate.color=v; }
    virtual void setSize(float32 v) { m_size=v; }

    virtual void addText(const vec2 &pos, const char *text, size_t len)
    {
        size_t wlen = mbstowcs(NULL, text, 0);
        if(wlen==size_t(-1)) { return; }
        wchar_t *wtext = (wchar_t*)istRawAlloca(sizeof(wchar_t)*wlen);
        mbstowcs(wtext, text, len);
        addText(pos, wtext, wlen);
    }

    virtual void addText(const vec2 &pos, const wchar_t *text, size_t len)
    {
        m_fss.makeQuads(pos, text, len, m_size, m_quads);
    }

    virtual void flush(DeviceContext *dc)
    {
        if(m_quads.empty()) { return; }

        size_t num_quad = stl::min<size_t>(m_quads.size(), MaxCharsPerDraw);
        size_t num_vertex = num_quad*4;
        {
            VertexT *vertex = (VertexT*)m_vbo->map(I3D_MAP_WRITE);
            for(size_t qi=0; qi<num_quad; ++qi) {
                const FontQuad &quad = m_quads[qi];
                VertexT *v = &vertex[qi*4];
                const vec2 pos_min = quad.pos;
                const vec2 pos_max = quad.pos + quad.size;
                const vec2 tex_min = quad.uv_pos;
                const vec2 tex_max = quad.uv_pos + quad.uv_size;
                v[0] = VertexT(vec2(pos_min.x, pos_min.y), vec2(tex_min.x, tex_min.y));
                v[1] = VertexT(vec2(pos_min.x, pos_max.y), vec2(tex_min.x, tex_max.y));
                v[2] = VertexT(vec2(pos_max.x, pos_max.y), vec2(tex_max.x, tex_max.y));
                v[3] = VertexT(vec2(pos_max.x, pos_min.y), vec2(tex_max.x, tex_min.y));
            }
            m_vbo->unmap();
        }
        MapAndWrite(*m_ubo, &m_renderstate, sizeof(m_renderstate));
        m_quads.clear();

        dc->setVertexArray(m_va);
        dc->setShader(m_shader);
        dc->draw(I3D_QUADS, 0, num_vertex);
    }

private:
    FSS m_fss;
    stl::vector<FontQuad> m_quads;
    Texture2D *m_texture;
    Buffer *m_vbo;
    Buffer *m_ubo;
    VertexArray *m_va;
    VertexShader *m_vs;
    PixelShader *m_ps;
    ShaderProgram *m_shader;
    float m_size;
    RenderState m_renderstate;
};

IFontRenderer* CreateSpriteFont(Device *device, const char *path_to_sff, const char *path_to_png)
{
    return new SpriteFontRenderer(device, path_to_sff, path_to_png);
}


} // namespace i3d
} // namespace ist
#endif // __ist_with_OpenGL__
