#include "istPCH.h"
#ifdef ist_with_OpenGL
#include <wingdi.h>
#include "ist/Base.h"
#include "ist/Window.h"
#include "ist/GraphicsCommon/Image.h"
#include "ist/GraphicsCommon/EasyDrawer.h"
#include "ist/GraphicsGL/i3dglDevice.h"
#include "ist/GraphicsGL/i3dglDeviceContext.h"
#include "ist/GraphicsGL/i3duglFont.h"
#include "ist/GraphicsGL/i3dglUtil.h"


namespace ist {
namespace i3dgl {


IFontRenderer::IFontRenderer()
{
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
    vec2 uv_pos;
    vec2 uv_size;
    vec4 color;
};

class FSS
{
public:
    FSS()
        : m_header(NULL)
        , m_data(NULL)
        , m_color(1.0f, 1.0f, 1.0f, 1.0f)
        , m_size(0.0f)
        , m_spacing(1.0f)
        , m_monospace(false)
    {}

    void setTextureSize(const vec2 &v)
    {
        m_tex_size = v;
        m_rcp_tex_size = vec2(1.0f, 1.0f) / m_tex_size;
    }

    void setColor(const vec4 &v){ m_color=v; }
    void setSize(float32 v)     { m_size=v; }
    void setSpace(float32 v)    { m_spacing=v; }
    void setMonospace(bool v)   { m_monospace=v; }

    float getFontSize() const
    {
        return m_header!=NULL ? (float32)m_header->FontSize : 0.0f;
    }

    bool load(IBinaryStream &bf)
    {
        bf.setReadPos(0, IBinaryStream::Seek_End);
        m_buf.resize((size_t)bf.getReadPos());
        bf.setReadPos(0);
        bf.read(&m_buf[0], m_buf.size());
        if(m_buf.size()<4 || !(m_buf[0]=='F', m_buf[1]=='F', m_buf[2]=='S')) { return false; }

        m_header = (const SFF_HEAD*)&m_buf[0];
        m_data = (const SFF_DATA*)(&m_buf[0]+sizeof(SFF_HEAD));
        if(m_size==0.0f) { m_size=getFontSize(); }
        return true;
    }

    vec2 computeTextSize(const wchar_t *text, size_t len)
    {
        if(m_header==NULL) { return vec2(); }

        const float32 base_size = (float32)m_header->FontSize;
        const float32 scale = m_size / base_size;
        vec2 base;
        vec2 quad_pos;
        vec2 quad_size;
        for(size_t i=0; i<len; ++i) {
            uint32 di = m_header->IndexTbl[text[i]];
            float advance = (text[i] <= 0xff ? base_size*0.5f : base_size) * scale * m_spacing;
            if(di!=0xffff) {
                const SFF_DATA &cdata = m_data[di];
                vec2 wh = vec2(cdata.w, cdata.h);
                vec2 scaled_wh = wh * scale;
                vec2 scaled_offset = vec2(float32(cdata.Offset) * scale, 0.0f);
                quad_pos = base+scaled_offset;
                quad_size = scaled_wh;
                if(!m_monospace) { advance = (scaled_wh.x + scaled_offset.x) * m_spacing; }
            }
            base.x += advance;
        }
        return quad_pos+quad_size;
    }

    void makeQuads(const vec2 &pos, const wchar_t *text, size_t len, ist::raw_vector<FontQuad> &quads) const
    {
        if(m_header==NULL) { return; }

        const float32 base_size = (float32)m_header->FontSize;
        const float32 scale = m_size / base_size;
        vec2 base = pos;
        for(size_t i=0; i<len; ++i) {
            uint32 di = m_header->IndexTbl[text[i]];
            float advance = (text[i] <= 0xff ? base_size*0.5f : base_size) * scale * m_spacing;
            if(di!=0xffff) {
                const SFF_DATA &cdata = m_data[di];
                vec2 uv = vec2(cdata.u, cdata.v);
                vec2 wh = vec2(cdata.w, cdata.h);
                vec2 scaled_wh = wh * scale;
                vec2 scaled_offset = vec2(float32(cdata.Offset) * scale, 0.0f);
                vec2 uv_pos = uv*m_rcp_tex_size;
                vec2 uv_size = wh * m_rcp_tex_size;
                FontQuad q = {base+scaled_offset, scaled_wh, uv_pos, uv_size, m_color};
                quads.push_back(q);
                if(!m_monospace) { advance = (scaled_wh.x + scaled_offset.x) * m_spacing; }
            }
            base.x += advance;
        }
    }

private:
    ist::raw_vector<char> m_buf;
    const SFF_HEAD *m_header;
    const SFF_DATA *m_data;
    vec2 m_tex_size;
    vec2 m_rcp_tex_size;

    vec4 m_color;
    float32 m_size;
    float32 m_spacing;
    bool m_monospace;
};


const char *g_font_pssrc = "\
#version 330 core\n\
uniform sampler2D u_Texture;\
in vec2 vs_Texcoord;\
in vec4 vs_Color;\
layout(location=0) out vec4 ps_FragColor;\
\
void main()\
{\
    vec4 color = vs_Color;\
    color.a *= texture(u_Texture, vs_Texcoord).r;\
    ps_FragColor = color;\
}\
";

class SpriteFontRenderer : public IFontRenderer
{
public:
    typedef VertexP2T2C4 Vertex;

public:
    SpriteFontRenderer()
        : m_drawer(NULL)
        , m_sampler(NULL)
        , m_texture(NULL)
        , m_shader(NULL)
    {}

    ~SpriteFontRenderer()
    {
        istSafeRelease(m_shader);
        istSafeRelease(m_sampler);
        istSafeRelease(m_texture);
        istSafeRelease(m_drawer);
    }

    bool initialize(IBinaryStream &fss_stream, IBinaryStream &img_stream, EasyDrawer *drawer)
    {
        Device *dev = GetDevice();
        if(drawer) {
            drawer->addRef();
            m_drawer = drawer;
        }
        else {
            m_drawer = CreateEasyDrawer();
        }

        {
            Image img, alpha;
            if(!img.load(img_stream)) { return false; }
            // alpha だけ抽出。RGBA ではない画像であれば red だけ抽出
            if(!ExtractAlpha(img, alpha)) { ExtractRed(img, alpha); }
            m_texture = CreateTexture2DFromImage(dev, alpha);
        }
        if(!m_fss.load(fss_stream)) {
            return false;
        }
        m_fss.setTextureSize(vec2(m_texture->getDesc().size));
        m_sampler = dev->createSampler(SamplerDesc(I3D_CLAMP_TO_EDGE, I3D_CLAMP_TO_EDGE, I3D_CLAMP_TO_EDGE, I3D_LINEAR, I3D_LINEAR));

        VertexShader *vs = CreateVertexShaderFromString(dev, g_vs_p2t2c4);
        PixelShader *ps  = CreatePixelShaderFromString(dev, g_font_pssrc);
        m_shader = dev->createShaderProgram(ShaderProgramDesc(vs, ps));
        istSafeRelease(ps);
        istSafeRelease(vs);

        m_state.setSampler(m_sampler);
        m_state.setTexture(m_texture);
        m_state.setShader(m_shader);

        return true;
    }

    virtual void setScreen(float32 left, float32 right, float32 bottom, float32 top)
    {
        m_state.setScreen(left, right, bottom, top);
    }
    virtual void setColor(const vec4 &v)    { m_fss.setColor(v); }
    virtual void setSize(float32 v)         { m_fss.setSize(v); }
    virtual void setSpacing(float32 v)      { m_fss.setSpace(v); }
    virtual void setMonospace(bool v)       { m_fss.setMonospace(v); }

    virtual void addText(const vec2 &pos, const char *text, size_t len)
    {
        // _alloca() で一時領域高速に取りたいところだが、_alloca() はマルチスレッド非対応っぽいので素直な実装で
        if(len==0) { len = strlen(text); }
        if(len==0) { return; }
        stl::string tmp(text, len);

        size_t wlen = mbstowcs(NULL, tmp.c_str(), 0);
        if(wlen==size_t(-1)) { return; }

        stl::wstring wtext;
        wtext.resize(wlen);
        mbstowcs(&wtext[0], tmp.c_str(), wlen);
        addText(pos, wtext.c_str(), wlen);
    }

    virtual void addText(const vec2 &pos, const wchar_t *text, size_t len)
    {
        m_fss.makeQuads(pos, text, len, m_quads);
    }

    virtual vec2 computeTextSize(const char *text, size_t len=0)
    {
        if(len==0) { len = strlen(text); }
        if(len==0) { return vec2(); }
        stl::string tmp(text, len);

        size_t wlen = mbstowcs(NULL, tmp.c_str(), 0);
        if(wlen==size_t(-1)) { return vec2(); }

        stl::wstring wtext;
        wtext.resize(wlen);
        mbstowcs(&wtext[0], tmp.c_str(), wlen);
        return computeTextSize(wtext.c_str(), wlen);
    }

    virtual vec2 computeTextSize(const wchar_t *text, size_t len=0)
    {
        return m_fss.computeTextSize(text, len);
    }

    virtual void draw()
    {
        if(m_quads.empty()) { return; }

        EasyDrawState state_prev = m_drawer->getRenderStates();
        m_drawer->forceSetRenderStates(m_state);

        size_t num_quad = m_quads.size();
        size_t num_vertex = num_quad*4;
        m_vertices.resize(num_quad*4);
        m_indices.resize(num_quad*6);
        for(size_t qi=0; qi<num_quad; ++qi) {
            const FontQuad &quad = m_quads[qi];
            const vec2 pos_min = quad.pos;
            const vec2 pos_max = quad.pos + quad.size;
            const vec2 tex_min = quad.uv_pos;
            const vec2 tex_max = quad.uv_pos + quad.uv_size;
            size_t qi4 = qi*4;
            size_t qi6 = qi*6;
            m_vertices[qi4+0] = Vertex(vec2(pos_min.x, pos_min.y), vec2(tex_min.x, tex_min.y), quad.color);
            m_vertices[qi4+1] = Vertex(vec2(pos_min.x, pos_max.y), vec2(tex_min.x, tex_max.y), quad.color);
            m_vertices[qi4+2] = Vertex(vec2(pos_max.x, pos_max.y), vec2(tex_max.x, tex_max.y), quad.color);
            m_vertices[qi4+3] = Vertex(vec2(pos_max.x, pos_min.y), vec2(tex_max.x, tex_min.y), quad.color);
            m_indices[qi6+0] = qi4+0;
            m_indices[qi6+1] = qi4+1;
            m_indices[qi6+2] = qi4+2;
            m_indices[qi6+3] = qi4+2;
            m_indices[qi6+4] = qi4+3;
            m_indices[qi6+5] = qi4+0;
        }
        if(!m_vertices.empty()) {
            m_drawer->draw(I3D_TRIANGLES, &m_vertices[0], m_vertices.size(), &m_indices[0], m_indices.size());
        }
        m_quads.clear();
        m_vertices.clear();
        m_indices.clear();

        m_drawer->forceSetRenderStates(state_prev);
    }

private:
    FSS m_fss;
    ist::raw_vector<FontQuad> m_quads;
    ist::raw_vector<Vertex> m_vertices;
    ist::raw_vector<uint32> m_indices;
    EasyDrawer *m_drawer;
    EasyDrawState m_state;

    Sampler *m_sampler;
    Texture2D *m_texture;
    ShaderProgram *m_shader;
};

IFontRenderer* CreateSpriteFont(const char *path_to_sff, const char *path_to_img, EasyDrawer *drawer)
{
    FileStream sff(path_to_sff, "rb");
    FileStream img(path_to_img, "rb");
    if(!sff.isOpened()) { istPrint("%s load failed\n", path_to_sff); return NULL; }
    if(!img.isOpened()) { istPrint("%s load failed\n", path_to_img); return NULL; }
    return CreateSpriteFont(sff, img, drawer);
}

IFontRenderer* CreateSpriteFont(IBinaryStream &sff, IBinaryStream &img, EasyDrawer *drawer)
{
    SpriteFontRenderer *r = istNew(SpriteFontRenderer)();
    if(!r->initialize(sff, img, drawer)) {
        return NULL;
    }
    return r;
}


} // namespace i3d
} // namespace ist
#endif // ist_with_OpenGL
