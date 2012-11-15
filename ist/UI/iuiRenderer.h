#ifndef __ist_UI_iuiRenderer_h__
#define __ist_UI_iuiRenderer_h__
#include "iuiCommon.h"
#include "ist/Graphics.h"

namespace ist {
namespace iui {

// render command
enum RCType {
    RC_Rect,
    RC_Line,
    RC_Text,
};


struct istAlign(16) RCRect
{
    static void create(RCRect &out, const Rect &rect, const Color &color);

    RCType ctype;
    Rect shape;
    Color color;
};

struct istAlign(16) RCLine
{
    static void create(RCLine &out, Line &shape, const Color &color);

    RCType ctype;
    Line shape;
    Color color;
};

struct istAlign(16) RCText
{
    static void create(RCText &out,
        const char *text,
        size_t textlen,
        const Color &color,
        Float size,
        Float spacing,
        bool monospace );

    RCType ctype;
    const char *text;
    uint32 textlen;
    Color color;
    Float size;
    Float spacing;
    bool monospace;
};




struct VertexT
{
    vec4 pos;
    vec4 color;
    vec2 texcoord;
};

class istInterModule UIRenderer : public SharedObject
{
public:
    UIRenderer(i3d::Device *dev, i3d::DeviceContext *dc);
    ~UIRenderer();

    /// 全て thread unsafe です。専用のスレッドで処理する必要があります
    void begin(); /// 描画コマンド発行の前に呼びます
    void addCommand(const RCRect &command); /// begin() の後、end() の前に呼ぶ必要があります
    void addCommand(const RCLine &command); /// begin() の後、end() の前に呼ぶ必要があります
    void addCommand(const RCText &command); /// begin() の後、end() の前に呼ぶ必要があります
    void end(); /// この中で実際の描画 (DeviceContext へのコマンドの発行) が行われます

private:
    void draw(const RCRect &command);
    void draw(const RCLine &command);
    void draw(const RCText &command);

    i3d::Device *m_dev;
    i3d::DeviceContext *m_dc;
    i3d::Sampler *m_sampler;
    i3d::Texture2D *m_texture;
    i3d::Buffer *m_vbo;
    i3d::Buffer *m_ubo;
    i3d::VertexArray *m_va;
    i3d::VertexShader *m_vs;
    i3d::PixelShader *m_ps;
    i3d::ShaderProgram *m_shader;
    GLint m_uniform_loc;

    stl::vector<char> m_commands;
    stl::string m_text;
};


} // namespace iui
} // namespace ist
#endif // __ist_UI_iuiRenderer_h__
