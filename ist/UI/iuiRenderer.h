#ifndef __ist_UI_iuiRenderer_h__
#define __ist_UI_iuiRenderer_h__
#include "iuiCommon.h"
#include "ist/Graphics.h"

namespace ist {
namespace iui {


enum RenderCommandType {
    RC_Rect,
    RC_Circle,
    RC_Line,
    RC_Text,
};
enum FillMode {
    FM_Fill,
    FM_Line,
};
struct RCTextOption
{
    Color color;
    Float size;
    Float spacing;
    bool monospace;
};

class UIRenderer;
class IRenderCommand;
typedef stl::vector<IRenderCommand*> RenderCommands;


class istInterModule IRenderCommand
{
istMakeDestructable;
friend class UIRenderer;
protected:
    IRenderCommand();
    virtual ~IRenderCommand();
public:
    virtual RenderCommandType getType() const=0;
};

class istInterModule RCRect : public IRenderCommand
{
private:
    RCRect(const Rect &s, const Color &c, FillMode f);
public:
    static RCRect* create(const Rect &r, const Color &c, FillMode f);
    virtual RenderCommandType getType() const { return RC_Rect; }

    Rect shape;
    Color color;
    FillMode fillmode;
};

class istInterModule RCCircle : public IRenderCommand
{
private:
    RCCircle();
public:
    static RCCircle* create(const Circle &s, const Color &c, FillMode f);
    virtual RenderCommandType getType() const { return RC_Circle; }

    Circle shape;
    Color color;
    FillMode fillmode;
};

class istInterModule RCLine : public IRenderCommand
{
private:
    RCLine() {}
public:
    static RCLine* create(const Line &s, const Color &c);
    virtual RenderCommandType getType() const { return RC_Line; }

    Line shape;
    Color color;
};

class istInterModule RCText : public IRenderCommand
{
private:
    RCText() {}
public:
    static RCText* create();
    virtual RenderCommandType getType() const { return RC_Text; }

    String text;
    RCTextOption option;
};




struct VertexT
{
    vec2 pos;
    vec2 texcoord;
    vec4 color;
};

class istInterModule UIRenderer : public SharedObject
{
public:
    UIRenderer(i3d::Device *dev, i3d::DeviceContext *dc);
    ~UIRenderer();

    void addCommand(IRenderCommand *command);
    void flush();

    void draw(const RCRect &command);
    void draw(const RCCircle &command);
    void draw(const RCLine &command);
    void draw(const RCText &command);

private:
    void releaseAllCommands();

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

    RenderCommands m_commands;
};


} // namespace iui
} // namespace ist
#endif // __ist_UI_iuiRenderer_h__
