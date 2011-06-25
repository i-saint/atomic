#ifndef __ist_Graphics_DrawCommand__
#define __ist_Graphics_DrawCommand__

namespace ist {
namespace graphics {


enum DRAW_COMMAND_TYPE
{
    DC_NOP,
    DC_END,
    DC_WAIT,
    DC_JUMP,
    DC_CALL,

    DC_VIEWPORT,
    DC_MATRIX,
    DC_RENDER_STATE,
    DC_FRAME_BUFFER,
    DC_TEXTURE,
    DC_VERTEX_BUFFER,
    DC_INDEX_BUFFER,
    DC_UNIFORM_BUFFER,
    DC_VERTEX_SHADER,
    DC_FRAGMENT_SHADER,
    DC_GEOMETRY_SHADER,
    DC_DRAW_PRIMITIVE,
};


struct DC_Nop
{
    const int type;
    DC_Nop() : type(DC_NOP) {}
};

struct DC_End
{
    const int type;
    DC_End() : type(DC_END) {}
};

struct DC_Wait
{
    const int type;
    DC_Wait() : type(DC_WAIT) {}
};

struct DC_Jump
{
    const int type;
    void *address;
    DC_Jump() : type(DC_JUMP) {}
};

struct DC_Call
{
    const int type;
    void *address;
    DC_Call() : type(DC_CALL) {}
};


struct DC_Viewport
{
    const int type;
    Viewport viewport;
    DC_Viewport() : type(DC_VIEWPORT) {}
};

struct DC_Matrix
{
    enum {
        MAT_PROJECTION,
        MAT_MODELVIEW,
    };
    const int type;
    int matrix_type;
    XMMATRIX matrix;
    DC_Matrix() : type(DC_MATRIX) {}
}


class DrawCommandList
{
private:
    eastl::vector<char> m_commandbuf;

public:
    void resizeBuffer(size_t size);

    size_t pushNop();
    size_t pushWait();
    size_t pushJump(const DrawCommandList& cl, size_t pos=0);
    size_t pushCall(const DrawCommandList& cl, size_t pos=0);
};

} // namespace graphics
} // namespace ist
#endif __ist_Graphics_DrawCommand__
