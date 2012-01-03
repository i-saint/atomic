#ifndef __atomic_Graphics_Renderer_ForwardShading__
#define __atomic_Graphics_Renderer_ForwardShading__
namespace atomic {


class PassForwardShading_DistanceField : public IRenderer
{
private:
    VertexArray     *m_va_grid;
    AtomicShader    *m_sh_grid;

    VertexArray     *m_va_cell;
    VertexBuffer    *m_vbo_cell_pos;
    VertexBuffer    *m_vbo_cell_dist;
    AtomicShader    *m_sh_cell;

public:
    PassForwardShading_DistanceField();
    void beforeDraw();
    void draw();
};


} // namespace atomic
#endif // __atomic_Graphics_Renderer_ForwardShading__
