#ifndef atm_Graphics_Renderer_ForwardShading_h
#define atm_Graphics_Renderer_ForwardShading_h
namespace atm {


class PassForwardShading_DistanceField : public IRenderer
{
private:
    VertexArray     *m_va_grid;
    AtomicShader    *m_sh_grid;

    VertexArray     *m_va_cell;
    Buffer          *m_vbo_cell_pos;
    Buffer          *m_vbo_cell_dist;
    AtomicShader    *m_sh_cell;

public:
    PassForwardShading_DistanceField();
    void beforeDraw();
    void draw();
};


} // namespace atm
#endif // atm_Graphics_Renderer_ForwardShading_h
