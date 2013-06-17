#ifndef atm_Graphics_Renderer_ForwardShading_h
#define atm_Graphics_Renderer_ForwardShading_h
namespace atm {


class PassForward_DistanceField : public IRenderer
{
private:
    VertexArray     *m_va_grid;
    AtomicShader    *m_sh_grid;

    VertexArray     *m_va_cell;
    Buffer          *m_vbo_cell_pos;
    Buffer          *m_vbo_cell_dist;
    AtomicShader    *m_sh_cell;

public:
    PassForward_DistanceField();
    void beforeDraw();
    void draw();
};


class dpPatch PassForward_Generic : public IRenderer
{
public:
    PassForward_Generic();
    ~PassForward_Generic();
    void beforeDraw();
    void draw();

    void drawModel(SH_RID shader, MODEL_RID model, const mat4 &matrix);

private:
    typedef std::vector<mat4> mat_cont;
    typedef std::map<MODEL_RID, mat_cont> model_mat_cont;
    typedef std::map<SH_RID, model_mat_cont> sh_model_mat_cont;
    sh_model_mat_cont m_commands;
};

class dpPatch PassForward_BackGround : public IRenderer
{
public:
    PassForward_BackGround();
    ~PassForward_BackGround();
    void beforeDraw();
    void draw();

private:
    int32 m_shader;
};


} // namespace atm
#endif // atm_Graphics_Renderer_ForwardShading_h
