﻿#ifndef atm_Graphics_Renderer_ForwardShading_h
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


class PassForward_Generic : public IRenderer
{
public:
    PassForward_Generic();
    ~PassForward_Generic();
    void beforeDraw();
    void draw();

    struct InstanceParams {
        mat4 transform;
        vec4 params[4];
    };
    void drawModel(SH_RID shader, MODEL_RID model, const mat4 &transform);
    void drawModel(SH_RID shader, MODEL_RID model, const InstanceParams &params);

private:
    typedef std::vector<InstanceParams> ParamCont;
    typedef std::map<MODEL_RID, ParamCont> ModelParamCont;
    typedef std::map<SH_RID, ModelParamCont> ShModelParamCont;
    ShModelParamCont m_commands;
    ParamCont m_params;
};

class PassForward_Barrier : public IRenderer
{
public:
    PassForward_Barrier();
    ~PassForward_Barrier();
    void beforeDraw();
    void draw();

    void addParticles(PSET_RID psid, const PSetInstance &inst);

private:
    void drawParticles(PSetDrawData &pdd);

    PSetDrawData m_solids;
};

class PassForward_BackGround : public IRenderer
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
