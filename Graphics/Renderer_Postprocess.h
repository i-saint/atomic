#ifndef atm_Graphics_Renderer_Postprocess_h
#define atm_Graphics_Renderer_Postprocess_h
namespace atm {


class atmAPI PassPostprocess_Microscopic : public IRenderer
{
private:
    RenderTarget    *m_rt_gbuffer;
    VertexArray     *m_va_quad;
    AtomicShader    *m_sh;

public:
    PassPostprocess_Microscopic();
    void beforeDraw();
    void draw();
};

class atmAPI PassPostprocess_FXAA : public IRenderer
{
private:
    VertexArray     *m_va_quad;
    AtomicShader    *m_sh_FXAA_luma;
    AtomicShader    *m_sh_FXAA;
    int32           m_loc_fxaa_param;
    FXAAParams      m_fxaaparams;

public:
    PassPostprocess_FXAA();
    void beforeDraw();
    void draw();
};

class atmAPI PassPostprocess_Bloom : public IRenderer
{
private:
    RenderTarget    *m_rt_gbuffer;
    RenderTarget    *m_rt_gauss0;
    RenderTarget    *m_rt_gauss1;
    VertexArray     *m_va_luminance;
    VertexArray     *m_va_blur;
    VertexArray     *m_va_composite;
    Buffer          *m_ubo_states;
    AtomicShader    *m_sh_luminance;
    AtomicShader    *m_sh_hblur;
    AtomicShader    *m_sh_vblur;
    AtomicShader    *m_sh_composite;

public:
    PassPostprocess_Bloom();
    void beforeDraw();
    void draw();
};

class atmAPI PassPostprocess_Fade : public IRenderer
{
private:
    AtomicShader    *m_sh_fade;
    Buffer          *m_ubo_fade;
    VertexArray     *m_va_quad;
    int32       m_loc_fade_param;
    FadeParams  m_params;

    vec4 m_begin_color;
    vec4 m_end_color;
    float32 m_begin_frame;
    float32 m_end_frame;

public:
    PassPostprocess_Fade();
    void beforeDraw();
    void draw();

    void setColor(const vec4 &v) { m_params.Color=v; }
    void setFade(const vec4 &v, float32 frame);
};

} // namespace atm
#endif // atm_Graphics_Renderer_Postprocess_h
