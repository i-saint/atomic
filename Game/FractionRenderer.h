#ifndef __atomic_FractionRenderer__
#define __atomic_FractionRenderer__

namespace atomic {

inline void SetFloat3(float (&v)[3], float x, float y, float z)
{
    v[0] = x;
    v[1] = y;
    v[2] = z;
}
inline void SetFloat3(float (&v)[3], float (&s)[3])
{
    v[0] = s[0];
    v[1] = s[1];
    v[2] = s[2];
}
inline void SetFloat3(float (&v)[3], XMVECTOR s)
{
    v[0] = ((float*)&s)[0];
    v[1] = ((float*)&s)[1];
    v[2] = ((float*)&s)[2];
}



inline void CreateSphereModel(ModelData& model, float32 radius)
{
    const float pi = 3.14159f;
    const float radian = pi/180.0f;
    
    const int ydiv = 12;
    const int xzdiv = 24;
    XMVECTOR v[ydiv][xzdiv];
    float n[ydiv][xzdiv][3];
    int index[(ydiv-1)*(xzdiv)*4];

    for(int i=0; i<ydiv; ++i) {
        float ang = ((180.0f/(ydiv-1)*i-90.0f)*radian);
        v[i][0] = XMVectorSet(cos(ang)*radius, sin(ang)*radius, 0, 1.0);
    }
    XMMATRIX mat = XMMatrixIdentity();
    for(int j=0; j<xzdiv; ++j) {
        for(int i=0; i<ydiv; ++i) {
            v[i][j] = XMVector4Transform(v[i][0], mat);
            SetFloat3(n[i][j], XMVector3Normalize(v[i][j]));
        }
        mat = XMMatrixRotationY(360.0f/xzdiv*j*radian);
    }

    int *ci = index;
    for(int i=0; i<ydiv-1; ++i) {
        for(int j=0; j<xzdiv; ++j) {
            ci[0] = xzdiv*(i)  + j;
            ci[1] = xzdiv*(i)  + ((j+1)%xzdiv);
            ci[2] = xzdiv*(i+1)+ ((j+1)%xzdiv);
            ci[3] = xzdiv*(i+1)+ j;
            ci+=4;
        }
    }
    model.setVertex(v, ydiv*xzdiv, ModelData::VTX_FLOAT4, ModelData::USAGE_STATIC);
    model.setNormal(n, ydiv*xzdiv, ModelData::USAGE_STATIC);
    model.setIndex(index, ((ydiv-1)*(xzdiv)*4), ModelData::IDX_INT32, ModelData::PRM_QUADS, ModelData::USAGE_STATIC);

}

inline void CreateCubeModel(ModelData& model, float32 len)
{
    float vertex[24][3];
    float normal[24][3];
    int index[24];

    float n[3];
    float ur[3];
    float bl[3];
    SetFloat3(ur,  len/2.0f, len/2.0f, len/2.0f);
    SetFloat3(bl, -len/2.0f,-len/2.0f,-len/2.0f);

    SetFloat3(n, 1.0f, 0.0f, 0.0f);
    SetFloat3(normal[0], n);
    SetFloat3(normal[1], n);
    SetFloat3(normal[2], n);
    SetFloat3(normal[3], n);
    SetFloat3(vertex[0], ur[0], ur[1], ur[2]);
    SetFloat3(vertex[1], ur[0], bl[1], ur[2]);
    SetFloat3(vertex[2], ur[0], bl[1], bl[2]);
    SetFloat3(vertex[3], ur[0], ur[1], bl[2]);

    SetFloat3(n, -1.0f, 0.0f, 0.0f);
    SetFloat3(normal[4], n);
    SetFloat3(normal[5], n);
    SetFloat3(normal[6], n);
    SetFloat3(normal[7], n);
    SetFloat3(vertex[4], bl[0], ur[1], ur[2]);
    SetFloat3(vertex[5], bl[0], ur[1], bl[2]);
    SetFloat3(vertex[6], bl[0], bl[1], bl[2]);
    SetFloat3(vertex[7], bl[0], bl[1], ur[2]);

    SetFloat3(n, 0.0f, 1.0f, 0.0f);
    SetFloat3(normal[8], n);
    SetFloat3(normal[9], n);
    SetFloat3(normal[10], n);
    SetFloat3(normal[11], n);
    SetFloat3(vertex[8], ur[0], ur[1], ur[2]);
    SetFloat3(vertex[9], ur[0], ur[1], bl[2]);
    SetFloat3(vertex[10], bl[0], ur[1], bl[2]);
    SetFloat3(vertex[11], bl[0], ur[1], ur[2]);

    SetFloat3(n, 0.0f, -1.0f, 0.0f);
    SetFloat3(normal[12], n);
    SetFloat3(normal[13], n);
    SetFloat3(normal[14], n);
    SetFloat3(normal[15], n);
    SetFloat3(vertex[12], ur[0], bl[1], ur[2]);
    SetFloat3(vertex[13], bl[0], bl[1], ur[2]);
    SetFloat3(vertex[14], bl[0], bl[1], bl[2]);
    SetFloat3(vertex[15], ur[0], bl[1], bl[2]);

    SetFloat3(n, 0.0f, 0.0f, 1.0f);
    SetFloat3(normal[16], n);
    SetFloat3(normal[17], n);
    SetFloat3(normal[18], n);
    SetFloat3(normal[19], n);
    SetFloat3(vertex[16], ur[0], ur[1], ur[2]);
    SetFloat3(vertex[17], bl[0], ur[1], ur[2]);
    SetFloat3(vertex[18], bl[0], bl[1], ur[2]);
    SetFloat3(vertex[19], ur[0], bl[1], ur[2]);

    SetFloat3(n, 0.0f, 0.0f, -1.0f);
    SetFloat3(normal[20], n);
    SetFloat3(normal[21], n);
    SetFloat3(normal[22], n);
    SetFloat3(normal[23], n);
    SetFloat3(vertex[20], ur[0], ur[1], bl[2]);
    SetFloat3(vertex[21], ur[0], bl[1], bl[2]);
    SetFloat3(vertex[22], bl[0], bl[1], bl[2]);
    SetFloat3(vertex[23], bl[0], ur[1], bl[2]);

    for(size_t i=0; i<24; ++i) {
        index[i] = i;
    }

    model.setVertex(vertex, 24, ModelData::VTX_FLOAT3, ModelData::USAGE_STATIC);
    model.setNormal(normal, 24, ModelData::USAGE_STATIC);
    model.setIndex(index, 24, ModelData::IDX_INT32, ModelData::PRM_QUADS, ModelData::USAGE_STATIC);
}

void DrawScreen(float32 min_tx, float32 min_ty, float32 max_tx, float32 max_ty)
{
    OrthographicCamera cam;
    cam.setScreen(0.0f, 1.0f, 0.0f, 1.0f);
    cam.bind();

    float32 min_x = 0.0f;
    float32 min_y = 0.0f;
    float32 max_x = 1.0f;
    float32 max_y = 1.0f;

    glBegin(GL_QUADS);
    glTexCoord2f(min_tx, min_ty);
    glVertex2f(min_x, min_y);
    glTexCoord2f(max_tx, min_ty);
    glVertex2f(max_x, min_y);
    glTexCoord2f(max_tx, max_ty);
    glVertex2f(max_x, max_y);
    glTexCoord2f(min_tx, max_ty);
    glVertex2f(min_x, max_y);
    glEnd();
}


uint32 CalcFrameBufferWidth()
{
    uint32 r = 256;
    uint32 window_width = GetWindowWidth();
    while(r < window_width) {
        r *= 2;
    }
    return r;
}

uint32 CalcFrameBufferHeight()
{
    uint32 r = 256;
    uint32 window_height = GetWindowHeight();
    while(r < window_height) {
        r *= 2;
    }
    return r;
}


class FractionShader
{
private:
    ProgramObject m_shader;
    VertexShader m_vsh;
    FragmentShader m_fsh;
    GLuint m_ipos;

    Color4DepthBuffer m_render_target;

public:
    FractionShader();
    void bind() const;
    void unbind() const;
};

class FractionRenderer
{
public:
    static const uint32 RENDERBUFFER_SIZE = 1024;

private:
    stl::vector<XMVECTOR> m_fraction_pos;
    stl::vector<XMVECTOR> m_light_pos;
    ModelData m_fraction_model;
    ModelData m_light_model;

    ProgramObject m_sh_gbuffer;
    VertexShader m_vsh_gbuffer;
    FragmentShader m_fsh_gbuffer;
    UniformBufferObject m_fraction_pos_buf;
    GLuint m_i_fracton_pos;

    ProgramObject m_sh_deferred;
    VertexShader m_vsh_deferred;
    FragmentShader m_fsh_deferred;
    UniformBufferObject m_light_pos_buf;
    GLuint m_i_light_pos;

    ProgramObject m_sh_out;
    VertexShader m_vsh_out;
    FragmentShader m_fsh_out;

    Color3DepthBuffer m_rt_gbuffer;
    ColorDepthBuffer m_rt_deferred;
    //FrameBufferObject m_fbo;
    //Texture2D m_depth;
    //Texture2D m_color[4];

public:
    FractionRenderer();
    ~FractionRenderer();
    void resizePositin(uint32 size) { m_fraction_pos.resize(size); }
    void setPosition(uint32 i, XMVECTOR pos) { m_fraction_pos[i]=pos; }
    void draw();
};




FractionRenderer::FractionRenderer()
{
    uint32 framebuffer_width = CalcFrameBufferWidth();
    uint32 framebuffer_height = CalcFrameBufferHeight();

    CreateCubeModel(m_fraction_model, 6.0f);
    CreateSphereModel(m_light_model, 150.0f);

    CreateVertexShaderFromFile(m_vsh_gbuffer, "shader/gbuffer.vsh");
    CreateFragmentShaderFromFile(m_fsh_gbuffer, "shader/gbuffer.fsh");
    m_sh_gbuffer.initialize(&m_vsh_gbuffer, NULL, &m_fsh_gbuffer);
    m_i_fracton_pos = m_sh_gbuffer.getUniformBlockIndex("FractionData");
    m_fraction_pos_buf.initialize();

    CreateVertexShaderFromFile(m_vsh_deferred, "shader/deferred.vsh");
    CreateFragmentShaderFromFile(m_fsh_deferred, "shader/deferred.fsh");
    m_sh_deferred.initialize(&m_vsh_deferred, NULL, &m_fsh_deferred);
    m_i_light_pos = m_sh_deferred.getUniformBlockIndex("LightData");
    m_light_pos_buf.initialize();

    CreateVertexShaderFromFile(m_vsh_out, "shader/out.vsh");
    CreateFragmentShaderFromFile(m_fsh_out, "shader/out.fsh");
    m_sh_out.initialize(&m_vsh_out, NULL, &m_fsh_out);

    m_rt_gbuffer.initialize(framebuffer_width, framebuffer_height, Color3DepthBuffer::FMT_RGBA_F32);

    m_rt_deferred.setDepthBuffer(m_rt_gbuffer.getDepthBuffer());
    m_rt_deferred.initialize(framebuffer_width, framebuffer_height);
}

FractionRenderer::~FractionRenderer()
{
}

void FractionRenderer::draw()
{
    const PerspectiveCamera *camera = GetCamera();
    float aspect_ratio = camera->getAspect();
    float x_scale = float32(GetWindowWidth())/float32(m_rt_deferred.getWidth());
    float y_scale = float32(GetWindowHeight())/float32(m_rt_deferred.getHeight()) * aspect_ratio;

    // G-buffer pass
    {
        const uint32 instances_par_batch = 1024;
        const uint32 num_instances = m_fraction_pos.size();
        const uint32 num_batch = num_instances/instances_par_batch + (num_instances%instances_par_batch!=0 ? 1 : 0);

        m_fraction_pos_buf.allocate(sizeof(XMVECTOR)*num_instances, UniformBufferObject::USAGE_STREAM, &m_fraction_pos[0]);

        m_rt_gbuffer.bind();
        glClearColor(0.0f,0.0f,0.0f,0.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        camera->bind();
        for(uint32 i=0; i<num_batch; ++i) {
            uint32 n = std::min<uint32>(num_instances-(i*instances_par_batch), instances_par_batch);
            m_fraction_pos_buf.bindRange(i, sizeof(XMVECTOR)*(instances_par_batch*i), sizeof(XMVECTOR)*n);
            m_sh_gbuffer.setUniformBlockBinding(m_i_fracton_pos, i);
            m_sh_gbuffer.bind();
            m_fraction_model.drawInstanced(n);
            m_sh_gbuffer.unbind();
        }
        m_rt_gbuffer.unbind();
    }

    // deferred shading pass
    {
        const uint32 instances_par_batch = 1024;
        const uint32 num_fraction = m_fraction_pos.size();
        m_light_pos.clear();
        for(uint32 i=0; i<num_fraction; i+=200) {
            m_light_pos.push_back(m_fraction_pos[i]);
        }
        const uint32 num_lights = m_light_pos.size();
        const uint32 num_batch = num_lights/instances_par_batch + (num_lights%instances_par_batch!=0 ? 1 : 0);

        m_light_pos_buf.allocate(sizeof(XMVECTOR)*num_lights, UniformBufferObject::USAGE_STREAM, &m_light_pos[0]);

        m_rt_deferred.bind();
        m_sh_deferred.bind();
        m_rt_gbuffer.getColorBuffer(0)->bind(Texture2D::SLOT_0);
        m_rt_gbuffer.getColorBuffer(1)->bind(Texture2D::SLOT_1);
        m_rt_gbuffer.getColorBuffer(2)->bind(Texture2D::SLOT_2);
        m_sh_deferred.setUniform1i(m_sh_deferred.getUniformLocation("ColorBuffer"), Texture2D::SLOT_0);
        m_sh_deferred.setUniform1i(m_sh_deferred.getUniformLocation("NormalBuffer"), Texture2D::SLOT_1);
        m_sh_deferred.setUniform1i(m_sh_deferred.getUniformLocation("PositionBuffer"), Texture2D::SLOT_2);
        m_sh_deferred.setUniform1f(m_sh_deferred.getUniformLocation("AspectRatio"), aspect_ratio);
        m_sh_deferred.setUniform2f(m_sh_deferred.getUniformLocation("TexcoordScale"), x_scale, y_scale);
        glClearColor(0.0f,0.0f,0.0f,1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE);
        glDepthMask(GL_FALSE);
        camera->bind();
        for(uint32 i=0; i<num_batch; ++i) {
            uint32 n = std::min<uint32>(num_lights-(i*instances_par_batch), instances_par_batch);
            m_light_pos_buf.bindRange(i, sizeof(XMVECTOR)*(instances_par_batch*i), sizeof(XMVECTOR)*n);
            m_sh_deferred.setUniformBlockBinding(m_i_fracton_pos, i);
            m_light_model.drawInstanced(n);
        }
        glDepthMask(GL_TRUE);
        glDisable(GL_BLEND);
        m_rt_gbuffer.getColorBuffer(2)->unbind(Texture2D::SLOT_2);
        m_rt_gbuffer.getColorBuffer(1)->unbind(Texture2D::SLOT_1);
        m_rt_gbuffer.getColorBuffer(0)->unbind(Texture2D::SLOT_0);
        m_sh_deferred.unbind();
        m_rt_deferred.unbind();
    }

    // output
    {
        m_rt_deferred.getColorBuffer(0)->bind(Texture2D::SLOT_0);
        m_sh_out.bind();
        m_sh_out.setUniform1i(m_sh_out.getUniformLocation("ColorBuffer"), Texture2D::SLOT_0);
        DrawScreen(0.0f, 0.0f, float(GetWindowWidth())/float32(m_rt_deferred.getWidth()), float(GetWindowHeight())/float32(m_rt_deferred.getHeight()));
        m_sh_out.unbind();
        m_rt_deferred.getColorBuffer(0)->unbind(Texture2D::SLOT_0);
    }


    //for(uint32 i=0; i<num; ++i) {
    //    const float *pos = (float*)&m_pos[i];
    //    glPushMatrix();
    //    glTranslatef(pos[0], pos[1], pos[2]);
    //    m_model.draw();
    //    glPopMatrix();
    //}
}

} // namespace atomic
#endif // __atomic_FractionRenderer__