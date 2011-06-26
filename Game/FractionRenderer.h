#ifndef __atomic_FractionRenderer__
#define __atomic_FractionRenderer__

namespace atomic {


class FractionShader
{
private:
    ProgramObject m_shader;
    VertexShader m_vsh;
    FragmentShader m_fsh;
    GLuint m_ipos;

public:
    FractionShader();
    void bind() const;
    void unbind() const;
};

class FractionRenderer
{
private:
    eastl::vector<XMVECTOR> m_pos;
    ModelData m_model;
    ProgramObject m_shader;
    VertexShader m_vsh;
    FragmentShader m_fsh;
    eastl::vector<UniformBufferObject*> m_posbuf;
    GLuint m_ipos;

public:
    FractionRenderer();
    ~FractionRenderer();
    void resizePositin(uint32 size) { m_pos.resize(size); }
    void setPosition(uint32 i, XMVECTOR pos) { m_pos[i]=pos; }
    void draw();
};



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

inline void CreateCubeModel(ModelData& model)
{
    float vertex[24][3];
    float normal[24][3];
    int index[24];

    float n[3];
    float ur[3];
    float bl[3];
    SetFloat3(ur,  3, 3, 3);
    SetFloat3(bl, -3,-3,-3);

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

FractionRenderer::FractionRenderer()
{
    CreateCubeModel(m_model);
    CreateVertexShaderFromFile(m_vsh, "shader/fraction.vsh");
    CreateFragmentShaderFromFile(m_fsh, "shader/fraction.fsh");
    m_shader.initialize(&m_vsh, NULL, &m_fsh);

    m_ipos = m_shader.getUniformBlockIndex("FractionData");
}

FractionRenderer::~FractionRenderer()
{
    for(uint32 i=0; i<m_posbuf.size(); ++i) { EA_DELETE(m_posbuf[i]); }
    m_posbuf.clear();
}

void FractionRenderer::draw()
{
    const uint32 instances_par_batch = 1024;
    const uint32 num_instances = m_pos.size();
    const uint32 num_batch = num_instances/instances_par_batch + (num_instances%instances_par_batch!=0 ? 1 : 0);

    while(m_posbuf.size()<num_batch) {
        UniformBufferObject *uni = EA_NEW(UniformBufferObject) UniformBufferObject();
        uni->allocate(sizeof(float)*4*instances_par_batch, UniformBufferObject::USAGE_STREAM);
        m_posbuf.push_back(uni);
    }
    for(uint32 i=0; i<num_batch; ++i) {
        uint32 n = std::min<uint32>(num_instances-(i*instances_par_batch), instances_par_batch);
        UniformBufferObject *uni = m_posbuf[i];
        void *p = uni->lock(UniformBufferObject::LOCK_WRITE);
        memcpy(p,  &m_pos[i*instances_par_batch], n*sizeof(XMVECTOR));
        uni->unlock();

        uni->bindBase(0);
        m_shader.setUniformBlockBinding(m_ipos, 0);

        m_shader.bind();
        m_model.drawInstanced(n);
        m_shader.unbind();
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