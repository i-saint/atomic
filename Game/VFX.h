#ifndef atm_Game_VFX_h
#define atm_Game_VFX_h
namespace atm {

struct VFXScintillaSpawnData
{
    vec4 position;
    vec4 color;
    vec4 glow;
    vec4 velosity;
    float32 size;
    float32 lifetime;
    float32 scatter_radius;
    float32 diffuse_strength;
    uint32 num_particles;

    VFXScintillaSpawnData()
        : size(0.01f), lifetime(0.0f), scatter_radius(0.02f), diffuse_strength(0.01f), num_particles(0) {}
};

struct VFXScintillaParticleData
{
    vec4 position;
    vec4 color;
    vec4 glow;
    vec4 velosity;
    float32 frame;
    float32 size;
    CollisionHandle collision;

private:
    istSerializeBlock(
        istSerialize(position)
        istSerialize(color)
        istSerialize(glow)
        istSerialize(velosity)
        istSerialize(frame)
        istSerialize(size)
        istSerialize(collision)
        )

public:
    VFXScintillaParticleData() : frame(0.0f), size(0.01f), collision(0) {}
};
typedef ist::vector<VFXScintillaParticleData> VFXScintillaDataCont;


class IVFXComponent
{
    istSerializeBlock(
        )
public:
    virtual ~IVFXComponent() {}
    virtual void frameBegin()=0;
    virtual void update(float32 dt)=0;
    virtual void asyncupdate(float32 dt)=0;
    virtual void draw()=0;
    virtual void frameEnd()=0;

    // todo: serialize/deserialize
};

class VFXScintilla : public IVFXComponent
{
typedef IVFXComponent super;
public:
    typedef VFXScintillaParticleData ParticleData;
    typedef VFXScintillaDataCont ParticleCont;
private:
    ParticleCont m_particles;

    istSerializeBlock(
        istSerializeBase(super)
        istSerialize(m_particles)
        )

public:
    void frameBegin();
    void update(float32 dt);
    void asyncupdate(float32 dt);
    void draw();
    void frameEnd();

    void addData(const VFXScintillaSpawnData &data);
};


class VFXSet : public IAtomicGameModule
{
typedef IAtomicGameModule super;
private:
    VFXScintilla *m_scintilla;
    ist::vector<IVFXComponent*> m_components;

    // todo
    istSerializeBlock(
        istSerializeBase(super)
        istSerialize(m_scintilla)
        istSerialize(m_components)
        )
public:
    VFXSet();
    ~VFXSet();

    void initialize();
    void frameBegin();
    void update(float32 dt);
    void asyncupdate(float32 dt);
    void draw();
    void frameEnd();

    VFXScintilla* getScintilla() { return m_scintilla; }
};

} // namespace atm
#endif // atm_Game_VFX_h
