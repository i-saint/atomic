#ifndef __atomic_Game_VFX__
#define __atomic_Game_VFX__
namespace atomic {

struct __declspec(align(16)) VFXScintillaData
{
    vec4 position;
    vec4 color;
    vec4 velosity;
    float32 m_frame;
    float32 m_size;
    CollisionHandle m_collision;
    VFXScintillaData() : m_frame(0.0f), m_size(0.0f), m_collision(0) {}
};
typedef stl::vector<VFXScintillaData> VFXScintillaDataCont;

struct __declspec(align(16)) VFXExplosionData
{
    vec4 position;
    vec4 color;
    vec4 velosity;
    float32 m_frame;
    float32 m_size;
    VFXExplosionData() : m_frame(0.0f), m_size(0.0f) {}
};
typedef stl::vector<VFXExplosionData> VFXExplosionDataCont;

struct __declspec(align(16)) VFXDebrisData
{
    vec4 position;
    vec4 color;
    vec4 velosity;
    float32 m_frame;
    float32 m_size;
    VFXDebrisData() : m_frame(0.0f), m_size(0.0f) {}
};
typedef stl::vector<VFXDebrisData> VFXDebrisDataCont;



class IVFXComponent
{
public:
    virtual ~IVFXComponent() {}
    virtual void frameBegin()=0;
    virtual void update(float32 dt)=0;
    virtual void updateEnd()=0;
    virtual void asyncupdate(float32 dt)=0;
    virtual void draw()=0;
    virtual void frameEnd()=0;

    // todo: serialize/deserialize
};

class VFXScintilla : public IVFXComponent
{
private:
    VFXScintillaDataCont m_data;

public:
    void frameBegin();
    void update(float32 dt);
    void updateEnd();
    void asyncupdate(float32 dt);
    void draw();
    void frameEnd();

    void addData(const VFXScintillaData *data, uint32 data_num);
};

class VFXExplosion : public IVFXComponent
{
private:
    VFXExplosionDataCont m_data;

public:
    void frameBegin();
    void update(float32 dt);
    void updateEnd();
    void asyncupdate(float32 dt);
    void draw();
    void frameEnd();

    void addData(const VFXScintillaData *data, uint32 data_num);
};

class VFXDebris : public IVFXComponent
{
private:
    VFXDebrisDataCont m_data;

public:
    void frameBegin();
    void update(float32 dt);
    void updateEnd();
    void asyncupdate(float32 dt);
    void draw();
    void frameEnd();

    void addData(const VFXScintillaData *data, uint32 data_num);
};



class VFXSet : public AtomicGameModule
{
private:
    VFXScintilla *m_scintilla;
    VFXExplosion *m_explosion;
    VFXDebris    *m_debris;
    stl::vector<IVFXComponent*> m_components;

public:
    VFXSet();
    ~VFXSet();

    void frameBegin();
    void update(float32 dt);
    void asyncupdate(float32 dt);
    void draw();
    void frameEnd();

    VFXScintilla* getScintilla() { return m_scintilla; }
    VFXExplosion* getExplosion() { return m_explosion; }
    VFXDebris* getDebris() { return m_debris; }
};

} // namespace atomic
#endif // __atomic_Game_VFX__
