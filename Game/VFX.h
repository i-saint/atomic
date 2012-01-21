#ifndef __atomic_Game_VFX__
#define __atomic_Game_VFX__
namespace atomic {

struct __declspec(align(16)) VFXScintillaData
{
    vec4 position;
    vec4 color;
    vec4 velosity;
    float32 frame;
    float32 scale;
    CollisionHandle m_collision;
    VFXScintillaData() : frame(0.0f), scale(0.02f), m_collision(0) {}
};
typedef stl::vector<VFXScintillaData> VFXScintillaDataCont;

struct __declspec(align(16)) VFXExplosionData
{
    vec4 position;
    vec4 color;
    vec4 velosity;
    float32 m_frame;
    float32 m_size;
    VFXExplosionData() : m_frame(0.0f), m_size(0.02f) {}
};
typedef stl::vector<VFXExplosionData> VFXExplosionDataCont;

struct __declspec(align(16)) VFXDebrisData
{
    vec4 position;
    vec4 color;
    vec4 velosity;
    float32 m_frame;
    float32 m_size;
    VFXDebrisData() : m_frame(0.0f), m_size(0.02f) {}
};
typedef stl::vector<VFXDebrisData> VFXDebrisDataCont;



class IVFXComponent
{
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
public:
    typedef VFXScintillaData Data;
    typedef VFXScintillaDataCont DataCont;
private:
    DataCont m_data;

public:
    void frameBegin();
    void update(float32 dt);
    void asyncupdate(float32 dt);
    void draw();
    void frameEnd();

    void addData(const Data *data, uint32 data_num);
};

class VFXExplosion : public IVFXComponent
{
public:
    typedef VFXExplosionData Data;
    typedef VFXExplosionDataCont DataCont;

private:
    DataCont m_data;

public:
    void frameBegin();
    void update(float32 dt);
    void asyncupdate(float32 dt);
    void draw();
    void frameEnd();

    void addData(const Data *data, uint32 data_num);
};

class VFXDebris : public IVFXComponent
{
public:
    typedef VFXDebrisData Data;
    typedef VFXDebrisDataCont DataCont;

private:
    DataCont m_data;

public:
    void frameBegin();
    void update(float32 dt);
    void asyncupdate(float32 dt);
    void draw();
    void frameEnd();

    void addData(const Data *data, uint32 data_num);
};



class VFXSet : public IAtomicGameModule
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
