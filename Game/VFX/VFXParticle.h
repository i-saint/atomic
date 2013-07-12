#ifndef atm_Game_VFX_VFXParticle_h
#define atm_Game_VFX_VFXParticle_h
namespace atm {

struct istAlign(16) VFXScintillaParticleData
{
    simdvec4 position;
    simdvec4 velosity;
    simdvec4 color;
    simdvec4 glow;
    float32 scale;
    float32 time;
    float32 pad[2];
};
atmSerializeRaw(VFXScintillaParticleData);
istStaticAssert(sizeof(VFXScintillaParticleData)%16==0);

class VFXScintilla : public IVFXComponent
{
typedef IVFXComponent super;
private:
    typedef VFXScintillaParticleData VFXData;
    typedef ist::raw_vector<VFXScintillaParticleData> VFXDataCont;
    typedef ist::raw_vector<SingleParticle> DrawDataCont;

    VFXDataCont m_vfxdata;
    DrawDataCont m_drawdata; // serialize •s—v

    istSerializeBlock(
        istSerializeBase(super)
        istSerialize(m_vfxdata)
    )

public:
    void frameBegin() override;
    void update( float32 dt ) override;
    void asyncupdate( float32 dt ) override;
    void draw() override;
    void frameEnd() override;

    void addData( const VFXScintillaSpawnData &spawn );
};

} // namespace atm
#endif // atm_Game_VFX_VFXParticle_h
