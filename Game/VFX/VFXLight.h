#ifndef atm_Game_VFX_VFXLight_h
#define atm_Game_VFX_VFXLight_h
namespace atm {

typedef VFXLightSpawnData VFXLightData;
atmSerializeRaw(VFXLightData);

class VFXLight : public IVFXComponent
{
typedef IVFXComponent super;
private:
    typedef VFXLightSpawnData SpawnData;
    typedef VFXLightData VFXData;
    typedef ist::raw_vector<VFXData> VFXDataCont;

    VFXDataCont m_vfxdata;

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

    void addData( const SpawnData &spawn );
};

} // namespace atm
#endif // atm_Game_VFX_VFXLight_h
