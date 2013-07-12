#ifndef atm_Game_VFX_VFXBlur_h
#define atm_Game_VFX_VFXBlur_h
namespace atm {

typedef VFXFeedbackBlurSpawnData VFXFeedbackBlurData;
atmSerializeRaw(VFXFeedbackBlurData);

class VFXFeedbackBlur : public IVFXComponent
{
typedef IVFXComponent super;
private:
    typedef VFXFeedbackBlurSpawnData VFXData;
    typedef ist::raw_vector<VFXFeedbackBlurData> VFXDataCont;

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

    void addData( const VFXFeedbackBlurSpawnData &spawn );
};


} // namespace atm
#endif // atm_Game_VFX_VFXBlur_h
