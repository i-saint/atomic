#ifndef atm_Game_VFX_VFXDistortion_h
#define atm_Game_VFX_VFXDistortion_h
namespace atm {

    typedef VFXShockwaveSpawnData VFXShockwaveData;
    atmSerializeRaw(VFXShockwaveData);

    class VFXShockwave : public IVFXComponent
    {
        typedef IVFXComponent super;
    private:
        typedef VFXShockwaveSpawnData VFXData;
        typedef ist::raw_vector<VFXShockwaveData> VFXDataCont;

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

        void addData( const VFXShockwaveSpawnData &spawn );
    };

} // namespace atm
#endif // atm_Game_VFX_VFXDistortion_h
