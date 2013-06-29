#ifndef atm_Game_Entity_Level_h
#define atm_Game_Entity_Level_h
namespace atm {

template<class Transform>
class Unbreakable : public EntityTemplate<Transform>
{
typedef EntityTemplate<Transform> super;
private:
    istSerializeBlock(
        istSerializeBase(super)
    )

public:
    atmECallBlock(
        atmECallSuper(super)
    )

public:
    Unbreakable()
    {
    }

    wdmScope(
    void addDebugNode(const wdmString &path)
    {
        super::addDebugNodes(path);
    }
    )

    void initialize() override
    {
        super::initialize();
        setGlowColor(vec4(0.4f));
        setDiffuseColor(vec4(vec3(0.5f), 50.0f));
    }

    void update(float32 dt) override
    {
        super::update(dt);
    }

    void asyncupdate(float32 dt) override
    {
        super::asyncupdate(dt);
        transform::updateTransformMatrix();
        collision::updateCollisionByParticleSet(getModel(), getTransform());
        bloodstain::updateBloodstain(dt);
    }

    void draw() override
    {
        PSetInstance inst;
        inst.diffuse = getDiffuseColor();
        inst.glow = getGlowColor();
        inst.flash = vec4();
        inst.elapsed = 1000.0f;
        inst.appear_radius = 10000.0f;
        inst.translate = getTransform();
        atmGetFluidPass()->addPSetInstance(getModel(), inst);
        atmGetBloodStainPass()->addBloodstainParticles(getTransform(), getBloodStainParticles(), getNumBloodstainParticles());
    }

    void eventFluid(const FluidMessage *m) override
    {
        addBloodstain(getInverseTransform(), (vec4&)m->position);
    }
};

} // namespace atm
#endif // atm_Game_Entity_Level_h
