#ifndef atm_Game_Entity_Level_h
#define atm_Game_Entity_Level_h
namespace atm {

template<class Transform>
class EntityWithTransform : public IEntity, public Transform
{
typedef IEntity super;
typedef Transform transform;
private:
    istSerializeBlock(
        istSerializeBase(super)
        istSerializeBase(transform)
    )

public:
    atmECallBlock(
        atmECallSuper(super)
        atmECallSuper(transform)
    )

    atmJsonizeBlock(
        atmJsonizeSuper(transform)
    )
};
typedef EntityWithTransform<Attr_Position> EntityWithPosition;

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

    atmJsonizeBlock(
        atmJsonizeSuper(transform)
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
        collision::updateCollisionByParticleSet(getModel(), getTransformMatrix());
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
        inst.transform = transform::getTransformMatrix();
        inst.rotate = transform::computeRotationMatrix();
        atmGetFluidPass()->addParticlesSolid(getModel(), inst);
        atmGetBloodStainPass()->addBloodstainParticles(getTransformMatrix(), getBloodStainParticles(), getNumBloodstainParticles());
    }

    void eventFluid(const FluidMessage *m) override
    {
        addBloodstain(getInvTransformMatrix(), (vec4&)m->position);
    }
};

} // namespace atm
#endif // atm_Game_Entity_Level_h
