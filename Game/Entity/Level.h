#ifndef atm_Game_Entity_Level_h
#define atm_Game_Entity_Level_h
namespace atm {

enum Transition {
    atmE_Linear,
    atmE_Bezier,
};
struct ControlPoint
{
    float32 time;
    vec3 pos;
    Transition transition;

    ControlPoint() : time(0.0f), transition(atmE_Linear) {}
    ControlPoint(float32 t, const vec3 &p, Transition ts=atmE_Linear) : time(t), pos(p), transition(ts) {}
    bool operator<(const ControlPoint &p) const { return time<p.time; }
    void jsonize(stl::string &out)
    {
        out+=ist::Format("%d,%.2f,[%.2f,%.2f,%.2f]", transition, time, pos.x,pos.y,pos.z);
    }
};
atmSerializeRaw(ControlPoint);


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

    void jsonize(stl::string &out) override
    {
        transform::jsonize(out);
    }

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
