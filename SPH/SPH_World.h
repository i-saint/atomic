
namespace atomic {

class SPHWorld
{
private:
    sphFluidParticle *m_fluid_particles[2];

public:
    SPHWorld();
    ~SPHWorld();

    void serialize(Serializer& s) const;
    void deserialize(Deserializer& s);

    void frameBegin();
    void update(float32 dt);
    void asyncupdate(float32 dt);
    void draw();
    void frameEnd();

    // rigid/force は毎フレームクリアされるので、毎フレーム突っ込む必要がある
    void addRigid(const sphRigidPlane &s);
    void addRigid(const sphRigidSphere &s);
    void addRigid(const sphRigidBox &s);
    void addForce(const sphForcePointGravity &v);
    void addFluid(const sphFluidParticle *particles, uint32 num);
    void addFluid(PSET_RID psid, const mat4 &t);
};

} // namespace atomic
