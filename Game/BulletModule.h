#ifndef atm_Game_BulletModule_h
#define atm_Game_BulletModule_h

namespace atm {


class LaserManager;
class BulletManager;
class NeedleManager;

typedef uint32 LaserHandle;

class ILaser
{
    istSerializeBlock()
public:
    virtual ~ILaser() {}
    virtual LaserHandle getHandle() const=0;
    virtual const vec3& getPosition() const=0;
    virtual const vec3& getDirection() const=0;
    virtual void setPosition(const vec3 &v)=0;
    virtual void setDirection(const vec3 &v)=0;
    virtual void fade()=0;
    virtual void kill()=0;
};
class IBulletManager
{
    istSerializeBlock()
public:
    virtual ~IBulletManager() {}
    virtual void frameBegin() {}
    virtual void update(float32 dt) {}
    virtual void asyncupdate(float32 dt) {}
    virtual void draw() {}
    virtual void frameEnd() {}
};


class atmAPI BulletModule : public IAtomicGameModule
{
typedef IAtomicGameModule super;
public:
    BulletModule();
    ~BulletModule();
    void initialize() override;
    void frameBegin() override;
    void update(float32 dt) override;
    void asyncupdate(float32 dt) override;
    void draw() override;
    void frameEnd() override;

    void handleStateQuery(EntitiesQueryContext &ctx);

    void shootBullet(const vec3 &pos, const vec3 &vel, EntityHandle owner);
    LaserHandle createLaser(const vec3 &pos, const vec3 &dir, EntityHandle owner);
    ILaser* getLaser(LaserHandle v);

private:
    typedef stl::vector<IBulletManager*> managers;
    LaserManager    *m_lasers;
    BulletManager   *m_bullets;
    managers        m_managers;

    istSerializeBlockDecl();
};

} // namespace atm
#endif // atm_Game_BulletModule_h
