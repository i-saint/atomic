#include "entityPCH.h"
#include "Entity/EntityCommon.h"
#include "Entity/Enemy.h"
#include "Entity/Routine.h"
#include "Entity/Level.h"

namespace atm {


class LaserTurret : public Breakable<Entity_Direction>
{
typedef Breakable<Entity_Direction> super;
private:
    istSerializeBlock(
        istSerializeBase(super)
    )

public:
};
atmImplementEntity(LaserTurret);
atmExportClass(LaserTurret);


class InvisibleLaserTurret
    : public EntityWithDirection
    , public Attr_PastTime
{
typedef EntityWithDirection super;
typedef Attr_PastTime pasttime;
private:
    LaserHandle     m_laser;
    EntityHandle    m_owner;

    istSerializeBlock(
        istSerializeBase(super)
        istSerializeBase(pasttime)
        istSerialize(m_laser)
        istSerialize(m_owner)
    )

    atmECallBlock(
        atmECallSuper(super)
        atmECallSuper(pasttime)
        atmMethodBlock(
            atmECall(getOwner)
            atmECall(setOwner)
        )
    )

public:
    InvisibleLaserTurret() : m_laser(), m_owner()
    {}

    EntityHandle getOwner() const { return m_owner; }
    void setOwner(EntityHandle v) { m_owner=v; }

    void update(float32 dt) override
    {
        super::update(dt);
        pasttime::update(dt);

        if(m_laser==0) {
            vec3 pos = getPositionAbs();
            vec3 dir = getDirectionAbs();
            m_laser = atmGetBulletModule()->createLaser(pos, dir, m_owner);
        }
        else {
            float32 t = getPastTime();
            vec3 pos = getPositionAbs();
            vec3 dir = getDirectionAbs();
            if(ILaser *l = atmGetBulletModule()->getLaser(m_laser)) {
                l->setPosition(pos);
                l->setDirection(dir);
                if(t>50.0f) {
                    l->fade();
                }
            }
            else {
                atmDeleteEntity(getHandle());
            }
        }
    }
};
atmImplementEntity(InvisibleLaserTurret);
atmExportClass(InvisibleLaserTurret);

} // namespace atm
