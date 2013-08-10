#include "stdafx.h"
#include "Entity/EntityCommon.h"
#include "Entity/Routine.h"
#include "Entity/Level.h"

namespace atm {

class GateLinkage : public EntityWithPosition
{
typedef EntityWithPosition super;
private:
    EntityHandle m_gear;
    EntityHandle m_block;
    vec3    m_slide_dir;
    float32 m_link_speed;
    float32 m_prev_angle;

    istSerializeBlock(
        istSerializeBase(super)
        istSerialize(m_gear)
        istSerialize(m_block)
        istSerialize(m_slide_dir)
        istSerialize(m_link_speed)
        istSerialize(m_prev_angle)
    )

public:
    atmECallBlock(
        atmMethodBlock(
            atmECall(setGear)
            atmECall(setBlock)
            atmECall(setSlideDir)
            atmECall(setLinkSpeed)
        )
        atmECallSuper(super)
    )

public:
    GateLinkage() : m_gear(), m_block(), m_slide_dir(0.0f,1.0f,0.0f), m_link_speed(0.01f), m_prev_angle(0.0f)
    {
        wdmScope(
        wdmString path = wdmFormat("Level/GateLinkage/0x%p", this);
        super::addDebugNodes(path);
        )
    }

    ~GateLinkage()
    {
        wdmEraseNode(wdmFormat("Level/GateLinkage/0x%p", this));
    }

    void setGear(EntityHandle v)    { m_gear=v; }
    void setBlock(EntityHandle v)   { m_block=v; }
    void setSlideDir(const vec3 &v) { m_slide_dir=v; }
    void setLinkSpeed(float32 v)    { m_link_speed=v; }

    void initialize()
    {
        super::initialize();
    }

    void update(float32 dt) override
    {
        super::update(dt);
        SweepDeadEntities(m_gear);
        SweepDeadEntities(m_block);
        if(!m_gear && !m_block) {
            atmDeleteEntity(getHandle());
            return;
        }

        IEntity *gear = atmGetEntity(m_gear);
        IEntity *block = atmGetEntity(m_block);
        if(gear && block) {
            float32 angle = atmGetProperty(float32, gear, getSpinAngle);
            float32 diff = angle - m_prev_angle;
            m_prev_angle = angle;
            vec3 pos = atmGetProperty(vec3, block, getPosition);
            pos += m_slide_dir*(diff*m_link_speed);
            atmCall(block, setPosition, pos);
            setPosition(atmGetProperty(vec3, block, getPositionAbs)+vec3(0.0f, -0.1f, 0.0f));
        }
    }
};
atmImplementEntity(GateLinkage)
atmExportClass(GateLinkage)


class HingeLinkage : public EntityWithPosition
{
typedef EntityWithPosition super;
private:
    EntityHandle m_gear;
    EntityHandle m_block;
    float32 m_link_speed;
    float32 m_prev_angle;

    istSerializeBlock(
        istSerializeBase(super)
        istSerialize(m_gear)
        istSerialize(m_block)
        istSerialize(m_link_speed)
        istSerialize(m_prev_angle)
    )

public:
    atmECallBlock(
        atmMethodBlock(
            atmECall(setGear)
            atmECall(setBlock)
            atmECall(setLinkSpeed)
        )
        atmECallSuper(super)
    )

public:
    HingeLinkage() : m_gear(), m_block(), m_link_speed(0.01f), m_prev_angle(0.0f)
    {
        wdmScope(
        wdmString path = wdmFormat("Level/HingeLinkage/0x%p", this);
        super::addDebugNodes(path);
        )
    }

    ~HingeLinkage()
    {
        wdmEraseNode(wdmFormat("Level/HingeLinkage/0x%p", this));
    }

    void setGear(EntityHandle v)    { m_gear=v; }
    void setBlock(EntityHandle v)   { m_block=v; }
    void setLinkSpeed(float32 v)    { m_link_speed=v; }

    void initialize()
    {
        super::initialize();
    }

    void update(float32 dt) override
    {
        super::update(dt);
        SweepDeadEntities(m_gear);
        SweepDeadEntities(m_block);
        if(!m_gear && !m_block) {
            atmDeleteEntity(getHandle());
            return;
        }

        IEntity *gear = atmGetEntity(m_gear);
        IEntity *block = atmGetEntity(m_block);
        if(gear && block) {
            float32 angle = atmGetProperty(float32, gear, getSpinAngle);
            float32 diff = angle - m_prev_angle;
            m_prev_angle = angle;
            float32 rot = atmGetProperty(float32, block, getRotate);
            rot += (diff*m_link_speed);
            atmCall(block, setRotate, rot);
            setPosition(atmGetProperty(vec3, block, getPositionAbs)+vec3(0.0f, -0.1f, 0.0f));
        }
    }
};
atmImplementEntity(HingeLinkage)
atmExportClass(HingeLinkage)

} // namespace atm
