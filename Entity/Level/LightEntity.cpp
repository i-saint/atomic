#include "entityPCH.h"
#include "Entity/EntityCommon.h"
#include "Entity/Level.h"

namespace atm {


class LightEntityBase : public EntityWithParent
{
typedef EntityWithParent super;
private:
    vec4 m_difuse;
    vec4 m_ambient;

    istSerializeBlock(
        istSerializeBase(super)
        istSerialize(m_difuse)
        istSerialize(m_ambient)
    )

public:
    atmECallBlock(
        atmMethodBlock(
            atmECall(getDiffuse)
            atmECall(setDiffuse)
            atmECall(getAmbient)
            atmECall(setAmbient)
        )
        atmECallSuper(super)
    )

    wdmScope(
    void addDebugNodes(const wdmString &path)
    {
        super::addDebugNodes(path);
        wdmAddNode(path+"/m_difuse", &m_difuse);
        wdmAddNode(path+"/m_ambient", &m_ambient);
    }
    )

public:
    void update(float32 dt) override
    {
        super::update(dt);
        if(isParentDead()) {
            atmDeleteEntity(getHandle());
            return;
        }
    }

    void asyncupdate(float32 dt) override
    {
        super::asyncupdate(dt);
        transform::updateTransformMatrix();
    }

    const vec4& getDiffuse() const { return m_difuse; }
    const vec4& getAmbient() const { return m_ambient; }
    void setDiffuse(const vec4 &v) { m_difuse=v; }
    void setAmbient(const vec4 &v) { m_ambient=v; }
};


class PointLightEntity : public LightEntityBase
{
typedef LightEntityBase super;
private:
    float32 m_radius;

    istSerializeBlock(
        istSerializeBase(super)
        istSerialize(m_radius)
    )

public:
    atmECallBlock(
        atmMethodBlock(
            atmECall(getRadius)
            atmECall(setRadius)
        )
        atmECallSuper(super)
    )

public:
    PointLightEntity() : m_radius(0.5f)
    {
        wdmScope(
        wdmString path = wdmFormat("Level/PointLight/0x%p", this);
        super::addDebugNodes(path);
        wdmAddNode(path+"/m_radius", &m_radius, 0.0f, 3.0f);
        )
    }

    ~PointLightEntity()
    {
        wdmEraseNode(wdmFormat("Level/PointLight/0x%p", this));
    }

    float32 getRadius() const { return m_radius; }
    void setRadius(float32 v) { m_radius=v; }

    void draw() override
    {
        PointLight l;
        l.setPosition(getPositionAbs());
        l.setRadius(getRadius());
        l.setColor(getDiffuse());
        atmGetLightPass()->addLight(l);
    }
};
atmImplementEntity(PointLightEntity);
atmExportClass(PointLightEntity);


class DirectionalLightEntity : public LightEntityBase
{
typedef LightEntityBase super;
private:
    vec3 m_direction;

    istSerializeBlock(
        istSerializeBase(super)
        istSerialize(m_direction)
    )

public:
    atmECallBlock(
        atmMethodBlock(
            atmECall(getDirection)
            atmECall(setDirection)
        )
        atmECallSuper(super)
    )

public:
    DirectionalLightEntity()
    {
        wdmScope(
        wdmString path = wdmFormat("Level/DirectionalLight/0x%p", this);
        super::addDebugNodes(path);
        /*wdmAddNode(path+"/m_direction", this, &DirectionalLightEntity::getDirection, &DirectionalLightEntity::setDirection, -1.0f, 1.0f);*/
        )
    }

    ~DirectionalLightEntity()
    {
        wdmEraseNode(wdmFormat("Level/DirectionalLight/0x%p", this));
    }

    const vec3& getDirection() const { return m_direction; }
    void setDirection(const vec3 &v) { m_direction=glm::normalize(v); }

    void draw() override
    {
        DirectionalLight dl;
        dl.setDirection(getDirection());
        dl.setDiffuse(getDiffuse());
        dl.setAmbient(getAmbient());
        atmGetLightPass()->addLight(dl);
    }
};
atmImplementEntity(DirectionalLightEntity);
atmExportClass(DirectionalLightEntity);

// todo:
class BoxLightEntity;
class CylinderLightEntity;
class SpotLightEntity;


} // namespace atm
