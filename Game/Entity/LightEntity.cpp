#include "stdafx.h"
#include "Game/Entity/EntityCommon.h"

namespace atm {


template<class Transform>
class dpPatch LightEntityBase
    : public IEntity
    , public Transform
{
typedef IEntity super;
typedef Transform trans;
private:
    vec4 m_difuse;
    vec4 m_ambient;

    istSerializeBlock(
        istSerializeBase(super)
        istSerializeBase(trans)
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
        atmECallSuper(trans)
    )

    wdmScope(
    void addDebugNodes(const wdmString &path)
    {
        trans::addDebugNodes(path);
        wdmAddNode(path+"/m_difuse", &m_difuse);
        wdmAddNode(path+"/m_ambient", &m_ambient);
    }
    )

    const vec4& getDiffuse() const { return m_difuse; }
    void setDiffuse(const vec4 &v) { m_difuse=v; }
    const vec4& getAmbient() const { return m_ambient; }
    void setAmbient(const vec4 &v) { m_ambient=v; }
};


class dpPatch PointLightEntity : public LightEntityBase<Attr_Translate>
{
typedef LightEntityBase<Attr_Translate> super;
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
        l.setPosition(getPosition());
        l.setRadius(getRadius());
        l.setColor(getDiffuse());
        atmGetLightPass()->addLight(l);
    }
};
atmImplementEntity(PointLightEntity);
atmExportClass(PointLightEntity);


class dpPatch DirectionalLightEntity : public LightEntityBase<Attr_Translate>
{
typedef LightEntityBase<Attr_Translate> super;
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
        wdmAddNode(path+"/m_direction", this, &DirectionalLightEntity::getDirection, &DirectionalLightEntity::setDirection, -1.0f, 1.0f);
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
class dpPatch BoxLightEntity;
class dpPatch CylinderLightEntity;
class dpPatch SpotLightEntity;


} // namespace atm
