#include "stdafx.h"
#include "Game/Entity/EntityCommon.h"
#include "Game/Entity/Level.h"

namespace atm {


class dpPatch GroundBlock : public Unbreakable<Entity_Orientation>
{
typedef Unbreakable<Entity_Orientation>  super;
private:
    istSerializeBlock(
        istSerializeBase(super)
    )

public:
    atmECallBlock(
        atmECallSuper(super)
    )

public:
    GroundBlock()
    {
        wdmScope(
        wdmString path = wdmFormat("Level/GroundBlock/0x%p", this);
        super::addDebugNodes(path);
        )
    }

    ~GroundBlock()
    {
        wdmEraseNode(wdmFormat("Level/GroundBlock/0x%p", this));
    }

    virtual void initialize() override
    {
        super::initialize();
        setPivot(vec3(-0.2f, 0.0f, -0.1f));

        initializeCollision(getHandle());
        setCollisionShape(CS_Box);
        setCollisionFlags(CF_Sender|CF_SPH_Sender);

        setModel(PSET_CUBE_MEDIUM);
    }
};
atmImplementEntity(GroundBlock);
atmExportClass(GroundBlock);



class dpPatch FluidFilter : public Unbreakable<Entity_Orientation>
{
typedef Unbreakable<Entity_Orientation>  super;
private:
    istSerializeBlock(
        istSerializeBase(super)
        )

public:
    atmECallBlock(
        atmECallSuper(super)
        )

public:
    FluidFilter()
    {
        wdmScope(
        wdmString path = wdmFormat("Level/GroundBlock/0x%p", this);
        super::addDebugNodes(path);
        )
    }

    ~FluidFilter()
    {
        wdmEraseNode(wdmFormat("Level/FluidFilter/0x%p", this));
    }

    virtual void initialize() override
    {
        super::initialize();
        setPivot(vec3(-0.2f, 0.0f, -0.1f));

        initializeCollision(getHandle());
        setCollisionShape(CS_Box);
        setCollisionFlags(CF_SPH_Sender);

        setModel(PSET_CUBE_MEDIUM);
    }

    virtual void draw() override
    {
        // todo
    }
};
atmImplementEntity(FluidFilter);
atmExportClass(FluidFilter);



class dpPatch LevelLayer
    : public IEntity
    , public TAttr_TransformMatrix<Attr_Transform>
{
typedef IEntity             super;
typedef TAttr_TransformMatrix<Attr_Transform>   transform;
private:
    vec3 m_scroll;

    istSerializeBlock(
        istSerializeBase(super)
        istSerializeBase(transform)
        istSerialize(m_scroll)
    )

public:
    atmECallBlock(
        atmMethodBlock(
            atmECall(getScroll)
            atmECall(setScroll)
        )
        atmECallSuper(super)
        atmECallSuper(transform)
    )

public:
    LevelLayer()
    {
        wdmScope(
        wdmString path = wdmFormat("Level/LevelLayer/0x%p", this);
        transform::addDebugNodes(path);
        wdmAddNode(path+"/m_scrool", &m_scroll);
        )
    }

    ~LevelLayer()
    {
        wdmEraseNode(wdmFormat("Level/LevelLayer/0x%p", this));
    }

    const vec3& getScroll() const { return m_scroll; }
    void setScroll(const vec3 &v) { m_scroll=v; }

    virtual void initialize() override
    {
    }

    virtual void update(float32 dt) override
    {
        // 子が参照するので asyncupdate ではダメ
        setPosition(getPosition()+getScroll());
        updateTransformMatrix();
    }

    virtual void asyncupdate(float32 dt) override
    {
    }
};
atmImplementEntity(LevelLayer);
atmExportClass(LevelLayer);

} // namespace atm
