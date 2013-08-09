#ifndef atm_Game_Entity_EntityTemplate_h
#define atm_Game_Entity_EntityTemplate_h
#include "EntityAttributes.h"
#include "EntityTransform.h"


namespace atm {

struct Entity_Direction
{
    typedef TAttr_TransformMatrixI< TAttr_HaveParent<Attr_Direction> > transform;
    typedef Attr_ParticleSet    model;
    typedef Attr_Collision      collision;
    typedef Attr_Bloodstain     bloodstain;
    typedef Attr_MessageHandler mhandler;
    typedef Attr_PastTime       pasttime;
};

struct Entity_AxisRotationI
{
    typedef TAttr_TransformMatrixI< TAttr_RotateSpeed<Attr_DoubleAxisRotation> > transform;
    typedef Attr_ParticleSet    model;
    typedef Attr_Collision      collision;
    typedef Attr_Bloodstain     bloodstain;
    typedef Attr_MessageHandler mhandler;
    typedef Attr_PastTime       pasttime;
};

struct Entity_AxisRotation
{
    typedef TAttr_TransformMatrix< TAttr_RotateSpeed<Attr_DoubleAxisRotation> > transform;
    typedef Attr_ParticleSet    model;
    typedef Attr_Collision      collision;
    typedef Attr_Bloodstain     bloodstain;
    typedef Attr_MessageHandler mhandler;
    typedef Attr_PastTime       pasttime;
};

struct Entity_Translate
{
    typedef TAttr_TransformMatrix<Attr_Position> transform;
    typedef Attr_ParticleSet    model;
    typedef Attr_Collision      collision;
    typedef Attr_Bloodstain     bloodstain;
    typedef Attr_MessageHandler mhandler;
    typedef Attr_PastTime       pasttime;
};

template<class Attributes>
class EntityTemplate
    : public IEntity
    , public Attributes::transform
    , public Attributes::model
    , public Attributes::collision
    , public Attributes::bloodstain
    , public Attributes::mhandler
    , public Attributes::pasttime
    , public Attributes
{
typedef IEntity super;
private:
    istSerializeBlock(
        istSerializeBase(super)
        istSerializeBase(transform)
        istSerializeBase(model)
        istSerializeBase(collision)
        istSerializeBase(bloodstain)
        istSerializeBase(mhandler)
        istSerializeBase(pasttime)
    )

public:
    atmECallBlock(
        atmECallSuper(super)
        atmECallSuper(transform)
        atmECallSuper(model)
        atmECallSuper(collision)
        atmECallSuper(bloodstain)
        atmECallSuper(mhandler)
        atmECallSuper(pasttime)
     )

    wdmScope(
    void addDebugNodes(const wdmString &path)
    {
        transform::addDebugNodes(path);
        model::addDebugNodes(path);
        collision::addDebugNodes(path);
        bloodstain::addDebugNodes(path);
        mhandler::addDebugNodes(path);
        pasttime::addDebugNodes(path);
    }
    )

    atmJsonizeBlock(
        atmJsonizeSuper(transform)
    )

public:
    EntityTemplate()
    {}

    virtual void update(float32 dt)
    {
        pasttime::update(dt);
    }

    virtual void asyncupdate(float32 dt)
    {
    }
};

} // namespace atm

#endif // atm_Game_Entity_EntityTemplate_h
