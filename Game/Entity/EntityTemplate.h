#ifndef atm_Game_Entity_EntityTemplate_h
#define atm_Game_Entity_EntityTemplate_h
#include "EntityAttributes.h"
#include "EntityTransform.h"


namespace atm {

struct Entity_Orientation
{
    typedef TAttr_TransformMatrixI<Attr_Orientation> transform;
    typedef Attr_ParticleSet    model;
    typedef Attr_Collision      collision;
    typedef Attr_Bloodstain     bloodstain;
    typedef Attr_MessageHandler mhandler;
};

struct Entity_AxisRotationI
{
    typedef TAttr_TransformMatrixI< TAttr_RotateSpeed<Attr_DoubleAxisRotation> > transform;
    typedef Attr_ParticleSet    model;
    typedef Attr_Collision      collision;
    typedef Attr_Bloodstain     bloodstain;
    typedef Attr_MessageHandler mhandler;
};

struct Entity_AxisRotation
{
    typedef TAttr_TransformMatrix< TAttr_RotateSpeed<Attr_DoubleAxisRotation> > transform;
    typedef Attr_ParticleSet    model;
    typedef Attr_Collision      collision;
    typedef Attr_Bloodstain     bloodstain;
    typedef Attr_MessageHandler mhandler;
};

struct Entity_Translate
{
    typedef TAttr_TransformMatrix<Attr_Translate> transform;
    typedef Attr_ParticleSet    model;
    typedef Attr_Collision      collision;
    typedef Attr_Bloodstain     bloodstain;
    typedef Attr_MessageHandler mhandler;
};

template<class Attributes>
class dpPatch EntityTemplate
    : public IEntity
    , public Attributes::transform
    , public Attributes::model
    , public Attributes::collision
    , public Attributes::bloodstain
    , public Attributes::mhandler
    , public Attributes
{
private:
    typedef IEntity super;
    istSerializeBlock(
        istSerializeBase(super)
        istSerializeBase(transform)
        istSerializeBase(model)
        istSerializeBase(collision)
        istSerializeBase(bloodstain)
        istSerializeBase(mhandler)
        )

public:
    atmECallBlock(
        atmECallSuper(super)
        atmECallSuper(transform)
        atmECallSuper(model)
        atmECallSuper(collision)
        atmECallSuper(bloodstain)
        atmECallSuper(mhandler)
        )

        wdmScope(
        void addDebugNodes(const wdmString &path)
    {
        transform::addDebugNodes(path);
        model::addDebugNodes(path);
        collision::addDebugNodes(path);
        bloodstain::addDebugNodes(path);
        mhandler::addDebugNodes(path);
    }
    )

        virtual void update(float32 dt) {}
    virtual void asyncupdate(float32 dt) {}
};

} // namespace atm

#endif // atm_Game_Entity_EntityTemplate_h
