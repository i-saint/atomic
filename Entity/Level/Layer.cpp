#include "entityPCH.h"
#include "Entity/EntityCommon.h"
#include "Entity/Level.h"

namespace atm {

class LevelLayer
    : public IEntity
    , public TAttr_TransformMatrixI< TAttr_HaveParent<Attr_Transform> >
    , public Attr_PastTime
{
typedef IEntity super;
typedef TAttr_TransformMatrixI< TAttr_HaveParent<Attr_Transform> > transform;
typedef Attr_PastTime pasttime;
private:
    ControlPoints m_posxcp;
    ControlPoints m_posycp;
    ControlPoints m_rotcp;
    float32 m_lifetime;

    istSerializeBlock(
        istSerializeBase(super)
        istSerializeBase(transform)
        istSerializeBase(pasttime)
        istSerialize(m_posxcp)
        istSerialize(m_posycp)
        istSerialize(m_rotcp)
        istSerialize(m_lifetime)
    )

public:
    atmECallBlock(
        atmMethodBlock(
            atmECall(setPositionXCP)
            atmECall(addPositionXCP)
            atmECall(clearPositionXCP)
            atmECall(setPositionYCP)
            atmECall(addPositionYCP)
            atmECall(clearPositionYCP)
            atmECall(setRotationCP)
            atmECall(addRotationCP)
            atmECall(clearRotationCP)
            atmECall(getLifeTime)
            atmECall(setLifeTime)
        )
        atmECallSuper(super)
        atmECallSuper(transform)
        atmECallSuper(pasttime)
    )

    atmJsonizeBlock(
        atmJsonizeSuper(transform)
        atmJsonizeCurve2D(m_pos, addPositionXCP,clearPositionXCP,m_posxcp, addPositionYCP,clearPositionYCP,m_posycp)
        atmJsonizeCurve1D(m_rotcp,addPositionXCP,clearPositionXCP)
    )

public:
    LevelLayer() : m_lifetime(0.0f)
    {
        wdmScope(
        wdmString path = wdmFormat("Level/LevelLayer/0x%p", this);
        transform::addDebugNodes(path);
        )
    }

    ~LevelLayer()
    {
        wdmEraseNode(wdmFormat("Level/LevelLayer/0x%p", this));
    }

    float32 getLifeTime() const    { return m_lifetime; }
    void    setLifeTime(float32 v) { m_lifetime=v; }

    void initialize() override
    {
    }

    void update(float32 dt) override
    {
        pasttime::update(dt);
        if(m_lifetime>0.0f && getPastTime()>m_lifetime) {
            atmDeleteEntity(getHandle());
            return;
        }

        // 子が参照するので asyncupdate ではダメ
        float32 t = getPastTime();
        if(!m_posxcp.empty() || !m_posycp.empty()) {
            vec3 pos(m_posxcp.computeValue(t), m_posycp.computeValue(t), 0.0f );
            setPosition(pos);
        }
        if(!m_rotcp.empty()) {
            setRotate(m_rotcp.computeValue(t));
        }
        updateTransformMatrix();
    }

    void asyncupdate(float32 dt) override
    {
    }


    void setPositionXCP(const ControlPoint *v, uint32 n){ m_posxcp.assign(v,v+n);   }
    void addPositionXCP(const ControlPoint &v)          { m_posxcp.addCP(v);        }
    void clearPositionXCP()                             { m_posxcp.clear();         }

    void setPositionYCP(const ControlPoint *v, uint32 n){ m_posycp.assign(v,v+n);   }
    void addPositionYCP(const ControlPoint &v)          { m_posycp.addCP(v);        }
    void clearPositionYCP()                             { m_posycp.clear();         }

    void setRotationCP(const ControlPoint *v, uint32 n) { m_rotcp.assign(v,v+n);    }
    void addRotationCP(const ControlPoint &v)           { m_rotcp.addCP(v);         }
    void clearRotationCP()                              { m_rotcp.clear();          }
};
atmImplementEntity(LevelLayer);
atmExportClass(LevelLayer);

} // namespace atm
