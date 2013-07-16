#include "stdafx.h"
#include "Game/Entity/EntityCommon.h"
#include "Game/Entity/Level.h"

namespace atm {

class LevelLayer
    : public IEntity
    , public TAttr_TransformMatrixI<Attr_Transform>
{
typedef IEntity super;
typedef TAttr_TransformMatrixI<Attr_Transform> transform;
private:
    ControlPoints m_posxcp;
    ControlPoints m_posycp;
    ControlPoints m_rotcp;
    float32 m_time;

    istSerializeBlock(
        istSerializeBase(super)
        istSerializeBase(transform)
        istSerialize(m_posxcp)
        istSerialize(m_posycp)
        istSerialize(m_rotcp)
        istSerialize(m_time)
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
        )
        atmECallSuper(super)
        atmECallSuper(transform)
    )

    atmJsonizeBlock(
        atmJsonizeSuper(transform)
        atmJsonizeCurve2D(m_pos, addPositionXCP,clearPositionXCP,m_posxcp, addPositionYCP,clearPositionYCP,m_posycp)
        atmJsonizeCurve1D(m_rotcp,addPositionXCP,clearPositionXCP)
    )

public:
    LevelLayer() : m_time(0.0f)
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


    void initialize() override
    {
    }

    void update(float32 dt) override
    {
        m_time += dt;

        // 子が参照するので asyncupdate ではダメ
        vec3 pos(m_posxcp.computeValue(m_time), m_posycp.computeValue(m_time), 0.0f );
        setPosition(pos);
        setRotate(m_rotcp.computeValue(m_time));
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
