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
            atmECall(addPositionXCP)
            atmECall(setPositionXCP)
            atmECall(erasePositionXCP)
            atmECall(clearPositionXCP)
            atmECall(addPositionYCP)
            atmECall(setPositionYCP)
            atmECall(erasePositionYCP)
            atmECall(clearPositionYCP)
            atmECall(addRotationCP)
            atmECall(setRotationCP)
            atmECall(eraseRotationCP)
            atmECall(clearRotationCP)
        )
        atmECallSuper(super)
        atmECallSuper(transform)
    )

    atmJsonizeBlock(
        atmJsonizeSuper(transform)
        atmJsonizeCall(addPositionXCP)
        atmJsonizeCall(setPositionXCP)
        atmJsonizeCall(erasePositionXCP)
        atmJsonizeCall(clearPositionXCP)
        atmJsonizeCall(addPositionYCP)
        atmJsonizeCall(setPositionYCP)
        atmJsonizeCall(erasePositionYCP)
        atmJsonizeCall(clearPositionYCP)
        atmJsonizeCall(addRotationCP)
        atmJsonizeCall(setRotationCP)
        atmJsonizeCall(eraseRotationCP)
        atmJsonizeCall(clearRotationCP)
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
        vec3 pos(Interpolate(m_posxcp, m_time), Interpolate(m_posycp, m_time), 0.0f );
        setPosition(-pos);
        setRotate(Interpolate(m_rotcp, m_time));
        updateTransformMatrix();
    }

    void asyncupdate(float32 dt) override
    {
    }


    void addPositionXCP(const ControlPoint &v)          { m_posxcp.addCP(v);   }
    void setPositionXCP(uint32 i, const ControlPoint &v){ m_posxcp.setCP(i,v); }
    void erasePositionXCP(uint32 i)                     { m_posxcp.eraseCP(i); }
    void clearPositionXCP()                             { m_posxcp.clear();    }

    void addPositionYCP(const ControlPoint &v)          { m_posycp.addCP(v);   }
    void setPositionYCP(uint32 i, const ControlPoint &v){ m_posycp.setCP(i,v); }
    void erasePositionYCP(uint32 i)                     { m_posycp.eraseCP(i); }
    void clearPositionYCP()                             { m_posycp.clear();    }

    void addRotationCP(const ControlPoint &v)           { m_rotcp.addCP(v);   }
    void setRotationCP(uint32 i, const ControlPoint &v) { m_rotcp.setCP(i,v); }
    void eraseRotationCP(uint32 i)                      { m_rotcp.eraseCP(i); }
    void clearRotationCP()                              { m_rotcp.clear();    }
};
atmImplementEntity(LevelLayer);
atmExportClass(LevelLayer);

} // namespace atm
