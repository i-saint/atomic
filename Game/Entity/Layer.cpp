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
    typedef ist::raw_vector<ControlPoint> ControlPoints;
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
            atmECall(addPotisionXCP)
            atmECall(erasePotisionXCP)
            atmECall(setPotisionXCP)
            atmECall(addPotisionYCP)
            atmECall(erasePotisionYCP)
            atmECall(setPotisionYCP)
            atmECall(addRotationCP)
            atmECall(eraseRotationCP)
            atmECall(setRotationCP)
        )
        atmECallSuper(super)
        atmECallSuper(transform)
    )

    atmJsonizeBlock(
        atmJsonizeSuper(transform)
        atmJsonizeCall(addPotisionXCP)
        atmJsonizeCall(erasePotisionXCP)
        atmJsonizeCall(setPotisionXCP)
        atmJsonizeCall(addPotisionYCP)
        atmJsonizeCall(erasePotisionYCP)
        atmJsonizeCall(setPotisionYCP)
        atmJsonizeCall(addRotationCP)
        atmJsonizeCall(eraseRotationCP)
        atmJsonizeCall(setRotationCP)
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
        vec3 pos(
            computeInterpolation(m_posxcp, m_time),
            computeInterpolation(m_posycp, m_time),
            0.0f );
        setPosition(-pos);
        setRotate(computeInterpolation(m_rotcp, m_time));
        updateTransformMatrix();
    }

    void asyncupdate(float32 dt) override
    {
    }


    void addPotisionXCP(const ControlPoint &v)
    {
        auto i = stl::lower_bound(m_posxcp.begin(), m_posxcp.end(), v);
        m_posxcp.insert(i, v);
    }
    void erasePotisionXCP(uint32 i)
    {
        if(i<m_posxcp.size()) {
            m_posxcp.erase(m_posxcp.begin()+i);
        }
    }
    void setPotisionXCP(uint32 i, const ControlPoint &v)
    {
        if(i<m_posxcp.size()) {
            m_posxcp[i] = v;
            stl::sort(m_posxcp.begin(), m_posxcp.end());
        }
    }

    void addPotisionYCP(const ControlPoint &v)
    {
        auto i = stl::lower_bound(m_posycp.begin(), m_posycp.end(), v);
        m_posycp.insert(i, v);
    }
    void erasePotisionYCP(uint32 i)
    {
        if(i<m_posycp.size()) {
            m_posycp.erase(m_posycp.begin()+i);
        }
    }
    void setPotisionYCP(uint32 i, const ControlPoint &v)
    {
        if(i<m_posycp.size()) {
            m_posycp[i] = v;
            stl::sort(m_posycp.begin(), m_posycp.end());
        }
    }

    void addRotationCP(const ControlPoint &v)
    {
        auto i = stl::lower_bound(m_rotcp.begin(), m_rotcp.end(), v);
        m_rotcp.insert(i, v);
    }
    void eraseRotationCP(uint32 i)
    {
        if(i<m_rotcp.size()) {
            m_rotcp.erase(m_rotcp.begin()+i);
        }
    }
    void setRotationCP(uint32 i, const ControlPoint &v)
    {
        if(i<m_rotcp.size()) {
            m_rotcp[i] = v;
            stl::sort(m_rotcp.begin(), m_rotcp.end());
        }
    }

    float32 computeInterpolation(ControlPoints &cps, float32 time) const
    {
        float32 r = 0.0f;
        if(cps.empty()) {}
        else if(time<=cps.front().time){ r=cps.front().value; }
        else if(time>=cps.back().time) { r=cps.back().value; }
        else {
            auto p2 = stl::lower_bound(cps.begin(), cps.end(), time,
                [&](const ControlPoint &v, float32 t){ return v.time<t; });
            auto p1 = p2-1;
            r = Interpolate(*p1, *p2, time);
        }
        return r;
    }
};
atmImplementEntity(LevelLayer);
atmExportClass(LevelLayer);

} // namespace atm
