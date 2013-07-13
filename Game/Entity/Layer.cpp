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
    ControlPoints m_poscp;
    ControlPoints m_oricp;
    float32 m_time;

    istSerializeBlock(
        istSerializeBase(super)
        istSerializeBase(transform)
        istSerialize(m_poscp)
        istSerialize(m_oricp)
        istSerialize(m_time)
    )

public:
    atmECallBlock(
        atmMethodBlock(
            atmECall(addPotisionCP)
            atmECall(erasePotisionCP)
            atmECall(setPotisionCP)
            atmECall(addRotationCP)
            atmECall(eraseRotationCP)
            atmECall(setRotationCP)
        )
        atmECallSuper(super)
        atmECallSuper(transform)
    )

    atmJsonizeBlock(
        atmJsonizeSuper(transform)
        atmJsonizeCall(addPotisionCP)
        atmJsonizeCall(erasePotisionCP)
        atmJsonizeCall(setPotisionCP)
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
        setPosition(-computeInterpolation(m_poscp, m_time));
        setRotate(computeInterpolation(m_oricp, m_time).x);
        updateTransformMatrix();
    }

    void asyncupdate(float32 dt) override
    {
    }


    void addPotisionCP(const ControlPoint &v)
    {
        auto i = stl::lower_bound(m_poscp.begin(), m_poscp.end(), v);
        m_poscp.insert(i, v);
    }
    void erasePotisionCP(uint32 i)
    {
        if(i<m_poscp.size()) {
            m_poscp.erase(m_poscp.begin()+i);
        }
    }
    void setPotisionCP(uint32 i, const ControlPoint &v)
    {
        if(i<m_poscp.size()) {
            m_poscp[i] = v;
            stl::sort(m_poscp.begin(), m_poscp.end());
        }
    }

    void addRotationCP(const ControlPoint &v)
    {
        auto i = stl::lower_bound(m_oricp.begin(), m_oricp.end(), v);
        m_oricp.insert(i, v);
    }
    void eraseRotationCP(uint32 i)
    {
        if(i<m_oricp.size()) {
            m_oricp.erase(m_oricp.begin()+i);
        }
    }
    void setRotationCP(uint32 i, const ControlPoint &v)
    {
        if(i<m_oricp.size()) {
            m_oricp[i] = v;
            stl::sort(m_oricp.begin(), m_oricp.end());
        }
    }

    vec3 computeInterpolation(ControlPoints &cps, float32 time) const
    {
        vec3 r;
        if(cps.empty()) {}
        else if(time<=cps.front().time){ r=cps.front().pos; }
        else if(time>=cps.back().time) { r=cps.back().pos; }
        else {
            auto p2 = stl::lower_bound(cps.begin(), cps.end(), time,
                [&](const ControlPoint &v, float32 t){ return v.time<t; });
            auto p1 = p2-1;

            switch(p1->transition) {
            case atmE_Linear:
                {
                    vec3 diff = p2->pos - p1->pos;
                    float32 t = (time - p1->time) / (p2->time-p1->time);
                    r = p1->pos + (diff*t);
                }
                break;
            case atmE_Bezier:
                {
                    // todo (if needed)
                }
                break;
            }
        }
        return r;
    }
};
atmImplementEntity(LevelLayer);
atmExportClass(LevelLayer);

} // namespace atm
