#include "stdafx.h"
#include "Game/Entity/EntityCommon.h"
#include "Game/Entity/Level.h"

namespace atm {

class dpPatch LevelLayer
    : public IEntity
    , public TAttr_TransformMatrixI<Attr_Transform>
{
typedef IEntity             super;
typedef TAttr_TransformMatrixI<Attr_Transform>   transform;
private:
    typedef ist::raw_vector<ControlPoint> ControlPoints;
    ControlPoints m_points;
    float32 m_time;

    istSerializeBlock(
        istSerializeBase(super)
        istSerializeBase(transform)
        istSerialize(m_points)
        istSerialize(m_time)
    )

public:
    atmECallBlock(
        atmMethodBlock(
            atmECall(addControlPoint)
            atmECall(eraseControlPoint)
            atmECall(setControlPoint)
        )
        atmECallSuper(super)
        atmECallSuper(transform)
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
        setPosition(computePosition(m_time));
        updateTransformMatrix();
    }

    void asyncupdate(float32 dt) override
    {
    }


    void addControlPoint(const ControlPoint &v)
    {
        auto i = stl::lower_bound(m_points.begin(), m_points.end(), v);
        m_points.insert(i, v);
    }

    void eraseControlPoint(size_t i)
    {
        if(i<m_points.size()) {
            m_points.erase(m_points.begin()+i);
        }
    }

    void setControlPoint(size_t i, const ControlPoint &v)
    {
        if(i<m_points.size()) {
            m_points[i] = v;
            stl::sort(m_points.begin(), m_points.end());
        }
    }

    vec3 computePosition(float32 time) const
    {
        vec3 r;
        if(m_points.empty()) {}
        else if(time<=m_points.front().time){ r=m_points.front().pos; }
        else if(time>=m_points.back().time) { r=m_points.back().pos; }
        else {
            auto p1 = stl::lower_bound(m_points.begin(), m_points.end(), time,
                [&](const ControlPoint &v, float32 t){ return v.time<t; });
            auto p2 = p1+1;
            switch(p1->transition) {
            case atmE_Linear:
                {
                    vec3 diff = p2->pos - p1->pos;
                    float32 t = (time - p1->time) / (p2->time-p1->time);
                    r = p1->pos + (diff*t);
                }
                break;;
            case atmE_Bezier:
                {
                    // todo (if needed)
                }
                break;
            }
        }
        return -r;
    }
};
atmImplementEntity(LevelLayer);
atmExportClass(LevelLayer);

} // namespace atm
