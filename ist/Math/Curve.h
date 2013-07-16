#ifndef ist_Math_Curve_h
#define ist_Math_Curve_h
#include "../Base/Types.h"
#include "../Base/Serialize.h"
#include "../stdex/ist_vector.h"
#include "Misc.h"

namespace ist {

// T: float,vec[234]
template<class T>
struct TControlPoint
{
    enum Interpolation {
        None,
        Linear,
        Decel,
        Accel,
        Smooth,
        Bezier,
    };
    T value;
    T in, out;
    float32 time;
    Interpolation interp;

    TControlPoint() : time(0.0f), value(0.0f), in(0.0f),out(0.0f), interp(Linear) {}
    TControlPoint(float32 t, T v, Interpolation ts=Linear) : time(t), value(v), in(0.0f),out(0.0f), interp(ts) {}
    TControlPoint(float32 t, T v, T i, T o, Interpolation ts=Linear) : time(t), value(v), in(i),out(o), interp(ts) {}
    bool operator<(const TControlPoint &p) const { return time<p.time; }
};

template<class T>
inline T Interpolate(const TControlPoint<T> &v1, const TControlPoint<T> &v2, float32 time)
{
    typedef TControlPoint<T> PointT;
    float32 u = (time - v1.time) / (v2.time-v1.time);
    T r;
    switch(v1.interp) {
    case PointT::None:   r=v1.value; break;
    case PointT::Linear: r=ist::interpolate_linear(v1.value, v2.value, u); break;
    case PointT::Decel:  r=ist::interpolate_sin90(v1.value, v2.value, u); break;
    case PointT::Accel:  r=ist::interpolate_cos90inv(v1.value, v2.value, u); break;
    case PointT::Smooth: r=ist::interpolate_cos180inv(v1.value, v2.value, u); break;
    case PointT::Bezier: r=ist::interpolate_bezier(v1.value, v1.out, v2.in, v2.value, u); break;
    }
    return r;
}

template<class T>
class TControlPoints : public ist::vector< TControlPoint<T> >
{
public:
    typedef TControlPoint<T> PointT;

    void addCP(const PointT &v)
    {
        auto i = stl::lower_bound(begin(), end(), v);
        insert(i, v);
    }

    void setCP(uint32 i, const PointT &v)
    {
        if(i<size()) {
            (*this)[i] = v;
            stl::sort(begin(), end());
        }
    }

    void eraseCP(uint32 i)
    {
        if(i<size()) {
            erase(begin()+i);
        }
    }

    T computeValue(float32 t)
    {
        float32 r = 0.0f;
        if(empty()) {}
        else if(t<=front().time){ r=front().value; }
        else if(t>=back().time) { r=back().value; }
        else {
            auto p2 = stl::lower_bound(begin(), end(), t, [&](const PointT &v, float32 t){ return v.time<t; });
            auto p1 = p2-1;
            r = Interpolate(*p1, *p2, t);
        }
        return r;
    }
};

typedef TControlPoint<float32>  ControlPoint;
typedef TControlPoints<float32> ControlPoints;
typedef TControlPoint<vec2>     ControlPoint2D;
typedef TControlPoints<vec2>    ControlPoints2D;
typedef TControlPoint<vec3>     ControlPoint3D;
typedef TControlPoints<vec3>    ControlPoints3D;

} // namespace ist

istSerializeRaw(ist::ControlPoint);
istSerializeRaw(ist::ControlPoint2D);
istSerializeRaw(ist::ControlPoint3D);

#endif // ist_Math_Curve_h
