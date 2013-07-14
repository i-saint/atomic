#include "istPCH.h"
#include "Curve.h"

namespace ist {

float32 Interpolate(const ControlPoint &v1, const ControlPoint &v2, float32 time)
{
    float32 u = (time - v1.time) / (v2.time-v1.time);
    float32 r;
    switch(v1.interp) {
    case ControlPoint::None:   r=v1.value; break;
    case ControlPoint::Linear: r=ist::interpolate_lerp(v1.value, v2.value, u); break;
    case ControlPoint::Decel:  r=ist::interpolate_sin90(v1.value, v2.value, u); break;
    case ControlPoint::Accel:  r=ist::interpolate_cos90inv(v1.value, v2.value, u); break;
    case ControlPoint::Smooth: r=ist::interpolate_cos180inv(v1.value, v2.value, u); break;
    case ControlPoint::Bezier: r=ist::interpolate_bezier(v1.value, v1.bez_out, v2.bez_in, v2.value, u); break;
    }
    return r;
}


void ControlPoints::addCP(const ControlPoint &v)
{
    auto i = stl::lower_bound(begin(), end(), v);
    insert(i, v);
}

void ControlPoints::setCP(uint32 i, const ControlPoint &v)
{
    if(i<size()) {
        (*this)[i] = v;
        stl::sort(begin(), end());
    }
}

void ControlPoints::eraseCP(uint32 i)
{
    if(i<size()) {
        erase(begin()+i);
    }
}

float32 ControlPoints::computeValue(float32 t)
{
    float32 r = 0.0f;
    if(empty()) {}
    else if(t<=front().time){ r=front().value; }
    else if(t>=back().time) { r=back().value; }
    else {
        auto p2 = stl::lower_bound(begin(), end(), t, [&](const ControlPoint &v, float32 t){ return v.time<t; });
        auto p1 = p2-1;
        r = Interpolate(*p1, *p2, t);
    }
    return r;
}

} // namespace ist
