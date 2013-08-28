#include "istPCH.h"
#include "../Math.h"

namespace ist {
namespace linear {

template<class vecN>
float32 PointPointDistanceImpl(const vecN &p1, const vecN &p2)
{
    return glm::distance(p1, p2);
}
istAPI float32 PointPointDistance(const vec2 &p1, const vec2 &p2) { return PointPointDistanceImpl(p1, p2); }
istAPI float32 PointPointDistance(const vec3 &p1, const vec3 &p2) { return PointPointDistanceImpl(p1, p2); }


template<class vecN>
float32 LinePointDistanceImpl(const vecN &l1, const vecN &l2, const vecN &p)
{
    const float len_sq = glm::distance2(l2, l1);
    if(len_sq==0.0) {
        return glm::distance(p, l1);
    }
    const float t = glm::dot(p - l1, l2 - l1) / len_sq;
    if(t < 0.0) {
        return glm::distance(p, l1);
    }
    else if (t > 1.0) {
        return glm::distance(p, l2);
    }
    const vecN projection = l1 + t * (l2 - l1);
    return glm::distance(p, projection);
}
istAPI float32 LinePointDistance(const vec2 &l1, const vec2 &l2, const vec2 &p) { return LinePointDistanceImpl(l1, l2, p); }
istAPI float32 LinePointDistance(const vec3 &l1, const vec3 &l2, const vec3 &p) { return LinePointDistanceImpl(l1, l2, p); }


template<class vecN>
float32 LineLineDistanceImpl(const vecN &l1p1, const vecN &l1p2, const vecN &l2p1, const vecN &l2p2)
{
    vecN    u = l1p2 - l1p1;
    vecN    v = l2p2 - l2p1;
    vecN    w = l1p1 - l2p1;
    float32 a = glm::dot(u,u);
    float32 b = glm::dot(u,v);
    float32 c = glm::dot(v,v);
    float32 d = glm::dot(u,w);
    float32 e = glm::dot(v,w);
    float32 D = a*c - b*b;
    float32 sc = 0.0f;
    float32 tc = 0.0f;

    if (D < 0.00001f) {
        sc = 0.0;
        tc = (b>c ? d/b : e/c);
    }
    else {
        sc = (b*e - c*d) / D;
        tc = (a*e - b*d) / D;
    }

    vecN dP = w + (sc * u) - (tc * v);
    return glm::length(dP);
}
istAPI float32 LineLineDistance(const vec2 &l1p1, const vec2 &l1p2, const vec2 &l2p1, const vec2 &l2p2) { return LineLineDistanceImpl(l1p1, l1p2, l2p1, l2p2); }
istAPI float32 LineLineDistance(const vec3 &l1p1, const vec3 &l1p2, const vec3 &l2p1, const vec3 &l2p2) { return LineLineDistanceImpl(l1p1, l1p2, l2p1, l2p2); }

} // namespace linear
} // namespace ist
