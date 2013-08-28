#ifndef ist_Graphic_Linear_h
#define ist_Graphic_Linear_h

#include "ist/Config.h"


namespace ist {

istAPI float32 PointPointDistance(const vec2 &p1, const vec2 &p2);
istAPI float32 PointPointDistance(const vec3 &p1, const vec3 &p2);

istAPI float32 LinePointDistance(const vec2 &l1, const vec2 &l2, const vec2 &p);
istAPI float32 LinePointDistance(const vec3 &l1, const vec3 &l2, const vec3 &p);

istAPI float32 LineLineDistance(const vec2 &l1p1, const vec2 &l1p2, const vec2 &l2p1, const vec2 &l2p2);
istAPI float32 LineLineDistance(const vec3 &l1p1, const vec3 &l1p2, const vec3 &l2p1, const vec3 &l2p2);

} // namespace ist
#endif // ist_Graphic_Linear_h
