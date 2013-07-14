#ifndef ist_Math_Curve_h
#define ist_Math_Curve_h
#include "../Base/Types.h"
#include "../Base/Serialize.h"
#include "../stdex/ist_vector.h"
#include "Misc.h"

namespace ist {

struct ControlPoint
{
    enum Interpolation {
        None,
        Linear,
        Decel,
        Accel,
        Smooth,
        Bezier,
    };
    float32 time;
    float32 value;
    float32 bez_in, bez_out;
    Interpolation interp;

    ControlPoint() : time(0.0f), value(0.0f), bez_in(0.0f),bez_out(0.0f), interp(Linear) {}
    ControlPoint(float32 t, float32 v, Interpolation ts=Linear) : time(t), value(v), bez_in(0.0f),bez_out(0.0f), interp(ts) {}
    ControlPoint(float32 t, float32 v, float32 _in, float32 _out, Interpolation ts=Linear) : time(t), value(v), bez_in(_in),bez_out(_out), interp(ts) {}
    bool operator<(const ControlPoint &p) const { return time<p.time; }
};
float32 Interpolate(const ControlPoint &v1, const ControlPoint &v2, float32 time);


class ControlPoints : public ist::vector<ControlPoint>
{
public:
    void addCP(const ControlPoint &v);
    void setCP(uint32 i, const ControlPoint &v);
    void eraseCP(uint32 i);
    float32 computeValue(float32 t);
};

} // namespace ist

istSerializeRaw(ist::ControlPoint);

#endif // ist_Math_Curve_h
