#ifndef __ist_Sound_types__
#define __ist_Sound_types__

namespace ist {
namespace sound {

struct float2 { float x,y; };
struct float3 { float x,y,z; };
struct float4 { float x,y,z,w; };

inline float2 make_float2(float v)
{
    float2 r = {v,v};
    return r;
}
inline float2 make_float2(float *v)
{
    float2 r = {v[0],v[1]};
    return r;
}
inline float2 make_float2(float x, float y)
{
    float2 r = {x,y};
    return r;
}

inline float3 make_float3(float v)
{
    float3 r = {v,v,v};
    return r;
}
inline float3 make_float3(float *v)
{
    float3 r = {v[0],v[1],v[2]};
    return r;
}
inline float3 make_float3(float x, float y, float z)
{
    float3 r = {x,y,z};
    return r;
}

inline float4 make_float4(float v)
{
    float4 r = {v,v,v,v};
    return r;
}
inline float4 make_float4(float *v)
{
    float4 r = {v[0],v[1],v[2],v[3]};
    return r;
}
inline float4 make_float4(float x, float y, float z, float w)
{
    float4 r = {x,y,z,w};
    return r;
}

} // namespace sound
} // namespace ist

#endif // __ist_Sound_types__
