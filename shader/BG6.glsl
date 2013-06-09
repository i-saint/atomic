#include "Common.h"
#include "DistanceFunctions.h"

#ifdef GLSL_VS
ia_out(GLSL_POSITION)           vec2 ia_VertexPosition;
#endif

#if defined(GLSL_VS)

void main()
{
    gl_Position = vec4(ia_VertexPosition, 0.99, 1.0);
}

#elif defined(GLSL_PS)

float sdCross( in vec3 p )
{
    float da = sdBox(p.xy,vec2(1.0));
    float db = sdBox(p.yz,vec2(1.0));
    float dc = sdBox(p.zx,vec2(1.0));
    return min(da,min(db,dc));
}

float map(vec3 p)
{
    float d3 = p.z - 0.3;

    p = mod(p, vec3(3.0)) - vec3(1.5);
    float sr = sin(radians(45.0));
    float cr = cos(radians(45.0));
    mat3 rotz = mat3(
        cr, sr, 0,
        sr,-cr, 0,
         0,  0, 1 );
    mat3 roty = mat3(
      cr, 0, sr,
       0, 1,  0,
     -sr, 0, cr );
    p = rotz * p;
    p = roty * p;

    p.z += 0.7;
    float d = sdBox(p,vec3(1.0));

    float s = 1.0;
    for( int m=0; m<3; m++ )
    {
       vec3 a = mod( p*s, 2.0 )-1.0;
       s *= 3.0;
       vec3 r = 3.0*abs(a);

       float c = sdCross(r)/s;
       d = max(d,-c);
    }

    return max(d, d3);
}

vec3 genNormal(vec3 p)
{
    const float d = 0.01;
    return normalize( vec3(
        map(p+vec3(  d,0.0,0.0))-map(p+vec3( -d,0.0,0.0)),
        map(p+vec3(0.0,  d,0.0))-map(p+vec3(0.0, -d,0.0)),
        map(p+vec3(0.0,0.0,  d))-map(p+vec3(0.0,0.0, -d)) ));
}

ps_out(0) vec4 ps_FlagColor;

void main()
{
    float time = u_RS.Frame / 60.0;
    vec2 pos = (gl_FragCoord.xy*2.0 - u_RS.ScreenSize) * u_RS.RcpScreenSize.y;
    vec3 camPos = u_RS.CameraPosition.xyz;
    camPos.x +=  -time*0.4;
    camPos.y +=  -time*0.1;
    vec3 camDir = u_RS.CameraDirection.xyz;
    vec3 camUp  = vec3(0.0, 1.0, 0.0);
    vec3 camSide = cross(camDir, camUp);
    float focus = 1.8;

    vec3 rayDir = normalize(camSide*pos.x + camUp*pos.y + camDir*focus);

    vec3 ray = camPos;
    int i = 0;
    float d = 0.0, total_d = 0.0;
    const int MAX_MARCH = 32;
    const float MAX_DISTANCE = 750.0;
    for(0; i<MAX_MARCH; ++i) {
        d = map(ray);
        total_d += d;
        ray += rayDir * d;
        if(d<0.001) { break; }
        if(total_d>MAX_DISTANCE) {
            total_d = MAX_DISTANCE;
            i = MAX_MARCH;
            ray = camPos + rayDir*MAX_DISTANCE;
            break;
        }
    }

    //vec3 normal;
    //if(total_d>MAX_DISTANCE) {
    //    normal = -rayDir;
    //}
    //else {
    //    normal = genNormal(ray);
    //}

    const float m = 1.0 / MAX_MARCH;

    //float glow = max((mod((ray.x+ray.y+ray.z)-time*2.0, 10.0)-9.0)/2.0, 0.0);
    float glow = 0.0;

    float fog = m*i;
    ps_FlagColor = vec4(vec3(0.4+glow, 0.4+glow, 0.5+glow)*fog, 1.0);
}

#endif

