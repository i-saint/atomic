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

float sr1 = sin(radians(45.0));
float cr1 = cos(radians(45.0));
float sr2 = sin(radians(0.0));
float cr2 = cos(radians(0.0));
mat3 rot= mat3(
  cr1, 0, sr1,
   0, 1,  0,
 -sr1, 0, cr1 )
* mat3(
    cr2, sr2, 0,
    sr2,-cr2, 0,
     0,  0, 1 );

float sdCross( in vec3 p, in vec2 b )
{
    float da = sdBox(p.xy,b);
    float db = sdBox(p.yz,b);
    float dc = sdBox(p.zx,b);
    return min(da,min(db,dc));
}

float map(vec3 p)
{
    float d3 = p.z - 0.3;
    p = rot * p;
    p = mod(p, vec3(3.0)) - vec3(1.5);
    float s1 = sdCross(p, vec2(1.06));

    vec3 p2 = mod(p, vec3(0.15)) - vec3(0.075);
    float d1 = sdBox(p2,vec3(0.05));

    float s = 1.0;

    return max(max(-s1,d1), d3);
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
    camPos.x +=  -time*0.1;
    camPos.y +=  -time*0.4;
    vec3 camDir = u_RS.CameraDirection.xyz;
    vec3 camUp  = vec3(0.0, 1.0, 0.0);
    vec3 camSide = cross(camDir, camUp);
    float focus = 1.8;

    vec3 rayDir = normalize(camSide*pos.x + camUp*pos.y + camDir*focus);

    vec3 ray = camPos;
    int march = 0;
    float d = 0.0, total_d = 0.0;
    const int MAX_MARCH = 32;
    const float MAX_DISTANCE = 750.0;
    for(int mi=0; mi<MAX_MARCH; ++mi) {
        march = mi;
        d = map(ray);
        total_d += d;
        ray += rayDir * d;
        if(d<0.001) {break; }
    }

    float glow = 0.0;
    {
        const float s = 0.0075;
        vec3 p = ray;
        vec3 n1 = genNormal(ray);
        vec3 n2 = genNormal(ray+vec3(s, 0.0, 0.0));
        vec3 n3 = genNormal(ray+vec3(0.0, s, 0.0));
        glow = max(1.0-abs(dot(camDir, n1)-0.5), 0.0)*0.5;
        if(dot(n1, n2)<0.8 || dot(n1, n3)<0.8) {
            glow += 0.6;
        }
    }
    {
        vec3 p = rot * ray;
        float grid1 = max(0.0, max((mod((p.x+p.y+p.z*2.0)-time*3.0, 5.0)-4.0)*1.5, 0.0) );
        float grid2 = max(0.0, max((mod((p.x+p.y*2.0+p.z)-time*2.0, 7.0)-6.0)*1.2, 0.0) );
        vec3 gp1 = abs(mod(p, vec3(0.24)));
        vec3 gp2 = abs(mod(p, vec3(0.36)));
        if(gp1.x<0.235 && gp1.y<0.235) {
            grid1 = 0.0;
        }
        if(gp2.y<0.35 && gp2.z<0.35) {
            grid2 = 0.0;
        }
        glow += grid1+grid2;
    }

    float fog = min(1.0, (1.0 / float(MAX_MARCH)) * float(march));
    vec3  fog2 = 0.0075 * vec3(1, 1, 1.5) * total_d;
    glow *= min(1.0, 4.0-(4.0 / float(MAX_MARCH-1)) * float(march));
    ps_FlagColor = vec4(vec3(0.075+glow*0.75, 0.075+glow*0.75, 0.1+glow)*fog + fog2, 1.0);
}

#endif

