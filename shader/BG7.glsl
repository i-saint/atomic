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

float map(vec3 p)
{
    float h = 1.0;
    float rh = 0.5;
    vec2 grid = vec2(1.2, 0.8);
    vec2 grid_half = grid*0.5;
    float radius = 0.35;
    vec3 orig = p;

    p.y = -abs(p.y);

    vec2 g1 = vec2(ceil(orig.xz/grid));
    vec2 g2 = vec2(ceil((orig.xz+grid_half)/grid));
    vec3 rxz =  nrand3(g1);
    vec3 ryz =  nrand3(g2);

    float d1 = p.y + h + rxz.x*rh;
    float d2 = p.y + h + ryz.y*rh;

    vec2 p1 = mod(p.xz, grid) - grid_half;
    float c1 = sdHexPrism(vec2(p1.x,p1.y), vec2(radius));

    vec2 p2 = mod(p.xz+grid_half, grid) - vec2(grid_half);
    float c2 = sdHexPrism(vec2(p2.x,p2.y), vec2(radius));

    float dz = (grid.y*g1.y - p.z + 0.1)*0.5;
    float dz1 = -(abs(p.y)-h)+0.1;

    return min(min(max(c1,d1), max(c2,d2)), max(dz,dz1));
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
    vec3 camPos = u_RS.CameraPosition.xyz*0.2;
    vec3 camDir = normalize(u_RS.CameraDirection.xyz+vec3(0.8, 0.0, -1.0));
    vec3 camUp  = vec3(0.0, 1.0, 0.0);
    vec3 camSide = cross(camDir, camUp);
    float focus = 1.8;
    camPos -=  vec3(time*0.1,0.0,time*0.7);

    vec3 rayDir = normalize(camSide*pos.x + camUp*pos.y + camDir*focus);	    
    vec3 ray = camPos;
    int march = 0;
    float d = 0.0;

    float total_d = 0.0;
    const int MAX_MARCH = 48;
    const float MAX_DIST = 50.0;
    for(int mi=0; mi<MAX_MARCH; ++mi) {
        d = map(ray);
        march=mi;
        total_d += d;
        ray += rayDir * d;
        if(d<0.001) {break; }
        if(total_d>MAX_DIST) {
            total_d = MAX_DIST;
            march = MAX_MARCH-1;
            break;
        }
    }

    float glow = 0.0;
    float sn = 0.0;
    {
        const float s = 0.001;
        vec3 p = ray;
        vec3 n1 = genNormal(ray);
        vec3 n2 = genNormal(ray+vec3(s, 0.0, 0.0));
        vec3 n3 = genNormal(ray+vec3(0.0, -s, 0.0));
        glow = max(1.0-abs(dot(camDir, n1)-0.5), 0.0)*0.5;
        if(dot(n1, n2)<0.999 || dot(n1, n3)<0.999) {
            sn += 1.0;
        }
    }
    {
        vec3 p = ray;
        float grid1 = max(0.0, max((mod((p.x+p.y+p.z*2.0)-time*3.0, 5.0)-4.0)*1.5, 0.0) );
        float grid2 = max(0.0, max((mod((p.x+p.y*2.0+p.z)-time*2.0, 7.0)-6.0)*1.2, 0.0) );
        sn = sn*0.2 + sn*(grid1+grid2)*1.0;
    }
    glow += sn;

    float fog = min(1.0, (1.0 / float(MAX_MARCH)) * float(march))*1.0;
    vec3  fog2 = 0.005 * vec3(1, 1, 1.5) * total_d;
    glow *= min(1.0, 4.0-(4.0 / float(MAX_MARCH-1)) * float(march));
    float scanline = mod(gl_FragCoord.y, 4.0) < 2.0 ? 0.7 : 1.0;
    ps_FlagColor = vec4(vec3(0.15+glow*0.75, 0.15+glow*0.75, 0.2+glow)*fog + fog2, 1.0) * scanline;
}

#endif

