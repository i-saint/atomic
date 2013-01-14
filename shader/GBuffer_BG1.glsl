#pragma include("Common.h")
#pragma include("DistanceFunctions.h")

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

float fractal1(vec3 p)
{
    const float bailout = 1000.0;
    const float scale = 2;
    const vec3 offset = vec3(1.0);
    float r = dot(p, p);
    int i;
    for(i=0; i<10 && r<bailout; i++){
        //p = abs(p);
        if(p.x+p.y<0) { p.xy = -p.yx; }
        if(p.x+p.z<0) { p.xz = -p.zx; }
        if(p.y+p.z<0) { p.zy = -p.yz; }
        p = scale*p - offset*(scale-1);
        r = dot(p, p);
    }
    return (sqrt(r)-1.0) * pow(scale, -i);
}


void sphereFold(inout vec3 z, inout float dz)
{
    const float fixedRadius2 = 1.0;
    const float minRadius2 = 0.5;
    float r2 = dot(z,z);
    if (r2<minRadius2) { 
        // linear inner scaling
        float temp = (fixedRadius2/minRadius2);
        z *= temp;
        dz*= temp;
    } else if (r2<fixedRadius2) { 
        // this is the actual sphere inversion
        float temp =(fixedRadius2/r2);
        z *= temp;
        dz*= temp;
    }
}
void boxFold(inout vec3 z, inout float dz)
{
    const float foldingLimit = 2.0;
    z = clamp(z, -foldingLimit, foldingLimit) * 2.0 - z;
}
float fractal2(vec3 z)
{
    float sr = sin(radians(u_RS.Frame*0.05));
    float cr = cos(radians(u_RS.Frame*0.05));
    mat3 rotz = mat3(
        cr, sr, 0,
        sr,-cr, 0,
         0,  0, 1 );
    mat3 roty = mat3(
      cr, 0, sr,
       0, 1,  0,
     -sr, 0, cr );
    //z = rotz * z;
    z = roty * z;


    const float Scale = -2.5;
    z.z += 4;
    vec3 offset = z;
    float dr = 1.0;
    for(int n = 0; n<12; n++) {
        boxFold(z,dr);
        sphereFold(z,dr);
        z = Scale*z + offset;
        dr = dr*abs(Scale)+1.0;
    }
    float r = length(z);
    return r/abs(dr);
}

float map(vec3 p)
{
    return fractal2(p);
}

float map_(vec3 p)
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

float map1(vec3 p)
{
    p.z += 0.7;
    vec3 p1 = mod(p + vec3(0, u_RS.Frame*0.001, 0.0), 0.2)-0.1;
    vec3 p2 = mod(p + vec3(0, u_RS.Frame*0.003, 0.0), 0.8)-0.4;
    float d1 = udBox(p1, vec3(0.05, 0.07, 0.05));
    float d2 = udBox(p2, vec3(0.075, 0.1, 0.075));
    float d3 = p.z - 1.2;

    return max(min(d1, d2), d3);
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
ps_out(1) vec4 ps_FragNormal;
ps_out(2) vec4 ps_FragPosition;
ps_out(3) vec4 ps_FragGlow;

void main()
{
    vec2 pos = (gl_FragCoord.xy*2.0 - u_RS.ScreenSize) * u_RS.RcpScreenSize.y;
    vec3 camPos = u_RS.CameraPosition.xyz;
    vec3 camDir = u_RS.CameraDirection.xyz;
    vec3 camUp  = vec3(0.0, 1.0, 0.0);
    vec3 camSide = cross(camDir, camUp);
    float focus = 1.8;

    vec3 rayDir = normalize(camSide*pos.x + camUp*pos.y + camDir*focus);

    vec3 ray = camPos;
    int i = 0;
    float d = 0.0, total_d = 0.0;
    const int MAX_MARCH = 64;
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
    vec3 normal;
    if(total_d>MAX_DISTANCE) {
        normal = -rayDir;
    }
    else {
        normal = genNormal(ray);
    }


    ps_FlagColor    = vec4(0.6, 0.6, 0.8, 70.0);
    ps_FragNormal   = vec4(normal, 0.0);
    ps_FragPosition = vec4(ray, total_d);
    const float fog = 1.0 / MAX_MARCH;
    ps_FragGlow     = min(vec4(fog*i*1.2, fog*i*1.2, fog*i*2.5, i), 1.0) * 0.5;
}

#endif

