
float hash( float n )
{
    return fract(sin(n)*43758.5453);
}

vec2 hash( vec2 p )
{
    p = vec2( dot(p,vec2(127.1,311.7)), dot(p,vec2(269.5,183.3)) );
    return fract(sin(p)*43758.5453);
}

vec3 hash( vec3 p )
{
    p = vec3( dot(p,vec3(127.1,311.7,311.7)), dot(p,vec3(269.5,183.3,183.3)), dot(p,vec3(269.5,183.3,183.3)) );
    return fract(sin(p)*43758.5453);
}

float voronoi( in vec2 x )
{
    vec2 n = floor(x);
    vec2 f = fract(x);
    vec2 mg, mr;

    float md = 8.0;
    for( int j=-1; j<=1; j++ ) {
    for( int i=-1; i<=1; i++ ) {
        vec2 g = vec2(float(i),float(j));
        vec2 o = hash( n + g );
        vec2 r = g + o - f;
        float d = dot(r,r);
        if( d<md ) {
            md = d;
            mr = r;
            mg = g;
        }
    }}

    md = 8.0;
    for( int j=-2; j<=2; j++ ) {
    for( int i=-2; i<=2; i++ ) {
        vec2 g = mg + vec2(float(i),float(j));
        vec2 o = hash( n + g );
        vec2 r = g + o - f;
        if( dot(mr-r,mr-r)>0.000001 ) {
            float d = dot( 1.5*(mr+r), normalize(r-mr) );
            md = min( md, d );
        }
    }}

    return md;
}

float voronoi( in vec3 x )
{
    vec3 n = floor(x);
    vec3 f = fract(x);
    vec3 mg, mr;

    float md = 8.0;
    for( int j=-1; j<=1; j++ ) {
    for( int i=-1; i<=1; i++ ) {
    for( int k=-1; k<=1; k++ ) {
        vec3 g = vec3(float(i),float(j),float(k));
        vec3 o = hash( n + g );
        vec3 r = g + o - f;
        float d = dot(r,r);
        if( d<md ) {
            md = d;
            mr = r;
            mg = g;
        }
    }}}

    md = 8.0;
    for( int j=-2; j<=2; j++ ) {
    for( int i=-2; i<=2; i++ ) {
    for( int k=-2; k<=2; k++ ) {
        vec3 g = mg + vec3(float(i),float(j),float(k));
        vec3 o = hash( n + g );
        vec3 r = g + o - f;
        if( dot(mr-r,mr-r)>0.000001 ) {
            float d = dot( 1.5*(mr+r), normalize(r-mr) );
            md = min( md, d );
        }
    }}}

    return md;
}
