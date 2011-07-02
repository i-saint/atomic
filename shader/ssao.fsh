#version 410 compatibility

uniform sampler2D ColorBuffer;
uniform sampler2D NormalBuffer;
uniform sampler2D DepthBuffer;
uniform sampler2D RandomBuffer;
in vec2 Texcoord;

uniform float totStrength = 1.38;
uniform float strength = 0.05;
uniform float offset = 18.0;
uniform float falloff = 0.000002;
uniform float rad = 0.006;
#define SAMPLES 16 // 10 is good
const float invSamples = 1.0/16.0;

layout(location=0) out vec4 FragColor;
layout(location=1) out vec4 FragColor1;
layout(location=2) out vec4 FragColor2;


void main()
{
    FragColor = texture2D(ColorBuffer, Texcoord);


    // these are the random vectors inside a unit sphere
    vec3 pSphere[16] = vec3[](vec3(0.53812504, 0.18565957, -0.43192),vec3(0.13790712, 0.24864247, 0.44301823),vec3(0.33715037, 0.56794053, -0.005789503),vec3(-0.6999805, -0.04511441, -0.0019965635),vec3(0.06896307, -0.15983082, -0.85477847),vec3(0.056099437, 0.006954967, -0.1843352),vec3(-0.014653638, 0.14027752, 0.0762037),vec3(0.010019933, -0.1924225, -0.034443386),vec3(-0.35775623, -0.5301969, -0.43581226),vec3(-0.3169221, 0.106360726, 0.015860917),vec3(0.010350345, -0.58698344, 0.0046293875),vec3(-0.08972908, -0.49408212, 0.3287904),vec3(0.7119986, -0.0154690035, -0.09183723),vec3(-0.053382345, 0.059675813, -0.5411899),vec3(0.035267662, -0.063188605, 0.54602677),vec3(-0.47761092, 0.2847911, -0.0271716));

    vec2 uv = Texcoord;
   // grab a normal for reflecting the sample rays later on
   vec3 fres = normalize((texture2D(RandomBuffer, uv*offset).xyz*2.0) - vec3(1.0));
   vec4 currentPixelSample = texture2D(NormalBuffer, uv);
   float currentPixelDepth = texture2D(DepthBuffer, uv).r;

   // current fragment coords in screen space
   vec3 ep = vec3(uv.xy, currentPixelDepth);
   // get the normal of current fragment
   vec3 norm = currentPixelSample.xyz;

   float bl = 0.0;
   // adjust for the depth ( not shure if this is good..)
   float radD = rad/currentPixelDepth;

   vec3 ray, se, occNorm;
   float occluderDepth, depthDifference, normDiff;

   for(int i=0; i<SAMPLES; ++i)
   {
      // get a vector (randomized inside of a sphere with radius 1.0) from a texture and reflect it
      ray = radD*reflect(pSphere[i],fres);

      // if the ray is outside the hemisphere then change direction
      se = ep + sign(dot(ray,norm) )*ray;

      // get the normal of the occluder fragment
      occNorm = texture2D(NormalBuffer, se.xy).rgb;
      occluderDepth = texture2D(DepthBuffer, se.xy).r;

      // if depthDifference is negative = occluder is behind current fragment
      depthDifference = currentPixelDepth-occluderDepth;

      // calculate the difference between the normals as a weight

      normDiff = (1.0-dot(occNorm,norm));
      // the falloff equation, starts at falloff and is kind of 1/x^2 falling
      bl += step(falloff,depthDifference)*normDiff*(1.0-smoothstep(falloff,strength,depthDifference));
   }

   // output the result
   float ao = 1.0-totStrength*bl*invSamples;
   FragColor1 = vec4(ao, ao, ao, 1.0);
   FragColor2 = vec4(FragColor.rgb*ao, 1.0);
}
