ist_EasyDrawer_NamespaceBegin

const char *g_vs_p2c4 = "\
#version 330 core\n\
struct RenderStates\
{\
    mat4 ViewProjectionMatrix;\
};\
layout(std140) uniform render_states\
{\
    RenderStates u_RS;\
};\
layout(location=0) in vec2 ia_VertexPosition;\
layout(location=1) in vec4 ia_VertexColor;\
out vec4 vs_Color;\
\
void main(void)\
{\
    vs_Color    = ia_VertexColor;\
    gl_Position = u_RS.ViewProjectionMatrix * vec4(ia_VertexPosition, 0.0f, 1.0);\
}\
";

const char *g_vs_p2t2c4 = "\
#version 330 core\n\
struct RenderStates\
{\
    mat4 ViewProjectionMatrix;\
};\
layout(std140) uniform render_states\
{\
    RenderStates u_RS;\
};\
layout(location=0) in vec2 ia_VertexPosition;\
layout(location=1) in vec2 ia_VertexTexcoord;\
layout(location=2) in vec4 ia_VertexColor;\
out vec2 vs_Texcoord;\
out vec4 vs_Color;\
\
void main(void)\
{\
    vs_Texcoord = ia_VertexTexcoord;\
    vs_Color    = ia_VertexColor;\
    gl_Position = u_RS.ViewProjectionMatrix * vec4(ia_VertexPosition, 0.0f, 1.0);\
}\
";

const char *g_vs_p3t2c4 = "\
#version 330 core\n\
struct RenderStates\
{\
    mat4 ViewProjectionMatrix;\
};\
layout(std140) uniform render_states\
{\
    RenderStates u_RS;\
};\
layout(location=0) in vec3 ia_VertexPosition;\
layout(location=1) in vec2 ia_VertexTexcoord;\
layout(location=2) in vec4 ia_VertexColor;\
out vec2 vs_Texcoord;\
out vec4 vs_Color;\
\
void main(void)\
{\
    vs_Texcoord = ia_VertexTexcoord;\
    vs_Color    = ia_VertexColor;\
    gl_Position = u_RS.ViewProjectionMatrix * vec4(ia_VertexPosition, 1.0);\
}\
";

const char *g_ps_colored = "\
#version 330 core\n\
in vec4 vs_Color;\
layout(location=0) out vec4 ps_FragColor;\
\
void main()\
{\
    ps_FragColor = vs_Color;\
}\
";

const char *g_ps_colored_textured = "\
#version 330 core\n\
uniform sampler2D u_Texture;\
in vec2 vs_Texcoord;\
in vec4 vs_Color;\
layout(location=0) out vec4 ps_FragColor;\
\
void main()\
{\
    vec4 color = vs_Color;\
    color *= texture(u_Texture, vs_Texcoord);\
    ps_FragColor = vec4(color);\
}\
";

ist_EasyDraw_NamespaceEnd
