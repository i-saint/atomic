#include "stdafx.h"
#include "ist/ist.h"
#include "types.h"
#include "Graphics/CreateModelData.h"


namespace atomic {


void CreateQuadModel(ModelData& model, float32 len)
{
    vec4 vertex[4];
    vec3 normal[4];
    int index[4];

    const float32 half_len = len/2.0f;
    vertex[0] = vec4( half_len,  half_len,  half_len, 1.0f);
    vertex[1] = vec4( half_len, -half_len,  half_len, 1.0f);
    vertex[2] = vec4( half_len, -half_len, -half_len, 1.0f);
    vertex[3] = vec4(-half_len, -half_len, -half_len, 1.0f);

    normal[0] = vec3(0.0f, 0.0f, 1.0f);
    normal[1] = normal[0];
    normal[2] = normal[0];
    normal[3] = normal[0];

    index[0] = 0;
    index[1] = 1;
    index[2] = 2;
    index[3] = 3;

    model.setData(0, vertex, _countof(vertex), 4);
    model.setData(1, normal, _countof(normal), 3);
    model.setIndex(index, _countof(index), ModelData::IDX_INT32, ModelData::PRM_TRIANGLE_FAN);
}

void CreateCubeModel(ModelData& model, float32 len)
{
    vec4 vertex[24];
    vec3 normal[24];
    int index[24];

    const float32 half_len = len/2.0f;
    vec3 ur = vec3( half_len, half_len, half_len);
    vec3 bl = vec3(-half_len,-half_len,-half_len);
    vec3 n;

    n = vec3(1.0f, 0.0f, 0.0f);
    normal[0] = n;
    normal[1] = n;
    normal[2] = n;
    normal[3] = n;
    vertex[0] = vec4(ur[0], ur[1], ur[2], 1.0f);
    vertex[1] = vec4(ur[0], bl[1], ur[2], 1.0f);
    vertex[2] = vec4(ur[0], bl[1], bl[2], 1.0f);
    vertex[3] = vec4(ur[0], ur[1], bl[2], 1.0f);

    n = vec3(-1.0f, 0.0f, 0.0f);
    normal[4] = n;
    normal[5] = n;
    normal[6] = n;
    normal[7] = n;
    vertex[4] = vec4(bl[0], ur[1], ur[2], 1.0f);
    vertex[5] = vec4(bl[0], ur[1], bl[2], 1.0f);
    vertex[6] = vec4(bl[0], bl[1], bl[2], 1.0f);
    vertex[7] = vec4(bl[0], bl[1], ur[2], 1.0f);

    n = vec3(0.0f, 1.0f, 0.0f);
    normal[8] = n;
    normal[9] = n;
    normal[10] = n;
    normal[11] = n;
    vertex[8] = vec4(ur[0], ur[1], ur[2], 1.0f);
    vertex[9] = vec4(ur[0], ur[1], bl[2], 1.0f);
    vertex[10] = vec4(bl[0], ur[1], bl[2], 1.0f);
    vertex[11] = vec4(bl[0], ur[1], ur[2], 1.0f);

    n = vec3(0.0f, -1.0f, 0.0f);
    normal[12] = n;
    normal[13] = n;
    normal[14] = n;
    normal[15] = n;
    vertex[12] = vec4(ur[0], bl[1], ur[2], 1.0f);
    vertex[13] = vec4(bl[0], bl[1], ur[2], 1.0f);
    vertex[14] = vec4(bl[0], bl[1], bl[2], 1.0f);
    vertex[15] = vec4(ur[0], bl[1], bl[2], 1.0f);

    n = vec3(0.0f, 0.0f, 1.0f);
    normal[16] = n;
    normal[17] = n;
    normal[18] = n;
    normal[19] = n;
    vertex[16] = vec4(ur[0], ur[1], ur[2], 1.0f);
    vertex[17] = vec4(bl[0], ur[1], ur[2], 1.0f);
    vertex[18] = vec4(bl[0], bl[1], ur[2], 1.0f);
    vertex[19] = vec4(ur[0], bl[1], ur[2], 1.0f);

    n = vec3(0.0f, 0.0f, -1.0f);
    normal[20] = n;
    normal[21] = n;
    normal[22] = n;
    normal[23] = n;
    vertex[20] = vec4(ur[0], ur[1], bl[2], 1.0f);
    vertex[21] = vec4(ur[0], bl[1], bl[2], 1.0f);
    vertex[22] = vec4(bl[0], bl[1], bl[2], 1.0f);
    vertex[23] = vec4(bl[0], ur[1], bl[2], 1.0f);

    for(size_t i=0; i<24; ++i) {
        index[i] = i;
    }

    model.setData(0, vertex, _countof(vertex), 4);
    model.setData(1, normal, _countof(normal), 3);
    model.setIndex(index, _countof(index), ModelData::IDX_INT32, ModelData::PRM_QUADS);
}

void CreateOctahedronModel(ModelData& model, float32 len_xz, float32 len_y)
{
    vec4 vertex[8*3];
    vec3 normal[8*3];
    int index[8*3];

    IST_ASSERT("������");

    model.setData(0, vertex, _countof(vertex), 4);
    model.setData(1, normal, _countof(normal), 3);
    model.setIndex(index, _countof(index), ModelData::IDX_INT32, ModelData::PRM_TRIANGLES);
}

void CreateSphereModel(ModelData& model, float32 radius, uint32 div_xz, uint32 div_y)
{
    stl::vector<vec4> v(div_y*div_xz);
    stl::vector<vec3> n(div_y*div_xz);
    stl::vector<int32> index((div_y-1)*(div_xz)*4);

    for(uint32 yi=0; yi<div_y; ++yi) {
        float ang = XMConvertToRadians(180.0f/(div_y-1)*yi-90.0f);
        v[div_xz*yi] = vec4(cos(ang)*radius, sin(ang)*radius, 0, 1.0);
    }

    mat4 mat;
    for(uint32 xzi=0; xzi<div_xz; ++xzi) {
        for(uint32 yi=0; yi<div_y; ++yi) {
            vec4 *vr = &v[div_xz*yi];
            vec3 *nr = &n[div_xz*yi];
            vr[xzi] = mat * vr[0];
            nr[xzi] = glm::normalize(vec3(vr[xzi].x, vr[xzi].y, vr[xzi].z));
        }
        mat = glm::rotate(mat4(), 360.0f/div_xz*xzi, vec3(0.0f, 1.0f, 0.0f));
    }

    int32 *ci = &index[0];
    for(uint32 yi=0; yi<div_y-1; ++yi) {
        for(uint32 xzi=0; xzi<div_xz; ++xzi) {
            ci[0] = div_xz*(yi)  + xzi;
            ci[1] = div_xz*(yi)  + ((xzi+1)%div_xz);
            ci[2] = div_xz*(yi+1)+ ((xzi+1)%div_xz);
            ci[3] = div_xz*(yi+1)+ xzi;
            ci+=4;
        }
    }
    model.setData(0, &v[0], div_y*div_xz, 4);
    model.setData(1, &n[0], div_y*div_xz, 3);
    model.setIndex(&index[0], index.size(), ModelData::IDX_INT32, ModelData::PRM_QUADS);
}

} // namespace atomic