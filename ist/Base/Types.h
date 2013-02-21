#ifndef ist_Base_Types_h
#define ist_Base_Types_h

#define GLM_FORCE_SSE2
#include <glm/glm.hpp>
#include <glm/gtx/simd_vec4.hpp>
#include <glm/gtx/simd_mat4.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "Half/half.h"
#include "Serialize.h"

namespace ist {

typedef char                int8;
typedef short               int16;
typedef int                 int32;
typedef long long           int64;
typedef unsigned char       uint8;
typedef unsigned short      uint16;
typedef unsigned int        uint32;
typedef unsigned long long  uint64;
typedef __m128i             uint128;
typedef half                float16;
typedef float               float32;
typedef double              float64;

typedef glm::vec2       vec2;
typedef glm::vec3       vec3;
typedef glm::vec4       vec4;
typedef glm::ivec2      ivec2;
typedef glm::ivec3      ivec3;
typedef glm::ivec4      ivec4;
typedef glm::uvec2      uvec2;
typedef glm::uvec3      uvec3;
typedef glm::uvec4      uvec4;
typedef glm::mat2       mat2;
typedef glm::mat3       mat3;
typedef glm::mat4       mat4;
typedef glm::simdVec4   simdvec4;
//typedef glm::simdVec8   simdvec8;
typedef glm::simdMat4   simdmat4;

} // namespace ist

istSerializePrimitive(ist::vec2);
istSerializePrimitive(ist::vec3);
istSerializePrimitive(ist::vec4);
istSerializePrimitive(ist::ivec2);
istSerializePrimitive(ist::ivec3);
istSerializePrimitive(ist::ivec4);
istSerializePrimitive(ist::uvec2);
istSerializePrimitive(ist::uvec3);
istSerializePrimitive(ist::uvec4);
istSerializePrimitive(ist::mat2);
istSerializePrimitive(ist::mat3);
istSerializePrimitive(ist::mat4);
istSerializePrimitive(ist::simdvec4);
istSerializePrimitive(ist::simdmat4);

#endif // ist_Base_Types_h
