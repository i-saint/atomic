#ifndef __ist_Graphics_types__
#define __ist_Graphics_types__

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/simd_vec4.hpp>
#include <glm/gtx/simd_mat4.hpp>

namespace ist {

using glm::vec2;
using glm::vec3;
using glm::vec4;
using glm::ivec2;
using glm::ivec3;
using glm::ivec4;
using glm::mat2;
using glm::mat3;
using glm::mat4;

typedef glm::simdVec4 simdvec4;
typedef glm::simdMat4 simdmat4;

} // namespace ist
#endif // __ist_Graphics_types__
