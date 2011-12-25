#ifndef __atomic_Util__
#define __atomic_Util__

struct sphRigidSphere;
struct sphRigidBox;

namespace atomic {

    vec4 GenRotateAxis();
    void CreateRigidSphere(sphRigidSphere &o, EntityHandle h, const vec4& pos, float32 r);
    void CreateRigidBox(sphRigidBox &o, EntityHandle h, const mat4& t, const vec4 &size);

} // namespace atomic
#endif // __atomic_Util__
