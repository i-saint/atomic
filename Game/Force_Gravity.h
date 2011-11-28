#ifndef __atomic_Game_Force_h__
#define __atomic_Game_Force_h__

namespace atomic {


class Force_SphericalGravity : public IForce
{
private:
    float4 m_center;
    float32 m_gravity;

public:
    Force_SphericalGravity()
        : m_gravity(0.1f)
    {
        m_center = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
};

class Force_InvertedSphere
{

};


} // namespace atomic
#endif // __atomic_Game_Force_h__
