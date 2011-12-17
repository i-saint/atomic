#ifndef __atomic_Game_Force_inl__
#define __atomic_Game_Force_inl__

namespace atomic {


template<class Data>
void AffectSphericalGravity(const SphericalGravity g, Data *data, uint32 num_data)
{

}

class Force_SphericalGravity : public IForce
{
private:
    simdvec4 m_center;
    float32 m_gravity;

public:
    Force_SphericalGravity()
        : m_gravity(0.1f)
    {
        m_center = simdvec4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    void updateFractions(FractionData* beg, FractionData* end)
    {
        simdvec4 center1 = m_center;
        simdvec4 gravity = simdvec4(m_gravity);
        soavec34 center = soavec34(center1, center1, center1);
        for(; beg<end; beg+=4) {
            soavec34 pos = ist::soa_transpose34(beg[0].pos, beg[1].pos, beg[2].pos, beg[3].pos);
            soavec34 dist = center - pos;
            simdvec4 len = ist::soa_length34(dist);
            soavec34 dir = dist / len;
            soavec34 vel = soavec34(beg[0].vel, beg[1].vel, beg[2].vel, beg[3].vel);
            vel = (vel + (dir * gravity));

            soavec34 velv = ist::soa_transpose34(vel.x, vel.y, vel.z);
            beg[0].vel = velv.sv[0];
            beg[1].vel = velv.sv[1];
            beg[2].vel = velv.sv[2];
            beg[3].vel = velv.sv[3];
        }
    }
};

class Force_InvertedSphere
{

};


} // namespace atomic
#endif // __atomic_Game_Force_inl__
