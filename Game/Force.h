#ifndef __atomic_Game_Force__
#define __atomic_Game_Force__

namespace atomic
{

class FractionData;
class Enemy;

class IForce
{
public:
    virtual ~IForce() {}
    virtual updateFractions(FractionData* fraction, uint32 num_fraction) = 0;
};


class ForceSet
{
private:
    ForceSet *m_prev, *m_next

public:
    ForceSet(ForceSet* prev);
    ~ForceSet();

    void update();
    void sync();
    void flushMessage();
    void processMessage();
    void draw();

};


class Force_SphereGravity : public IForce
{
private:
    XMVECTOR m_center;
    float32 m_gravity;

public:
    Force_SphereGravity()
        : m_gravity(0.1f)
    {
        m_center = XMVectorSet(0.0f, 0.0f, 0.0f, 0.0f);
    }

    updateFractions(FractionData* beg, FractionData* end)
    {
        XMVECTOR center1 = m_center;
        XMVECTOR gravity = _mm_set1_ps(m_gravity);
        SOAVECTOR3 center = SOAVectorSet3(center1, center1, center1);
        for(; beg<end; beg+=4) {
            SOAVECTOR3 pos = SOAVectorTranspose3(beg[0].pos, beg[1].pos, beg[2].pos, beg[3].pos);
            SOAVECTOR3 dist = SOAVectorSubtract3(center, pos);
            XMVECTOR len = SOAVectorLength3(dist);
            SOAVECTOR3 dir = SOAVectorDivide3S(dist, len);
            SOAVECTOR3 vel = SOAVectorSet3(beg[0].vel, beg[1].vel, beg[2].vel, beg[3].vel);
            vel = SOAVectorAdd3(vel, SOAVectorMultiply3S(dir, gravity));

            SOAVECTOR3 velv = SOAVectorTranspose3(vel.x, vel.y, vel.z);
            beg[0].vel = velv.v[0];
            beg[1].vel = velv.v[1];
            beg[2].vel = velv.v[2];
            beg[3].vel = velv.v[3];
        }
    }
};

class Force_InvertedSphere
{

};


} // namespace atomic
#endif // __atomic_Game_Force__
