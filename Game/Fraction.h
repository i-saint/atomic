#ifndef __atomic_Fraction_h__
#define __atomic_Fraction_h__

#include "FractionCollider.h"
#include "GPGPU/SPH.cuh"

namespace atomic {

class FractionGrid;
class Task_FractionUpdateAsync;
class Task_FractionDraw;




struct __declspec(align(16)) FractionData
{
    union {
        struct {
            uint32 id;
            uint32 cell;
            uint32 end_frame;
            float32 density;
        };
        XMVECTOR param[1];
    };
    XMVECTOR pos;
    XMVECTOR vel;
    XMVECTOR accel;
};
BOOST_STATIC_ASSERT(sizeof(FractionData)==64);


struct FractionGridData
{
    uint32 begin_index;
    uint32 end_index;
};


class FractionSet : boost::noncopyable
{
private:
    typedef stl::vector<FractionData, FrameAllocator> DataCont;
    typedef stl::vector<FractionGridData, FrameAllocator> HashGridCont;
    typedef stl::vector<Message_GenerateFraction, FrameAllocator> GenMessageCont;

    Task_FractionUpdateAsync            *m_task_asyncupdate;
    SPHParticle                         m_particles[SPH_MAX_PARTICLE_NUM];
    SPHSphericalGravity                 m_sgravity[ SPH_MAX_SPHERICAL_GRAVITY_NUM ];
    thrust::host_vector<SPHParticle>    m_spawn;
    HashGridCont    m_grid;
    uint32          m_idgen;

public:
    FractionSet();
    ~FractionSet();

    void initialize();
    void serialize(Serializer& s) const;
    void deserialize(Deserializer& s);

    void update();
    void draw() const;
    void sync() const;
    void updateAsync();

    const SPHParticle* getFraction(uint32 i) const { return &m_particles[i]; }

public:
    void updateSPH();
    void processMessage();
};




} // namespace atomic
#endif // __atomic_Fraction_h__
