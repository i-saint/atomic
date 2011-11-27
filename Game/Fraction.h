#ifndef __atomic_Fraction_h__
#define __atomic_Fraction_h__

#include "FractionCollider.h"
#include "GPGPU/SPH.cuh"

namespace atomic {

class FractionGrid;
class Task_FractionBeforeDraw;
class Task_FractionAfterDraw;
class Task_FractionDraw;
class Task_FractionCopy;




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
public:
    static const uint32 BLOCK_SIZE;
    static const float32 RADIUS;
    static const float32 BOUNCE;
    static const float32 MAX_VEL;
    static const float32 DECELERATE;
    static const float32 GRIDSIZE;
    static const uint32 NUM_GRID_CELL; // must be power of two

    class __declspec(align(16)) Interframe : boost::noncopyable
    {
    private:
        Task_FractionBeforeDraw *m_task_beforedraw;
        Task_FractionAfterDraw *m_task_afterdraw;
        Task_FractionCopy *m_task_copy;
        FractionGrid *m_grid;

    public:
        Interframe();
        ~Interframe();
        void                        resizeColliders(uint32 block_num);          // thread unsafe
        Task_FractionBeforeDraw*    getTask_BeforeDraw() { return m_task_beforedraw; }
        Task_FractionAfterDraw*     getTask_AfterDraw() { return m_task_afterdraw; }
        Task_FractionCopy*          getTask_Copy() { return m_task_copy; }
        FractionGrid*               getGrid() { return m_grid; }
    };


private:
    static Interframe *s_interframe;

public:
    static void InitializeInterframe();
    static void FinalizeInterframe();
    static Interframe* getInterframe() { return s_interframe; }


private:
    typedef stl::vector<FractionData, FrameAllocator> DataCont;
    typedef stl::vector<FractionGridData, FrameAllocator> HashGridCont;
    typedef stl::vector<Message_GenerateFraction, FrameAllocator> GenMessageCont;

    const FractionSet   *m_prev;
    FractionSet         *m_next;

    SPHParticle     m_particles[SPH_MAX_PARTICLE_NUM];
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

    uint32 getNumBlocks() const;
    void setNext(FractionSet *next);
    FractionSet* getNext() { return m_next; }
    const FractionSet* getPrev() const { return m_prev; }

    const SPHParticle* getFraction(uint32 i) const { return &m_particles[i]; }

    // 以下非同期更新タスク用
public:
    void updateSPH();

    void taskBeforeDraw();
    void taskBeforeDraw(uint32 block);
    void taskAfterDraw();
    void taskCopy(FractionSet *dst) const;

    void sphDensity(uint32 block);
    void sphForce(uint32 block);

private:
    void processMessage();
    void move(uint32 block);
    void updateGrid();
};




} // namespace atomic
#endif // __atomic_Fraction_h__
