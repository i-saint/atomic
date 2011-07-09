#ifndef __atomic_Fraction__
#define __atomic_Fraction__

#include "FractionCollider.h"

namespace atomic {




struct __declspec(align(16)) FractionData
{
    union {
        struct {
            uint32 id;
            uint32 alive;
            uint32 index;
            uint32 frame;
        };
        XMVECTOR param[1];
    };
    XMVECTOR pos;
    XMVECTOR vel;
};
BOOST_STATIC_ASSERT(sizeof(FractionData)==48);


class FractionCollider;
class FractionGrid;
class FractionRenderer;

class Task_FractionUpdate;

class FractionSet : boost::noncopyable
{
public:
    static const size_t BLOCK_SIZE;
    static const float RADIUS;
    static const float BOUNCE;
    static const float MAX_VEL;
    static const float DECELERATE;

    struct GridRange
    {
        XMVECTOR range_min;
        XMVECTOR range_max;
    };

    class Interframe : boost::noncopyable
    {
    private:
        typedef stl::vector<QWordVector*> CollisionResultCont;
        typedef stl::vector<GridRange> GridRangeCont;

        /// m_collision_results のメモリ配置は↓のようになっています
        /// [ResultHeader][Result*ResultHeader::num_collision]...[ResultHeader(num_collision=0)]
        CollisionResultCont m_collision_results;
        GridRangeCont m_grid_range;
        Task_FractionUpdate *m_update_task;
        FractionGrid *m_grid;

    public:
        Interframe();
        ~Interframe();
        void                    resizeColliders(uint32 block_num);          // thread unsafe
        QWordVector*            getCollisionResultContainer(uint32 uint32) { return m_collision_results[uint32]; }
        GridRange*              getGridRange(uint32 uint32) { return &m_grid_range[uint32]; }
        Task_FractionUpdate*    getUpdateTask() { return m_update_task; }
        FractionGrid*           getGrid() { return m_grid; }
    };


private:
    static Interframe *s_interframe;

public:
    static void InitializeInterframe();
    static void FinalizeInterframe();
    static Interframe* getInterframe() { return s_interframe; }


private:
    typedef stl::vector<FractionData, FrameAllocator> DataCont;
    typedef stl::vector<Message_GenerateFraction, FrameAllocator> GenMessageCont;

    DataCont    m_data;

    FractionSet *m_prev;
    uint32 m_idgen;

public:
    FractionSet();
    ~FractionSet();

    void initialize(FractionSet* prev, FrameAllocator& alloc);

    void update();
    void draw();

    FractionSet* getPrev() { return m_prev; }
    uint32 getRequiredMemoryOnNextFrame();

    FractionData* getFraction(uint32 i) { return &m_data[i]; }

    // 以下非同期更新タスク用
public:
    uint32 getNumBlocks() const;
    void processGenerateMessage();
    void move(uint32 block);
    void collisionTest(uint32 block);
    void collisionProcess(uint32 block);
    void updateGrid();

    void generateVertex(uint32 block);
};








} // namespace atomic

#endif // __atomic_Fraction__
