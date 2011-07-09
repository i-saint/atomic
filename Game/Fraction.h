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

class Task_FractionBeforeDraw;
class Task_FractionAfterDraw;

class __declspec(align(16)) FractionSet : boost::noncopyable
{
public:
    static const size_t BLOCK_SIZE;
    static const float RADIUS;
    static const float BOUNCE;
    static const float MAX_VEL;
    static const float DECELERATE;

    struct __declspec(align(16)) GridRange
    {
        XMVECTOR range_min;
        XMVECTOR range_max;
    };

    class __declspec(align(16)) Interframe : boost::noncopyable
    {
    private:
        typedef stl::vector<QWordVector*> CollisionResultCont;

        /// m_collision_results のメモリ配置は↓のようになっています
        /// [ResultHeader][Result*ResultHeader::num_collision]...[ResultHeader(num_collision=0)]
        CollisionResultCont m_collision_results;
        Task_FractionBeforeDraw *m_task_beforedraw;
        Task_FractionAfterDraw *m_task_afterdraw;
        FractionGrid *m_grid;

    public:
        Interframe();
        ~Interframe();
        void                        resizeColliders(uint32 block_num);          // thread unsafe
        QWordVector*                getCollisionResultContainer(uint32 uint32) { return m_collision_results[uint32]; }
        Task_FractionBeforeDraw*    getTask_BeforeDraw() { return m_task_beforedraw; }
        Task_FractionAfterDraw*     getTask_AfterDraw() { return m_task_afterdraw; }
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
    typedef stl::vector<Message_GenerateFraction, FrameAllocator> GenMessageCont;
    typedef stl::vector<GridRange> GridRangeCont;

    DataCont    m_data;
    GridRangeCont m_grid_range;
    FractionSet *m_prev;
    uint32 m_idgen;

public:
    FractionSet();
    ~FractionSet();

    void initialize(FractionSet* prev);

    void update();
    void draw();
    void sync();

    const FractionSet* getPrev() const { return m_prev; }
    const FractionData* getFraction(uint32 i) const { return &m_data[i]; }

    // 以下非同期更新タスク用
public:
    uint32 getNumBlocks() const;
    void processMessage();
    void updateState(uint32 block);
    void updateGrid();

    void move(uint32 block);
    void collisionTest(uint32 block);
    void collisionProcess(uint32 block);
};








} // namespace atomic

#endif // __atomic_Fraction__
