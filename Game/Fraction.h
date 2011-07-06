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
        XMVECTOR param[2];
    };
    XMVECTOR pos;
    XMVECTOR vel;
};


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

    struct XHolder
    {
        uint32 index;
        float32 x;
    };
    struct YHolder
    {
        uint32 index;
        float32 y;
    };
    struct ZHolder
    {
        uint32 index;
        float32 z;
    };
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
    typedef stl::vector<FractionData, FrameScopedAllocator> DataCont;
    typedef stl::vector<XHolder, FrameScopedAllocator> XCont;
    typedef stl::vector<YHolder, FrameScopedAllocator> YCont;
    typedef stl::vector<ZHolder, FrameScopedAllocator> ZCont;
    typedef stl::vector<Message_GenerateFraction, FrameScopedAllocator> GenMessageCont;

    DataCont    m_data;
    XCont       m_xorder;
    YCont       m_yorder;
    ZCont       m_zorder;
    GenMessageCont m_gen_mes;

    FractionSet *m_prev, *m_next;
    FrameScopedAllocator *m_alloc;
    uint32 m_idgen;
    uint32 m_num_dead;

public:
    FractionSet(FractionSet* prev, FrameScopedAllocator *alloc);
    ~FractionSet();

    void update();
    void sync();
    void flushMessage();
    void processMessage();
    void draw();

    FractionSet* getPrev() { return m_prev; }
    FractionSet* getNext() { return m_next; }
    uint32 getRequiredMemoryOnNextFrame();

    void pushGenerateMessage(const Message_GenerateFraction& mes) { m_gen_mes.push_back(mes); }
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
