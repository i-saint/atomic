#ifndef __atomic_Fraction__
#define __atomic_Fraction__


namespace atomic {




struct __declspec(align(16)) FractionData
{
    union {
        struct {
            uint32 id;
            uint32 alive;
            uint32 index;
            uint32 xindex;
            uint32 yindex;
            uint32 zindex;
            uint32 frame;
            uint32 color;
        };
        XMVECTOR param[2];
    };
    XMVECTOR pos;
    XMVECTOR vel;
    XMVECTOR axis1;
    XMVECTOR axis2;
};


class FractionCollider;
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

    class Interframe : boost::noncopyable
    {
    private:
        typedef stl::vector<FractionCollider*> FractionColliderCont;

        FractionColliderCont m_colliders;
        Task_FractionUpdate* m_update_task;

    public:
        Interframe();
        ~Interframe();
        void                    resizeColliders(uint32 block_num);          // thread unsafe
        FractionCollider*       getCollider(uint32 block);                  // thread safe
        Task_FractionUpdate*    getUpdateTask() { return m_update_task; }
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
    void sortXOrder();
    void sortYOrder();
    void sortZOrder();

    void generateVertex(uint32 block);
};








} // namespace atomic

#endif // __atomic_Fraction__
