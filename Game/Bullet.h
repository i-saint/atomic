#ifndef __atomic_Bullet__
#define __atomic_Bullet__


namespace atomic {

class Task_BulletBeforeDraw;
class Task_BulletAfterDraw;
class Task_BulletDraw;
class Task_BulletCopy;



class Bullet_OctahedronSet : boost::noncopyable
{
public:
    struct __declspec(align(16)) BulletData
    {
        XMVECTOR pos;
        XMVECTOR vel;
        union {
            struct {
                float32 height;
                float32 random;
                uint32 lifetime;
            };
            XMVECTOR param;
        };
    };

private:
    typedef stl::vector<BulletData> BulletCont;

    BulletCont m_data;

public:
    Bullet_OctahedronSet();
    ~Bullet_OctahedronSet();

    uint32 getNumBullets() const { return m_data.size(); }
    const BulletData* getBullet(uint32 i) const { return &m_data[i]; }

public:
    void taskBeforeDraw();
    void taskAfterDraw();
    void taskDraw();
    void taskCopy();
};


class Bullet_OctahedronSet;


class BulletSet : boost::noncopyable
{
public:
    class __declspec(align(16)) Interframe : boost::noncopyable
    {
    private:

    public:
        Interframe();
        ~Interframe();
    };

    static void InitializeInterframe();
    static void FinalizeInterframe();
    static Interframe* getInterframe() { return s_interframe; }

private:
    static Interframe *s_interframe;


private:
    Bullet_OctahedronSet m_octahedron;

    const BulletSet *m_prev;
    BulletSet *m_next;

public:
    BulletSet();
    ~BulletSet();

    void initialize();
    void serialize(Serializer& s) const;
    void deserialize(Deserializer& s);

    void update();
    void sync() const;

    Task* getDrawTask();

    uint32 getNumBlocks() const;
    void setNext(BulletSet *next);
    BulletSet* getNext() { return m_next; }
    const BulletSet* getPrev() const { return m_prev; }

    // 以下非同期更新タスク用
public:
    void taskBeforeDraw();
    void taskBeforeDraw(uint32 block);
    void taskAfterDraw();
    void taskDraw() const;
    void taskCopy(FractionSet *dst) const;

private:
    void processMessage();
    void move(uint32 block);
    void collisionTest(uint32 block);
    void collisionProcess(uint32 block);
    void updateGrid();
};


} // namespace atomic
#endif // __atomic_Bullet__
