#ifndef __atomic_Bullet_h__
#define __atomic_Bullet_h__


namespace atomic {

class BulletSubset;

class Task_BulletBeforeDraw;
class Task_BulletAfterDraw;
class Task_BulletDraw;
class Task_BulletCopy;


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
    typedef stl::vector<BulletSubset*> BulletSubsets;
    BulletSubsets m_subsets;

    const BulletSet *m_prev;
    BulletSet *m_next;

public:
    BulletSet();
    ~BulletSet();

    void initialize();
    void serialize(Serializer& s) const;
    void deserialize(Deserializer& s);

    void update();
    void draw() const;
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
    void taskCopy(BulletSet *dst) const;

private:
    void processMessage();
    void move(uint32 block);
    void collisionTest(uint32 block);
    void collisionProcess(uint32 block);
    void updateGrid();
};


} // namespace atomic
#endif // __atomic_Bullet_h__
