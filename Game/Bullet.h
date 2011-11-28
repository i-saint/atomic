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
    void draw();
    void sync();
    void updateAsync();

    Task* getDrawTask();

    uint32 getNumBlocks() const;
};


} // namespace atomic
#endif // __atomic_Bullet_h__
