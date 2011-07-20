#ifndef __atomic_Bullet_Subset_h__
#define __atomic_Bullet_Subset_h__

namespace atomic {

class BulletSubset : boost::noncopyable
{
private:
    uint32 m_block_index;

public:
    BulletSubset() : m_block_index(0) {}
    virtual ~BulletSubset() {}
    virtual void taskBeforeDraw()=0;
    virtual void taskAfterDraw()=0;
    virtual void taskDraw() const=0;
    virtual void taskCopy() const=0;

    void setBlockIndex(uint32 v) { m_block_index=v; }
    uint32 getBlockIndex() const { return m_block_index; }
};

}
#endif // __atomic_Bullet_Subset_h__
