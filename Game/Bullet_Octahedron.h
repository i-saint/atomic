#ifndef __atomic_Bullet_Octahedron_h__
#define __atomic_Bullet_Octahedron_h__

#include "Bullet_Subset.h"

namespace atomic {

class Bullet_OctahedronSet : public BulletSubset
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
    void update();
    void draw();
    void updateAsync() const;
};

}
#endif // __atomic_Bullet_Octahedron_h__
