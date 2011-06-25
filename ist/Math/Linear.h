#ifndef __ist_Graphic_Linear_H__
#define __ist_Graphic_Linear_H__

namespace ist {

struct OBB
{
    XMVECTOR m_ur;
    XMVECTOR m_bl;
    XMMATRIX m_mat;
};

struct Sphere
{
    union {
        struct {
            float x,y,z,r;
        };
        XMVECTOR v;
    };
};

} // namespace ist
#endif // __ist_Graphic_Linear_H__
