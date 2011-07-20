#ifndef __ist_Graphic_Linear_h__
#define __ist_Graphic_Linear_h__

namespace ist {

// union に入れたいなどのシチュエーションが予想されるため、コンストラクタを持たない構造にしています

struct __declspec(align(16)) Point
{
    union {
        struct {
            float x, y, z;
        };
        XMVECTOR v;
    };

    static Point create() { Point ret={{0.0f, 0.0f, 0.0f}}; return ret; }
    static Point create(float x, float y, float z) { Point ret={{x,y,z}}; return ret; }
    static Point create(XMVECTOR v) { Point ret; ret.v=v; return ret; }

    XMVECTOR getPosition() const { return v; }
};


struct __declspec(align(16)) Sphere
{
    union {
        struct {
            float x, y, z, r;
        };
        XMVECTOR v;
    };

    static Sphere create() { Sphere ret={{0.0f, 0.0f, 0.0f, 0.0f}}; return ret; }
    static Sphere create(float x, float y, float z, float r) { Sphere ret={{x,y,z,r}}; return ret; }
    static Sphere create(XMVECTOR v, float r) { Sphere ret; ret.v=v; ret.r=r; return ret; }
    static Sphere create(const Point &p, float r) { Sphere ret; ret.v=p.v; ret.r=r; return ret; }

    XMVECTOR getPosition() const { return v; }
    float getRadius() const { return r; }
};


struct __declspec(align(16)) Segment
{
    Point begin_pos;
    Point end_pos;

    static Segment create() { Segment ret={Point::create(), Point::create()}; return ret; }
    static Segment create(const Point &a, const Point &b) { Segment ret={a,b}; return ret; }

    XMVECTOR getBeginPos() const { return begin_pos.v; }
    XMVECTOR getEndPos() const { return end_pos.v; }
};


struct __declspec(align(16)) Capsule
{
    Sphere begin_pos;
    Point end_pos;

    static Capsule create() { Capsule ret={Sphere::create(), Point::create()}; return ret; }
    static Capsule create(const Point &a, const Point &b, float r) { Capsule ret={Sphere::create(a,r), b}; return ret; }

    XMVECTOR getBeginPos() const { return begin_pos.v; }
    XMVECTOR getEndPos() const { return end_pos.v; }
    float getRadius() {}
};


struct __declspec(align(16)) Plane
{
    union {
        struct {
            float nx, ny, nz, d;
        };
        XMVECTOR v;
    };

    static Plane create() { Plane ret={0.0f, 0.0f, 0.0f, 0.0f}; return ret; }
    static Plane create(XMVECTOR n, float d) { Plane ret; ret.v=n; ret.d=d; return ret; }
};


struct __declspec(align(16)) Triangle
{
    Point pos[3];

    static Triangle create() { Triangle ret={{Point::create(), Point::create(), Point::create()}}; return ret;}
    static Triangle create(const Point &p0, const Point &p1, const Point &p2) { Triangle ret={p0,p1,p2}; return ret; }
};


struct __declspec(align(16)) AABB
{
    Point ur;
    Point bl;

    static AABB create() { AABB ret={Point::create(), Point::create()}; return ret; }
    static AABB create(const Point &ur, const Point &bl) { AABB ret={ur, bl}; return ret; }
};


struct __declspec(align(16)) OBB
{
    AABB aabb;
    XMMATRIX mat;

    static OBB create() {}
    static OBB create(const Point &ur, const Point &bl, const XMMATRIX &mat) { OBB ret={AABB::create(ur,bl), mat}; return ret; }
    static OBB create(const AABB &aabb, const XMMATRIX &mat) { OBB ret={aabb, mat}; return ret; }
};


struct __declspec(align(16)) Frustum
{
    XMMATRIX m_mat;

    static Frustum create() { Frustum ret={XMMatrixIdentity()}; return ret; }
    static Frustum create(const XMMATRIX &mat) { Frustum ret={mat}; return ret; }
};



namespace linear {
    bool GetDistance(const Point& point, const Point &point1, float *distance=NULL);
    bool GetDistance(const Point& point, const Sphere &sphere, float *distance=NULL);
    bool GetDistance(const Point& point, const Capsule &capsule, float *distance=NULL);
    bool GetDistance(const Point& point, const Plane &plane, float *distance=NULL);
    bool GetDistance(const Point& point, const Triangle &triangle, float *distance=NULL);
    bool GetDistance(const Point& point, const AABB &aabb, float *distance=NULL);
    bool GetDistance(const Point& point, const OBB &obb, float *distance=NULL);

    bool GetDistance(const Sphere& sphere, const Sphere &sphere1, float *distance=NULL);
    bool GetDistance(const Sphere& sphere, const Capsule &capsule, float *distance=NULL);
    bool GetDistance(const Sphere& sphere, const Plane &plane, float *distance=NULL);
    bool GetDistance(const Sphere& sphere, const Triangle &triangle, float *distance=NULL);
    bool GetDistance(const Sphere& sphere, const AABB &aabb, float *distance=NULL);
    bool GetDistance(const Sphere& sphere, const OBB &obb, float *distance=NULL);

    bool GetDistance(const Capsule& capsule, const Capsule &capsule1, float *distance=NULL);
    bool GetDistance(const Capsule& capsule, const Plane &plane, float *distance=NULL);
    bool GetDistance(const Capsule& capsule, const Triangle &triangle, float *distance=NULL);
    bool GetDistance(const Capsule& capsule, const AABB &aabb, float *distance=NULL);
    bool GetDistance(const Capsule& capsule, const OBB &obb, float *distance=NULL);

    bool GetDistance(const Plane& plane, const Plane &plane1, float *distance=NULL);
    bool GetDistance(const Plane& plane, const Triangle &triangle, float *distance=NULL);
    bool GetDistance(const Plane& plane, const AABB &aabb, float *distance=NULL);
    bool GetDistance(const Plane& plane, const OBB &obb, float *distance=NULL);

    bool GetDistance(const Triangle& triangle, const Triangle &triangle1, float *distance=NULL);
    bool GetDistance(const Triangle& triangle, const AABB &aabb, float *distance=NULL);
    bool GetDistance(const Triangle& triangle, const OBB &obb, float *distance=NULL);
} // namespace linear
} // namespace ist
#endif // __ist_Graphic_Linear_h__
