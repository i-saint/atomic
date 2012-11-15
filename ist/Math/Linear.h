#ifndef __ist_Graphic_Linear_h__
#define __ist_Graphic_Linear_h__


namespace ist {

// union に入れたいなどのシチュエーションが予想されるため、コンストラクタを持たない構造にしています

struct istAlign(16) Point
{
    vec4 v;

    static Point create() { Point ret; return ret; }
    static Point create(float x, float y, float z) { Point ret={vec4(x,y,z,0.0f)}; return ret; }
    static Point create(vec4 v) { Point ret={v}; return ret; }

    vec4 getPosition() const { return v; }
};


struct istAlign(16) Sphere
{
    vec4 v;

    static Sphere create() { Sphere ret; return ret; }
    static Sphere create(float x, float y, float z, float r) { Sphere ret={vec4(x, y, z, r)}; return ret; }
    static Sphere create(vec4 v, float r) { Sphere ret={vec4(v.x, v.y, v.z, r)}; return ret; }
    static Sphere create(const Point &p, float r) { Sphere ret={vec4(p.v.x, p.v.y, p.v.z, r)}; return ret; }

    vec4 getPosition() const { return vec4(v.x, v.y, v.z, 0.0f); }
    float getRadius() const { return v.w; }
};


struct istAlign(16) Segment
{
    Point begin_pos;
    Point end_pos;

    static Segment create() { Segment ret={Point::create(), Point::create()}; return ret; }
    static Segment create(const Point &a, const Point &b) { Segment ret={a,b}; return ret; }

    vec4 getBeginPos() const { return begin_pos.v; }
    vec4 getEndPos() const { return end_pos.v; }
};


struct istAlign(16) Capsule
{
    Sphere begin_pos;
    Point end_pos;

    static Capsule create() { Capsule ret={Sphere::create(), Point::create()}; return ret; }
    static Capsule create(const Point &a, const Point &b, float r) { Capsule ret={Sphere::create(a,r), b}; return ret; }

    vec4 getBeginPos() const { return begin_pos.v; }
    vec4 getEndPos() const { return end_pos.v; }
    float getRadius() { return begin_pos.getRadius(); }
};


struct istAlign(16) Plane
{
    vec4 v;

    static Plane create() { Plane ret={vec4(0.0f, 0.0f, 0.0f, 0.0f)}; return ret; }
    static Plane create(vec4 n, float d) { Plane ret; ret.v=n; ret.v.w=d; return ret; }
};


struct istAlign(16) Triangle
{
    Point pos[3];

    static Triangle create() { Triangle ret={{Point::create(), Point::create(), Point::create()}}; return ret;}
    static Triangle create(const Point &p0, const Point &p1, const Point &p2) { Triangle ret={p0,p1,p2}; return ret; }
};


struct istAlign(16) AABB
{
    Point ur;
    Point bl;

    static AABB create() { AABB ret={Point::create(), Point::create()}; return ret; }
    static AABB create(const Point &ur, const Point &bl) { AABB ret={ur, bl}; return ret; }
};



struct istAlign(16) OBB
{
    AABB aabb;
    mat4 mat;

    static OBB create() { OBB ret={AABB::create(), mat4()}; return ret; }
    static OBB create(const Point &ur, const Point &bl, const mat4 &mat) { OBB ret={AABB::create(ur,bl), mat}; return ret; }
    static OBB create(const AABB &aabb, const mat4 &mat) { OBB ret={aabb, mat}; return ret; }
};


struct istAlign(16) Frustum
{
    mat4 m_mat;
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
