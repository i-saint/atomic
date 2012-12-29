#ifndef __atomic_Util__
#define __atomic_Util__


namespace atomic {

    struct CollisionSphere;
    struct CollisionBox;

    void FillScreen(const vec4 &color);

    vec4 GenRandomVector2();
    vec4 GenRandomVector3();
    vec4 GenRandomUnitVector2();
    vec4 GenRandomUnitVector3();
    void UpdateCollisionSphere(CollisionSphere &o, const vec4& pos, float32 r);
    void UpdateCollisionBox(CollisionBox &o, const mat4& t, const vec4 &size);

    vec4 GetNearestPlayerPosition(const vec4 &pos);
    void ShootSimpleBullet(EntityHandle owner, const vec4 &pos, const vec4 &vel);

    void CreateDateString(char *buf, uint32 len);

    template<class T>
    inline T clamp(T v, T vmin, T vmax) { return stl::min<T>(stl::max<T>(v, vmin), vmax); }

    inline void assign_float2(float32 (&dst)[2], const float32 (&src)[2]) { for(size_t i=0; i<2; ++i) { dst[i]=src[i]; } }
    inline void assign_float3(float32 (&dst)[3], const float32 (&src)[3]) { for(size_t i=0; i<3; ++i) { dst[i]=src[i]; } }
    inline void assign_float4(float32 (&dst)[4], const float32 (&src)[4]) { for(size_t i=0; i<4; ++i) { dst[i]=src[i]; } }
    inline void assign_float2(float32 (&dst)[2], float32 x, float32 y) { dst[0]=x; dst[1]=y; }
    inline void assign_float3(float32 (&dst)[3], float32 x, float32 y, float32 z) { dst[0]=x; dst[1]=y; dst[2]=z; }
    inline void assign_float4(float32 (&dst)[4], float32 x, float32 y, float32 z, float32 w) { dst[0]=x; dst[1]=y; dst[2]=z; dst[3]=w; }

    // 同一値が並ぶコンテナを巡回する際、同じ数値の連続をスキップする iterator adapter
    // (例: {1,2,2,3,3,3,1,1} を巡回すると 1,2,3,1,end の順に結果が返る)
    // 巡回のみの場合、遅い stl::unique() のそこそこ高速な代替手段となる
    template<class IteratorType>
    class unique_iterator
    {
    public:
        typedef IteratorType base_t;
        typedef typename stl::iterator_traits<base_t>::value_type value_type;
        typedef typename stl::iterator_traits<base_t>::difference_type difference_type;
        typedef typename stl::iterator_traits<base_t>::value_type value_type;
        typedef typename stl::iterator_traits<base_t>::pointer pointer;
        typedef typename stl::iterator_traits<base_t>::reference reference;
        typedef typename stl::iterator_traits<base_t>::iterator_category iterator_category;

    private:
        base_t m_iter, m_end;

    public:
        unique_iterator(const base_t &v, const base_t &e) : m_iter(v), m_end(e) {}
        unique_iterator& operator++() {
            const value_type &last = *(m_iter++);
            while(m_iter!=m_end && *m_iter==last) { ++m_iter; }
            return *this;
        }
        unique_iterator operator++(int) {
            unique_iterator r = *this;
            operator++();
            return r;
        }
        value_type& operator*() { return *m_iter; }
        base_t& operator->() { return m_iter; }
        bool operator==(const base_t &v) const { return m_iter==v; }
        bool operator!=(const base_t &v) const { return m_iter!=v; }
    };


} // namespace atomic

namespace glm {
namespace detail {

// glm の中に実体がないようなので…
__forceinline fvec4SIMD operator* (fmat4x4SIMD const & M, fvec4SIMD const & V)
{
    // Splat x,y,z and w
    fvec4SIMD vTempX = _mm_shuffle_ps(V.Data, V.Data, _MM_SHUFFLE(0,0,0,0));
    fvec4SIMD vTempY = _mm_shuffle_ps(V.Data, V.Data, _MM_SHUFFLE(1,1,1,1));
    fvec4SIMD vTempZ = _mm_shuffle_ps(V.Data, V.Data, _MM_SHUFFLE(2,2,2,2));
    fvec4SIMD vTempW = _mm_shuffle_ps(V.Data, V.Data, _MM_SHUFFLE(3,3,3,3));
    // Mul by the matrix
    vTempX = vTempX * M.Data[0];
    vTempY = vTempY * M.Data[1];
    vTempZ = vTempZ * M.Data[2];
    vTempW = vTempW * M.Data[3];
    // Add them all together
    vTempX = vTempX + vTempY;
    vTempZ = vTempZ + vTempW;
    vTempX = vTempX + vTempZ;
    return vTempX;
}

} // namespace detail
} // namespace glm

#endif // __atomic_Util__
