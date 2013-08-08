#ifndef atm_Util_h
#define atm_Util_h


namespace atm {

    template<class T>
    inline void clear(T &v) {
        v = T();
    }
    template<class T, size_t L>
    inline void clear(T (&v)[L]) {
        for(size_t i=0; i<L; ++i) { v[i]=T(); }
    }

    template<class T, class F>
    inline void each(T &v, const F &f) {
        std::for_each(v.begin(), v.end(), f);
    }

    template<class T, size_t L, class F>
    inline void each(T (&v)[L], const F &f) {
        for(size_t i=0; i<L; ++i) { f(v[i]); }
    }

    template<class T, class F>
    inline void each_with_index(T &v, const F &f) {
        size_t idx=0;
        for(auto i=v.begin(); i!=v.end(); ++i, ++idx) { f(*i, idx); }
    }

    template<class T, size_t L, class F>
    inline void each_with_index(T &v, const F &f) {
        for(size_t i=0; i!=L; ++i) { f(v[i], i); }
    }

    template<class T, class F>
    inline void parallel_each(T &v, size_t block_size, const F &f) {
        size_t num_tasks = ceildiv(v.size(), block_size);
        ist::parallel_for(size_t(0), num_tasks,
            [&](size_t bi) {
                auto first = v.begin() + (bi*block_size);
                auto last = v.begin() + (std::min<size_t>((bi+1)*block_size, v.size()));
                for(; first!=last; ++first) {
                    f(*first);
                }
        });
    }

    template<class T, class F>
    inline void parallel_each_with_block_index(T &v, size_t block_size, const F &f) {
        size_t num_tasks = ceildiv(v.size(), block_size);
        ist::parallel_for(size_t(0), num_tasks,
            [&](size_t bi) {
                auto first = v.begin() + (bi*block_size);
                auto last = v.begin() + (std::min<size_t>((bi+1)*block_size, v.size()));
                for(; first!=last; ++first) {
                    f(*first, bi);
                }
        });
    }

    template<class T, class F>
    inline void parallel_each_with_index(T &v, size_t block_size, const F &f) {
        size_t num_tasks = ceildiv(v.size(), block_size);
        ist::parallel_for(size_t(0), num_tasks,
            [&](size_t bi) {
                size_t index = bi*block_size;
                auto first = v.begin() + (bi*block_size);
                auto last = v.begin() + (std::min<size_t>((bi+1)*block_size, v.size()));
                for(; first!=last; ++first, ++index) {
                    f(*first, index);
                }
        });
    }

    template<class T, class F>
    inline void erase(T &v, const F &f) { v.erase(std::remove_if(v.begin(), v.end(), f), v.end()); }

    template<class T, class F>
    inline auto find(T &v, const F &f)->decltype(v.begin()) { return std::find_if(v.begin(), v.end(), f); }

    template<class IntType>
    inline IntType ceildiv(IntType a, IntType b)
    {
        return a/b + (a%b==0 ? 0 : 1);
    }

    inline int32 moddiv(float32 &v, float32 m)
    {
        float32 d = v/m;
        v = std::fmod(v,m);
        return (int32)d;
    }

    template<class T>
    inline T absmin(T a, T b)
    {
        return std::min<T>(abs(a), abs(b));
    }


    template<class F>
    void scan(const stl::string &str, const std::regex &reg, const F &f)
    {
        std::cmatch m;
        size_t pos = 0;
        for(;;) {
            if(std::regex_search(str.c_str()+pos, m, reg)) {
                f(m);
                pos += m.position()+m.length();
            }
            else {
                break;
            }
        }
    }


    atmAPI void FillScreen(const vec4 &color);

    atmAPI vec2 GenRandomVector2();
    atmAPI vec3 GenRandomVector3();
    atmAPI vec2 GenRandomUnitVector2();
    atmAPI vec3 GenRandomUnitVector3();
    atmAPI void CreateDateString(char *buf, uint32 len);
    atmAPI bool mkdir(const char *path);
    atmAPI void HTTPGet(
        const char *url,
        const std::function<void (std::istream &res)> &on_complete,
        const std::function<void (int32)> &on_fail=std::function<void (int32)>());
    atmAPI void HTTPGetAsync(
        const char *url,
        const std::function<void (std::istream &res)> &on_complete,
        const std::function<void (int32)> &on_fail=std::function<void (int32)>());

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


} // namespace atm

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

#endif // atm_Util_h
