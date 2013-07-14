#ifndef ist_Math_Misc_h
#define ist_Math_Misc_h

namespace ist {

/// メンバ関数ポインタとかは普通のキャストが効かないので union で強引にキャスト
template<class To, class From>
inline To force_cast(From v)
{
    union {
        From from;
        To to;
    } u = {v};
    return u.to;
}

template<class T>
inline T clamp(T v, T minmum, T maximum)
{
    return stl::min<T>(stl::max<T>(v, minmum), maximum);
}

template<class T>
inline T div_ceil(T v, T d)
{
    return v/d + (v%d==0 ? 0 : 1);
}


template<class F, class CharT>
inline void separate_scan(CharT *str, CharT separator, const F &f)
{
    for(;;) {
        size_t l = 0;
        while(str[l]!=separator && str[l]!=0) { ++l; }
        f(str, l);
        if(str[l]==separator) { str+=l+1; }
        else { break; }
    }
}


/// T: float,vec[234]
template<class T>
inline T interpolate_linear(T v1, T v2, float32 u)
{
    T d = v2-v1;
    return v1 + d*u;
}

/// T: float,vec[234]
template<class T>
inline T interpolate_sin90(T v1, T v2, float32 u)
{
    float32 d = v2-v1;
    return v1 + d*std::sin(glm::radians(90.0f*u));
}

/// T: float,vec[234]
template<class T>
inline T interpolate_cos90inv(T v1, T v2, float32 u)
{
    float32 d = v2-v1;
    return v1 + d*(1.0f-std::cos(glm::radians(90.0f*u)));
}

/// T: float,vec[234]
template<class T>
inline T interpolate_cos180inv(T v1, T v2, float32 u)
{
    float32 d = v2-v1;
    return v1 + d*(1.0f-(std::cos(glm::radians(180.0f*u))/2.0f+0.5f));
}

/// T: float,vec[234]
template<class T>
inline T interpolate_pow(T v1, T v2, float32 p, float32 u)
{
    T d = v2-v1;
    return v1 + d*std::pow(u, p);
}

/// T: float,vec[234]
template<class T>
inline T interpolate_bezier(T v1, T v1out, T v2_in, T v2, float32 u)
{
    float w[4] = {
        (1.0f-u) * (1.0f-u) * (1.0f-u),
        u * (1.0f-u) * (1.0f-u)*3.0f,
        u * u * (1.0f-u)*3.0f,
        u * u * u,
    };
    return v1*w[0] + v1out*w[1] + v2_in*w[2] + v2*w[3];
}

} // namespace ist

#endif // ist_Math_Misc_h
