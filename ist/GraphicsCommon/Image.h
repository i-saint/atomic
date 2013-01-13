#ifndef ist_GraphicsCommon_Image_h
#define ist_GraphicsCommon_Image_h

#include "ist/Base.h"

namespace ist {


template <typename T>
struct TR
{
    union {
        struct { T r; };
        T v[1];
    };
    TR<T>() : r(0) {}
    TR<T>(T _r) : r(_r) {}
    TR<T> operator + (TR<T> &t) const { return TR<T>(r+t.r); }
    TR<T> operator - (TR<T> &t) const { return TR<T>(r-t.r); }
    TR<T> operator * (TR<T> &t) const { return TR<T>(r*t.r); }
    TR<T> operator / (TR<T> &t) const { return TR<T>(r/t.r); }
    TR<T>& operator +=(TR<T> &t) { *this=*this+t; return *this; }
    TR<T>& operator -=(TR<T> &t) { *this=*this-t; return *this; }
    TR<T>& operator *=(TR<T> &t) { *this=*this*t; return *this; }
    TR<T>& operator /=(TR<T> &t) { *this=*this/t; return *this; }
    bool operator ==(const TR<T> &t) const { return (r==t.r); }
    bool operator !=(const TR<T> &t) const { return !(*this==t); }
    T& operator [](int32 i) { return v[i]; }
    const T& operator [](int32 i) const { return v[i]; }
};

template <typename T>
struct TRG
{
    union {
        struct { T r,g; };
        T v[2];
    };
    TRG<T>() : r(0), g(0) {}
    TRG<T>(T _r, T _g) : r(_r), g(_g) {}
    TRG<T> operator + (TRG<T> &t) const { return TRG<T>(r+t.r, g+t.g); }
    TRG<T> operator - (TRG<T> &t) const { return TRG<T>(r-t.r, g-t.g); }
    TRG<T> operator * (TRG<T> &t) const { return TRG<T>(r*t.r, g*t.g); }
    TRG<T> operator / (TRG<T> &t) const { return TRG<T>(r/t.r, g/t.g); }
    TRG<T>& operator +=(TRG<T> &t) { *this=*this+t; return *this; }
    TRG<T>& operator -=(TRG<T> &t) { *this=*this-t; return *this; }
    TRG<T>& operator *=(TRG<T> &t) { *this=*this*t; return *this; }
    TRG<T>& operator /=(TRG<T> &t) { *this=*this/t; return *this; }
    bool operator ==(const TRG<T> &t) const { return (r==t.r && g==t.g); }
    bool operator !=(const TRG<T> &t) const { return !(*this==t); }
    T& operator [](int32 i) { return v[i]; }
    const T& operator [](int32 i) const { return v[i]; }
};

template <typename T>
struct TRGB
{
    union {
        struct { T r,g,b; };
        T v[3];
    };
    TRGB<T>() : r(0), g(0), b(0) {}
    TRGB<T>(T _r, T _g, T _b) : r(_r), g(_g), b(_b) {}
    TRGB<T> operator + (TRGB<T> &t) const { return TRGB<T>(r+t.r, g+t.g, b+t.b); }
    TRGB<T> operator - (TRGB<T> &t) const { return TRGB<T>(r-t.r, g-t.g, b-t.b); }
    TRGB<T> operator * (TRGB<T> &t) const { return TRGB<T>(r*t.r, g*t.g, b*t.b); }
    TRGB<T> operator / (TRGB<T> &t) const { return TRGB<T>(r/t.r, g/t.g, b/t.b); }
    TRGB<T>& operator +=(TRGB<T> &t) { *this=*this+t; return *this; }
    TRGB<T>& operator -=(TRGB<T> &t) { *this=*this-t; return *this; }
    TRGB<T>& operator *=(TRGB<T> &t) { *this=*this*t; return *this; }
    TRGB<T>& operator /=(TRGB<T> &t) { *this=*this/t; return *this; }
    bool operator ==(const TRGB<T> &t) const { return (r==t.r && g==t.g && b==t.b); }
    bool operator !=(const TRGB<T> &t) const { return !(*this==t); }
    T& operator [](int32 i) { return v[i]; }
    const T& operator [](int32 i) const { return v[i]; }
};

template <typename T>
struct TRGBA
{
    union {
        struct { T r,g,b,a; };
        T v[4];
    };
    TRGBA<T>() : r(0), g(0), b(0), a(0) {}
    TRGBA<T>(T _r, T _g, T _b, T _a) : r(_r), g(_g), b(_b), a(_a) {}
    TRGBA<T> operator + (TRGBA<T> &t) const { return TRGBA<T>(r+t.r, g+t.g, b+t.b, a+t.a ); }
    TRGBA<T> operator - (TRGBA<T> &t) const { return TRGBA<T>(r-t.r, g-t.g, b-t.b, a-t.a ); }
    TRGBA<T> operator * (TRGBA<T> &t) const { return TRGBA<T>(r*t.r, g*t.g, b*t.b, a*t.a ); }
    TRGBA<T> operator / (TRGBA<T> &t) const { return TRGBA<T>(r/t.r, g/t.g, b/t.b, a/t.a ); }
    TRGBA<T>& operator +=(TRGBA<T> &t) { *this=*this+t; return *this; }
    TRGBA<T>& operator -=(TRGBA<T> &t) { *this=*this-t; return *this; }
    TRGBA<T>& operator *=(TRGBA<T> &t) { *this=*this*t; return *this; }
    TRGBA<T>& operator /=(TRGBA<T> &t) { *this=*this/t; return *this; }
    bool operator ==(const TRGBA<T> &t) const { return (r==t.r && g==t.g && b==t.b && a==t.a); }
    bool operator !=(const TRGBA<T> &t) const { return !(*this==t); }
    T& operator [](int32 i) { return v[i]; }
    const T& operator [](int32 i) const { return v[i]; }
};

typedef TR<uint8> R_8U;
typedef TRG<uint8> RG_8U;
typedef TRGB<uint8> RGB_8U;
typedef TRGBA<uint8> RGBA_8U;
typedef TR<int8> R_8I;
typedef TRG<int8> RG_8I;
typedef TRGB<int8> RGB_8I;
typedef TRGBA<int8> RGBA_8I;
typedef TR<float32> R_32F;
typedef TRG<float32> RG_32F;
typedef TRGB<float32> RGB_32F;
typedef TRGBA<float32> RGBA_32F;

enum ImageFormat {
    IF_Unknown,
    IF_R8U,  IF_RG8U,  IF_RGB8U,  IF_RGBA8U,
    IF_R8I,  IF_RG8I,  IF_RGB8I,  IF_RGBA8I,
    IF_R32F, IF_RG32F, IF_RGB32F, IF_RGBA32F,
    IF_RGBA_DXT1,
    IF_RGBA_DXT3,
    IF_RGBA_DXT5,
};

template<class T> struct GetImageFotmatID;
template<> struct GetImageFotmatID<char>    { enum { Result=IF_Unknown }; };
template<> struct GetImageFotmatID<R_8U>    { enum { Result=IF_R8U }; };
template<> struct GetImageFotmatID<RG_8U>   { enum { Result=IF_RG8U }; };
template<> struct GetImageFotmatID<RGB_8U>  { enum { Result=IF_RGB8U }; };
template<> struct GetImageFotmatID<RGBA_8U> { enum { Result=IF_RGBA8U }; };
template<> struct GetImageFotmatID<R_8I>    { enum { Result=IF_R8I }; };
template<> struct GetImageFotmatID<RG_8I>   { enum { Result=IF_RG8I }; };
template<> struct GetImageFotmatID<RGB_8I>  { enum { Result=IF_RGB8I }; };
template<> struct GetImageFotmatID<RGBA_8I> { enum { Result=IF_RGBA8I }; };
template<> struct GetImageFotmatID<R_32F>   { enum { Result=IF_R32F }; };
template<> struct GetImageFotmatID<RG_32F>  { enum { Result=IF_RG32F }; };
template<> struct GetImageFotmatID<RGB_32F> { enum { Result=IF_RGB32F }; };
template<> struct GetImageFotmatID<RGBA_32F>{ enum { Result=IF_RGBA32F }; };

inline R_32F ToF32(const R_8U &b) { return R_32F(float32(b.r)/255.0f); }
inline RG_32F ToF32(const RG_8U &b) { return RG_32F(float32(b.r)/255.0f, float32(b.g)/255.0f); }
inline RGB_32F ToF32(const RGB_8U &b) { return RGB_32F(float32(b.r)/255.0f, float32(b.g)/255.0f, float32(b.b)/255.0f); }
inline RGBA_32F ToF32(const RGBA_8U &b) { return RGBA_32F(float32(b.r)/255.0f, float32(b.g)/255.0f, float32(b.b)/255.0f, float32(b.a)/255.0f); }
inline R_8U ToU8(const R_32F &b) { return R_8U(uint8(b.r*255.0f)); }
inline RG_8U ToU8(const RG_32F &b) { return RG_8U(uint8(b.r*255.0f), uint8(b.g*255.0f)); }
inline RGB_8U ToU8(const RGB_32F &b) { return RGB_8U(uint8(b.r*255.0f), uint8(b.g*255.0f), uint8(b.b*255.0f)); }
inline RGBA_8U ToU8(const RGBA_32F &b) { return RGBA_8U(uint8(b.r*255.0f), uint8(b.g*255.0f), uint8(b.b*255.0f), uint8(b.a*255.0f)); }

template<class T>
inline TRGBA<T> GetShininess(const TRGBA<T> &b) { return TRGBA<T>(T(float32(b.r)*0.299f), T(float32(b.g)*0.587f), T(float32(b.b)*0.114f), b.a); }



class istInterModule Image
{
public:
    enum FileType
    {
        FileType_Auto,
        FileType_BMP,
        FileType_TGA,
        FileType_PNG,
        FileType_JPG,
        FileType_DDS,
        FileType_Unknown,
    };

    class IOConfig
    {
    public:
        IOConfig() : m_filetype(FileType_Auto), m_png_compress_level(9), m_jpg_quality(100)
        {}

        void setFileType(FileType v)               { m_filetype=v; }
        void setPngCompressLevel(uint8 v)       { m_png_compress_level=v; }
        void setJpgQuality(uint8 v)             { m_jpg_quality=v; }

        FileType getFileType() const           { return m_filetype; }
        uint8 getPngCompressLevel() const   { return m_png_compress_level; }
        uint8 getJpgQuality() const         { return m_jpg_quality; }

    private:
        FileType m_filetype;
        uint8 m_png_compress_level;
        uint8 m_jpg_quality;
    };

public:
    Image() : m_format(IF_Unknown), m_width(0), m_height(0) {}

    ImageFormat getFormat() const { return (ImageFormat)m_format; }

    void clear()
    {
        m_width = 0;
        m_height = 0;
        m_format = IF_Unknown;
        m_data.clear();
    }

    template<class T> void resize(uint32 w, uint32 h)
    {
        m_width = w;
        m_height = h;
        m_format = GetImageFotmatID<T>::Result;
        m_data.resize(w*h*sizeof(T));
    }

    // 圧縮データなど、w,h とデータサイズが完全には対応しないデータ用
    void setup(ImageFormat fmt, uint32 w, uint32 h, uint32 data_size)
    {
        m_width = w;
        m_height = h;
        m_format = fmt;
        m_data.resize(data_size);
    }

    template<class T> T& get(uint32 y, uint32 x)
    {
        istAssert(GetImageFotmatID<T>::Result==m_format, "フォーマット指定ミス\n");
        return ((T*)data())[width()*y + x];
    }

    template<class T> const T& get(uint32 y, uint32 x) const
    {
        return const_cast<Image*>(this)->get<T>(y, x);
    }

    template<class T> T* begin()            { return (T*)data(); }
    template<class T> const T* begin() const{ return (T*)data(); }
    template<class T> T* end()              { return (T*)data()+(width()*height()); }
    template<class T> const T* end() const  { return (T*)data()+(width()*height()); }

    size_t width() const    { return m_width; }
    size_t height() const   { return m_height; }
    size_t size() const     { return m_data.size(); }
    char* data()            { return &m_data[0]; }
    const char* data() const{ return &m_data[0]; }

    bool load(const char *path, const IOConfig &conf=IOConfig());
    bool load(IBinaryStream &f, const IOConfig &conf=IOConfig());
    bool save(const char *path, const IOConfig &conf=IOConfig()) const;
    bool save(IBinaryStream &f, const IOConfig &conf) const;

private:
    bool loadBMP(IBinaryStream &f, const IOConfig &conf);
    bool saveBMP(IBinaryStream &f, const IOConfig &conf) const;
    bool loadTGA(IBinaryStream &f, const IOConfig &conf);
    bool saveTGA(IBinaryStream &f, const IOConfig &conf) const;
    bool loadPNG(IBinaryStream &f, const IOConfig &conf);
    bool savePNG(IBinaryStream &f, const IOConfig &conf) const;
    bool loadJPG(IBinaryStream &f, const IOConfig &conf);
    bool saveJPG(IBinaryStream &f, const IOConfig &conf) const;
    bool loadDDS(IBinaryStream &f, const IOConfig &conf);
    bool saveDDS(IBinaryStream &f, const IOConfig &conf) const;

    stl::vector<char> m_data;
    int32 m_format;
    uint32 m_width, m_height;
};



//
// utilities
//

template<class SrcT, class DstT, class F>
inline void TExtract(const Image &src, Image &dst, const F &f)
{
    dst.resize<DstT>(src.width(), src.height());
    stl::transform(src.begin<SrcT>(), src.end<SrcT>(), dst.begin<DstT>(), f);
}

template<class SrcT, class DstT>
inline void TExtractAlpha(const Image &src, Image &dst)
{
    TExtract<SrcT, DstT>(src, dst, [&](const SrcT &s){ return DstT(s.a); });
}

inline bool ExtractAlpha(const Image &src, Image &dst)
{
    switch(src.getFormat()) {
    case IF_RGBA8U:  TExtractAlpha<RGBA_8U, R_8U>(src, dst); return true;
    case IF_RGBA8I:  TExtractAlpha<RGBA_8I, R_8I>(src, dst); return true;
    case IF_RGBA32F: TExtractAlpha<RGBA_32F, R_32F>(src, dst); return true;
    }
    return false;
}


template<class SrcT, class DstT>
inline void TExtractRed(const Image &src, Image &dst)
{
    TExtract<SrcT, DstT>(src, dst, [&](const SrcT &s){ return DstT(s.r); });
}

inline bool ExtractRed(const Image &src, Image &dst)
{
    switch(src.getFormat()) {
    case IF_R8U:  TExtractRed<R_8U, R_8U>(src, dst); return true;
    case IF_RG8U:  TExtractRed<RG_8U, R_8U>(src, dst); return true;
    case IF_RGB8U:  TExtractRed<RGB_8U, R_8U>(src, dst); return true;
    case IF_RGBA8U:  TExtractRed<RGBA_8U, R_8U>(src, dst); return true;
    case IF_R8I:  TExtractRed<R_8I, R_8I>(src, dst); return true;
    case IF_RG8I:  TExtractRed<RG_8I, R_8I>(src, dst); return true;
    case IF_RGB8I:  TExtractRed<RGB_8I, R_8I>(src, dst); return true;
    case IF_RGBA8I:  TExtractRed<RGBA_8I, R_8I>(src, dst); return true;
    case IF_R32F:  TExtractRed<R_32F, R_32F>(src, dst); return true;
    case IF_RG32F:  TExtractRed<RG_32F, R_32F>(src, dst); return true;
    case IF_RGB32F:  TExtractRed<RGB_32F, R_32F>(src, dst); return true;
    case IF_RGBA32F:  TExtractRed<RGBA_32F, R_32F>(src, dst); return true;
    }
    return false;
}

} // namespace ist

#endif // ist_GraphicsCommon_Image_h
