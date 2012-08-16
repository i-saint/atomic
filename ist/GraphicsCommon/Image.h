#ifndef __ist_GraphicsCommon_Image_h__
#define __ist_GraphicsCommon_Image_h__

#include <vector>
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
    bool operator ==(const TR<T> &t) const { return (r==t.r); }
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
    bool operator ==(const TRG<T> &t) const { return (r==t.r && g==t.g); }
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
    TRGB<T>(T _r, T _g, T _b, T _a) : r(_r), g(_g), b(_b) {}
    bool operator ==(const TRGB<T> &t) const { return (r==t.r && g==t.g && b==t.b); }
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
    bool operator ==(const TRGBA<T> &t) const { return (r==t.r && g==t.g && b==t.b && a==t.a); }
    T& operator [](int32 i) { return v[i]; }
    const T& operator [](int32 i) const { return v[i]; }
};

typedef TR<uint8> R_U8;
typedef TRG<uint8> RG_U8;
typedef TRGB<uint8> RGB_U8;
typedef TRGBA<uint8> RGBA_U8;
typedef TR<int8> R_I8;
typedef TRG<int8> RG_I8;
typedef TRGB<int8> RGB_I8;
typedef TRGBA<int8> RGBA_I8;
typedef TR<float32> R_F32;
typedef TRG<float32> RG_F32;
typedef TRGB<float32> RGB_F32;
typedef TRGBA<float32> RGBA_F32;

enum ImageFormat {
    IF_Unknown,
    IF_R_U8,  IF_RG_U8,  IF_RGB_U8,  IF_RGBA_U8,
    IF_R_I8,  IF_RG_I8,  IF_RGB_I8,  IF_RGBA_I8,
    IF_R_F32, IF_RG_F32, IF_RGB_F32, IF_RGBA_F32,
};

template<class T> struct GetImageFotmatID;
template<> struct GetImageFotmatID<char*>   { enum { Result=IF_Unknown }; };
template<> struct GetImageFotmatID<R_U8>    { enum { Result=IF_R_U8 }; };
template<> struct GetImageFotmatID<RG_U8>   { enum { Result=IF_RG_U8 }; };
template<> struct GetImageFotmatID<RGB_U8>  { enum { Result=IF_RGB_U8 }; };
template<> struct GetImageFotmatID<RGBA_U8> { enum { Result=IF_RGBA_U8 }; };
template<> struct GetImageFotmatID<R_I8>    { enum { Result=IF_R_I8 }; };
template<> struct GetImageFotmatID<RG_I8>   { enum { Result=IF_RG_I8 }; };
template<> struct GetImageFotmatID<RGB_I8>  { enum { Result=IF_RGB_I8 }; };
template<> struct GetImageFotmatID<RGBA_I8> { enum { Result=IF_RGBA_I8 }; };
template<> struct GetImageFotmatID<R_F32>   { enum { Result=IF_R_F32 }; };
template<> struct GetImageFotmatID<RG_F32>  { enum { Result=IF_RG_F32 }; };
template<> struct GetImageFotmatID<RGB_F32> { enum { Result=IF_RGB_F32 }; };
template<> struct GetImageFotmatID<RGBA_F32>{ enum { Result=IF_RGBA_F32 }; };

inline RGBA_F32 ToF32(const RGBA_U8 &b) { return RGBA_F32(float32(b.r)/255.0f, float32(b.g)/255.0f, float32(b.b)/255.0f, float32(b.a)/255.0f ); }
inline RGBA_U8 ToU8(const RGBA_F32 &b) { return RGBA_U8(uint8(b.r*255.0f), uint8(b.g*255.0f), uint8(b.b*255.0f), uint8(b.a*255.0f) ); }

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

        void setPath(const stl::string &path)   { m_path=path; }
        void setFileType(FileType v)               { m_filetype=v; }
        void setPngCompressLevel(uint8 v)       { m_png_compress_level=v; }
        void setJpgQuality(uint8 v)             { m_jpg_quality=v; }

        const stl::string& getPath() const  { return m_path; }
        FileType getFileType() const           { return m_filetype; }
        uint8 getPngCompressLevel() const   { return m_png_compress_level; }
        uint8 getJpgQuality() const         { return m_jpg_quality; }

    private:
        stl::string m_path;
        FileType m_filetype;
        uint8 m_png_compress_level;
        uint8 m_jpg_quality;
    };

public:
    Image() : m_format(IF_Unknown), m_width(0), m_height(0) {}

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

    size_t width() const { return m_width; }
    size_t height() const { return m_height; }
    char* data() { return &m_data[0]; }
    const char* data() const { return &m_data[0]; }

    bool load(const char *path);
    bool load(IBinaryStream &f, const IOConfig &conf=IOConfig());
    bool save(const char *path) const;
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

} // namespace ist

#endif // __ist_GraphicsCommon_Image_h__
