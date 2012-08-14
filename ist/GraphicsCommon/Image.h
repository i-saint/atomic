#ifndef __ist_GraphicsCommon_Image_h__
#define __ist_GraphicsCommon_Image_h__

#include <vector>
#include "ist/Base.h"

namespace ist {


template <typename T>
struct RGBA
{
    union
    {
        struct { T r,g,b,a; };
        T v[4];
    };

    RGBA<T> operator + (RGBA<T> &t) const { return RGBA<T>(r+t.r, g+t.g, b+t.b, a+t.a ); }
    RGBA<T> operator - (RGBA<T> &t) const { return RGBA<T>(r-t.r, g-t.g, b-t.b, a-t.a ); }
    RGBA<T> operator * (RGBA<T> &t) const { return RGBA<T>(r*t.r, g*t.g, b*t.b, a*t.a ); }
    RGBA<T> operator / (RGBA<T> &t) const { return RGBA<T>(r/t.r, g/t.g, b/t.b, a/t.a ); }
    RGBA<T>& operator +=(RGBA<T> &t) { *this=*this+t; return *this; }
    RGBA<T>& operator -=(RGBA<T> &t) { *this=*this-t; return *this; }
    RGBA<T>& operator *=(RGBA<T> &t) { *this=*this*t; return *this; }
    RGBA<T>& operator /=(RGBA<T> &t) { *this=*this/t; return *this; }
    bool operator ==(const RGBA<T> &t) const { return (r==t.r && g==t.g && b==t.b && a==t.a); }
    bool operator !=(const RGBA<T> &t) const { return !(*this==t); }
    T& operator [](int32 i) { return v[i]; }
    const T& operator [](int32 i) const { return v[i]; }

    RGBA<T>() : r(0), g(0), b(0), a(0) {}
    RGBA<T>(T _r, T _g, T _b, T _a) : r(_r), g(_g), b(_b), a(_a) {}
};

typedef RGBA<uint8> bRGBA;
typedef RGBA<float32> fRGBA;
inline fRGBA TofRGBA(const bRGBA& b) { return fRGBA(float32(b.r)/255.0f, float32(b.g)/255.0f, float32(b.b)/255.0f, float32(b.a)/255.0f ); }
inline bRGBA TobRGBA(const fRGBA& b) { return bRGBA(uint8(b.r*255.0f), uint8(b.g*255.0f), uint8(b.b*255.0f), uint8(b.a*255.0f) ); }

template<class T>
inline RGBA<T> GetShininess(const RGBA<T>& b) { return RGBA<T>(T(float32(b.r)*0.299f), T(float32(b.g)*0.587f), T(float32(b.b)*0.114f), b.a); }




template<class T>
class Array2D
{
public:
    typedef T value_type;
    typedef stl::vector<value_type> container_type;
    typedef typename container_type::iterator iterator;
    typedef typename container_type::const_iterator const_iterator;

    Array2D() : m_width(0), m_height(0) {}
    Array2D(uint32 w, uint32 h) : m_width(0), m_height(0) { resize(w, h); }

    void assign(const Array2D& v, uint32 x, uint32 y, uint32 width, uint32 height)
    {
        resize(width, height);
        copy(0, 0, v, x, y, width, height);
    }

    void copy(uint32 sx, uint32 sy, const Array2D& v, uint32 x, uint32 y, uint32 w, uint32 h) {
        for(uint32 i=y; i<y+h; ++i) {
            if(sy>=height() || i>=v.width())
                break;
            uint32 sbx = sx;
            for(uint32 j=x; j<x+w; ++j) {
                if(sx>=height() || j>=v.width())
                    break;
                (*this)[sy][sx] = v[i][j];
                ++sx;
            }
            sx = sbx;
            ++sy;
        }
    }

    uint32 size() const { return m_data.size(); }
    uint32 width() const { return m_width; }
    uint32 height() const { return m_height; }
    bool empty() const { return m_data.empty(); }
    void resize(uint32 w, uint32 h) { m_width=w; m_height=h; m_data.resize(w*h); }
    void clear() { m_width=0; m_height=0; m_data.clear(); }
    iterator begin() { return m_data.begin(); }
    iterator end()   { return m_data.end(); }
    const_iterator begin() const { return m_data.begin(); }
    const_iterator end()   const { return m_data.end(); }

    bool is_valid(uint32 x, uint32 y) { return x<width() && y<height(); }
    T& at(uint32 i) { return m_data[i]; }
    const T& at(uint32 i) const { return m_data[i]; }
    T& at(uint32 x, uint32 y) { return m_data[m_width*y+x]; }
    const T& get(uint32 x, uint32 y) const { return m_data[m_width*y+x]; }

    // bmp[y][x] 
    T* operator [] (uint32 i) { return &m_data[m_width*i]; }
    const T* operator [] (uint32 i) const { return &m_data[m_width*i]; }

private:
    uint32 m_width, m_height;
    container_type m_data;
};


class Image : public Array2D<bRGBA>
{
public:
    enum
    {
        FORMAT_AUTO,
        FORMAT_BMP,
        FORMAT_TGA,
        FORMAT_PNG,
        FORMAT_JPG,
        FORMAT_UNKNOWN,
    };

    class IOConfig
    {
    public:
        IOConfig() : m_format(FORMAT_AUTO), m_png_compress_level(9), m_jpg_quality(100)
        {}

        void setPath(const stl::string& path)  { m_path=path; }
        void setFormat(uint8 v)           { m_format=v; }
        void setPngCompressLevel(uint8 v) { m_png_compress_level=v; }
        void setJpgQuality(uint8 v)       { m_jpg_quality=v; }

        const stl::string& getPath() const     { return m_path; }
        uint8 getFormat() const           { return m_format; }
        uint8 getPngCompressLevel() const { return m_png_compress_level; }
        uint8 getJpgQuality() const       { return m_jpg_quality; }

    private:
        stl::string m_path;
        uint8 m_format;
        uint8 m_png_compress_level;
        uint8 m_jpg_quality;
    };

public:
    Image() {}
    explicit Image(uint32 w, uint32 h) { resize(w, h); }

    bool load(const stl::string& filename);
    bool load(const IOConfig& conf);
    bool load(std::istream& f, const IOConfig& conf);
    bool load(std::streambuf& f, const IOConfig& conf);

    bool save(const stl::string& filename) const;
    bool save(const IOConfig& conf) const;
    bool save(std::ostream& f, const IOConfig& conf) const;
    bool save(std::streambuf& f, const IOConfig& conf) const;

private:
    bool loadBMP(std::streambuf& f, const IOConfig& conf);
    bool saveBMP(std::streambuf& f, const IOConfig& conf) const;
    bool loadTGA(std::streambuf& f, const IOConfig& conf);
    bool saveTGA(std::streambuf& f, const IOConfig& conf) const;
    bool loadPNG(std::streambuf& f, const IOConfig& conf);
    bool savePNG(std::streambuf& f, const IOConfig& conf) const;
    bool loadJPG(std::streambuf& f, const IOConfig& conf);
    bool saveJPG(std::streambuf& f, const IOConfig& conf) const;
};

} // namespace ist

#endif // __ist_GraphicsCommon_Image_h__
