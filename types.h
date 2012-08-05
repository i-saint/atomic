#ifndef __atomic_Types__
#define __atomic_Types__

#include "ist/ist.h"

namespace atomic {

typedef char                int8;
typedef short               int16;
typedef int                 int32;
typedef long long           int64;
typedef unsigned char       uint8;
typedef unsigned short      uint16;
typedef unsigned int        uint32;
typedef unsigned long long  uint64;
typedef half                float16;
typedef float               float32;
typedef double              float64;

using ist::vec2;
using ist::vec3;
using ist::vec4;
using ist::mat2;
using ist::mat3;
using ist::mat4;
using ist::ivec2;
using ist::ivec3;
using ist::ivec4;
using ist::uvec2;
using ist::uvec3;
using ist::uvec4;
using ist::uvec2;
using ist::uvec3;
using ist::uvec4;
using ist::simdvec4;
using ist::simdmat4;
using ist::soavec24;
using ist::soavec34;
using ist::soavec44;

typedef ist::bostream Serializer;
typedef ist::bistream Deserializer;


using ist::SpinMutex;
using ist::Task;
using ist::TaskScheduler;
using ist::SFMT;
using ist::FrustumPlanes;
using ist::AABB;
typedef ist::Variant16 variant;

namespace i3d = ist::i3dgl;
using namespace ist::i3dgl;

typedef uint32 EntityHandle;
typedef uint32 CollisionHandle;

typedef stl::allocator PermanentAllocator;
typedef stl::allocator FrameAllocator;


enum ATOMIC_ERROR {
    ATERR_NOERROR,
    ATERR_CREATEWINDOW_FAILED,
    ATERR_CHANGEDISPLAYSETTINGS_FAILED,
    ATERR_OPENAL_OPENDEVICE_FAILED,
    ATERR_OPENAL_CREATECONTEXT_FAILED,
    ATERR_OPENGL_330_IS_NOT_SUPPORTED,
    ATERR_OPENGL_INITIALIZATION_FAILED,
    ATERR_CUDA_NO_DEVICE,
    ATERR_CUDA_INSUFFICIENT_DRIVER,
};

class IAtomicGameModule : public boost::noncopyable
{
public:
    virtual ~IAtomicGameModule() {}

    // フレーム開始時に呼ばれる。
    virtual void frameBegin()=0;

    // 同期更新。dt の単位はフレーム。
    virtual void update(float32 dt)=0;

    // 非同期更新。dt の単位はフレーム。
    // asyncupdate() と draw() は並列に走るため、draw() に asyncupdate() の結果に依存する処理がある場合、
    // draw() の中で asyncupdate() の完了を待つ必要がある。
    virtual void asyncupdate(float32 dt)=0;

    // 描画用データを作って Renderer へ送る。(i3d::DeviceContext などを直接触る処理があってはならない)
    // asyncupdate() と draw() は並列に走るため、draw() に asyncupdate() の結果に依存する処理がある場合、
    // draw() の中で asyncupdate() の完了を待つ必要がある。
    virtual void draw()=0;

    // フレーム終了時に呼ばれる。
    // 非同期更新処理がある場合、この中で完了を待つこと。(フレームを跨ぐ処理があってはならない)
    virtual void frameEnd()=0;

    // todo: serialize/deserialize
};



class PerformanceCounter
{
private:
    LARGE_INTEGER m_start;
    LARGE_INTEGER m_end;
    uint32 m_fps;
    uint32 m_prev_fps;
    float32 m_avg_fps;

public:
    PerformanceCounter()
        : m_fps(0)
        , m_prev_fps(0)
        , m_avg_fps(0.0f)
    {
        reset();
    }

    void reset()
    {
        m_start.QuadPart = 0;
        m_end.QuadPart = 0;
        m_fps = 0;
        ::QueryPerformanceCounter( &m_start );
    }

    float32 getElapsedMillisecond()
    {
        LARGE_INTEGER freq;
        ::QueryPerformanceCounter( &m_end );
        ::QueryPerformanceFrequency( &freq );
        return ((float32)(m_end.QuadPart - m_start.QuadPart) / (float32)freq.QuadPart)*1000.0f;
    }

    float32 getAverageFPS() const { return (float32)m_prev_fps; }

    void count()
    {
        ++m_fps;
        float32 elapsed = getElapsedMillisecond();
        if(elapsed > 1000.0f) {
            m_prev_fps = m_fps;
            m_avg_fps = elapsed/m_fps;
            reset();
        }
    }

    void print()
    {
        istPrint("%dfps (avg. %.2fms)\n", m_prev_fps, m_avg_fps);
    }
};

} // namespace atomic

#define atomicSafeRelease(Obj) if(Obj){Obj->release();Obj=NULL;}

#endif // __atomic_Types__
