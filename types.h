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


using ist::Task;
using ist::TaskScheduler;
using ist::SpinMutex;
using ist::SFMT;
typedef ist::Variant16 variant;

namespace i3d = ist::i3dgl;
using namespace ist::i3dgl;

typedef uint32 EntityHandle;

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
