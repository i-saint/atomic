#ifndef __atomic_Types__
#define __atomic_Types__

#include "ist/ist.h"

namespace atomic {

using namespace ist::i3d;

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

using glm::vec2;
using glm::vec3;
using glm::vec4;
using glm::mat2;
using glm::mat3;
using glm::mat4;
using glm::ivec2;
using glm::ivec3;
using glm::ivec4;

using ist::simdvec4;
using ist::simdmat4;
using ist::soavec24;
using ist::soavec34;
using ist::soavec44;

typedef ist::bostream Serializer;
typedef ist::bistream Deserializer;


using ist::Task;
using ist::TaskScheduler;
using ist::SpinLock;
using ist::SFMT;
typedef ist::Variant16 variant;


typedef uint32 EntityHandle;




typedef stl::allocator PermanentAllocator;
typedef stl::allocator FrameAllocator;


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
#endif // __atomic_Types__
