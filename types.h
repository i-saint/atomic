#ifndef __atomic_Types__
#define __atomic_Types__

#include "ist/ist.h"
using namespace ist::graphics;

namespace atomic
{

typedef char                int8;
typedef short               int16;
typedef int                 int32;
typedef long long           int64;
typedef unsigned char       uint8;
typedef unsigned short      uint16;
typedef unsigned int        uint32;
typedef unsigned long long  uint64;

typedef float   float32;
typedef double  float64;

using glm::vec2;
using glm::vec3;
using glm::vec4;
using glm::mat2;
using glm::mat3;
using glm::mat4;


// __m128 を直接 eastl::vector とかに格納すると、
// アライメントが 16 に揃ってないアドレスに SSE のコピー命令でアクセスしてクラッシュしたりする。
// なので eastl::vector に格納するときは下記の構造体で代用し、キャストでなんとかする。
struct __declspec(align(16)) quadword { char c[16]; };
typedef stl::vector<quadword> QWordVector;


using ist::Task;
using ist::TaskScheduler;


typedef unsigned int id_t;
enum {
    CHR_FRACTION    = 0x80000000,
    CHR_BULLET      = 0x40000000,
    CHR_GROUND      = 0x20000000,
    INVALID_ID = 0xffffffff,
};




typedef stl::allocator PermanentAllocator;
//typedef stack_allocator< stack_allocator<stl::allocator> > FrameScopedAllocator; // あとでこちらに
typedef stl::allocator FrameScopedAllocator;


class PerformanceCounter
{
private:
    LARGE_INTEGER m_start;
    LARGE_INTEGER m_end;
    uint32 m_fps;

public:
    PerformanceCounter()
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

    void count()
    {
        ++m_fps;
        float32 elapsed = getElapsedMillisecond();
        if(elapsed > 1000.0f) {
            IST_PRINT("%dfps (avg. %.2fms)\n", m_fps, elapsed/m_fps);
            reset();
        }
    }
};


} // namespace atomic
#endif // __atomic_Types__
