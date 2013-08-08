#include "istPCH.h"
#include "Timer.h"
#include <mmsystem.h>

namespace ist {

#ifdef ist_env_Windows

Timer::Timer()
{
    ::QueryPerformanceFrequency( &m_freq );
    reset();
}

void Timer::reset()
{
    m_start.QuadPart = 0;
    ::QueryPerformanceCounter( &m_start );
}

float32 Timer::getElapsedMillisec() const
{
    LARGE_INTEGER end;
    ::QueryPerformanceCounter( &end );
    return ((float32)(end.QuadPart - m_start.QuadPart) / (float32)m_freq.QuadPart)*1000.0f;
}

float32 Timer::getElapsedMicrosec() const
{
    LARGE_INTEGER end;
    ::QueryPerformanceCounter( &end );
    return ((float32)(end.QuadPart - m_start.QuadPart) / (float32)m_freq.QuadPart)*1000000.0f;
}

float32 Timer::getElapsedNanosec() const
{
    LARGE_INTEGER end;
    ::QueryPerformanceCounter( &end );
    return ((float32)(end.QuadPart - m_start.QuadPart) / (float32)m_freq.QuadPart)*1000000000.0f;
}

istAPI uint32 GetTick()
{
    return ::timeGetTime();
}

#else // todo

#endif // 

} // namespace ist
