#include "istPCH.h"
#include "Timer.h"

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
#else // todo

#endif // 

} // namespace ist
