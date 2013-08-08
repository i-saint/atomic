#ifndef ist_Concurrency_Timer_h
#define ist_Concurrency_Timer_h

#include "ist/Config.h"

namespace ist {

class istAPI Timer
{
public:
    Timer();
    void reset();
    float32 getElapsedMillisec() const;
    float32 getElapsedMicrosec() const;
    float32 getElapsedNanosec() const;

private:
#ifdef ist_env_Windows
    LARGE_INTEGER m_freq;
    LARGE_INTEGER m_start;
#else // todo
#endif // 
};

istAPI uint32 GetTick();

} // namespace ist

#endif // ist_Concurrency_Timer_h
