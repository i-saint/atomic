#include "istPCH.h"
#include "Sleep.h"
#include "Timer.h"
namespace ist {


void YieldCPU()
{
#ifdef ist_env_Windows
    ::SwitchToThread();
#else // ist_env_Windows
    ::pthread_yield();
#endif // ist_env_Windows
}

void MiliSleep( uint32 milisec )
{
#ifdef ist_env_Windows
    return ::Sleep(milisec);
#else // ist_env_Windows
    return ::usleep(milisec*1000);
#endif // ist_env_Windows
}

void MicroSleep( uint32 microsec )
{
#ifdef ist_env_Windows
    if(microsec==0) { return; }

    Timer timer;
    while(uint32(timer.getElapsedMicrosec())<microsec) {
        YieldCPU();
    }
#else // ist_env_Windows
    return ::usleep(microsec);
#endif // ist_env_Windows
}

void NanoSleep( uint32 nanosec )
{
#ifdef ist_env_Windows
    if(nanosec==0) { return; }

    Timer timer;
    while(uint32(timer.getElapsedNanosec())<nanosec) {
        YieldCPU();
    }
#else // ist_env_Windows
    timespec ts = {0, nanosec};
    return ::nanosleep(&ts);
#endif // ist_env_Windows
}

} // namespace ist
