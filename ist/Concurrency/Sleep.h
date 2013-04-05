#ifndef ist_Concurrency_Sleep_h
#define ist_Concurrency_Sleep_h
namespace ist {

void YieldCPU();
void MiliSleep(uint32 milisec);
void MicroSleep(uint32 microsec);
void NanoSleep(uint32 nanosec);

} // namespace ist
#endif // ist_Concurrency_Sleep_h
