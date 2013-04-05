#include "istPCH.h"
#include "ist.h"


namespace ist {

#ifdef ist_env_Windows
#pragma optimize("", off)
void forceLink()
{
    if(__argc!=0) { return; } // ここで絶対に return するはず
    assert(0);

    // ここに制御が来ることはない。
    // リンカに最適化で消してほしくない関数などをここに書く。
    ist::GetThisOfCaller();
    ist::IsStackMemory(NULL);
    ist::IsStaticMemory(NULL);
    ist::IsHeapMemory(NULL);
}
#pragma optimize("", on)
#else // ist_env_Windows

void forceLink()
{
}

#endif // ist_env_Windows

} // namespace ist
