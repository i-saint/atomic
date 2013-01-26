#ifndef ist_Concurrency_Process_h
#define ist_Concurrency_Process_h

#include "../Config.h"

namespace ist {

#if defined(ist_env_Windows) && !defined(ist_env_Master)

// vsvars32.bat が呼ばれた状態で command を実行します
bool ExecVCTool(const char *command);

#endif // defined(ist_env_Windows) && !defined(ist_env_Master)

} // namespace ist
#endif // ist_Concurrency_Process_h
