#include "stdafx.h"
#include "types.h"
#include "Game/AtomicApplication.h"
#include "FunctionID.h"
#include "ist/Base/BinaryCall.h"

istImplementOperatorNewDelete();

#ifdef _M_X64
#   define dpPlatform "x64"
#else
#   define dpPlatform "Win32"
#endif
#ifdef _DEBUG
#   define dpConfiguration "Debug"
#else
#   define dpConfiguration "Release"
#endif
#define dpObjDir "_tmp/" dpConfiguration "/atomic" 


namespace atm {
    void atmInitializeCrashReporter();
    void atmFinalizeCrashReporter();
} // namespace atm
using namespace atm;

void ExecApp(int argc, char* argv[])
{
    // クラッシュさせるテスト
    //*static_cast<int*>(NULL) = 0;

    atm::AtomicApplication app;
    if(app.initialize(argc, argv)) {
        app.mainLoop();
    }
    app.finalize();
}

int istmain(int argc, char* argv[])
{
    dpInitialize();
    ist::forceLink();
    //test();

    atm::atmInitializeCrashReporter();
istCrashReportBegin
    ExecApp(argc, argv);
istCrashReportRescue
istCrashReportEnd
    atm::atmFinalizeCrashReporter();

    dpFinalize();
    return 0;
}

