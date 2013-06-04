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


namespace atomic {
    void InitializeCrashReporter();
    void FinalizeCrashReporter();
} // namespace atomic
using namespace atomic;

void ExecApp(int argc, char* argv[])
{
    // クラッシュさせるテスト
    //*static_cast<int*>(NULL) = 0;

    atomic::AtomicApplication app;
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

    atomic::InitializeCrashReporter();
istCrashReportBegin
    ExecApp(argc, argv);
istCrashReportRescue
istCrashReportEnd
    atomic::FinalizeCrashReporter();

    dpFinalize();
    return 0;
}

