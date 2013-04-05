#include "stdafx.h"
#include "types.h"
#include "Game/AtomicApplication.h"
#include "FunctionID.h"
#include "ist/Base/BinaryCall.h"

istImplementOperatorNewDelete();

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
    ist::forceLink();
    //test();

    atomic::InitializeCrashReporter();
istCrashReportBegin
    ExecApp(argc, argv);
istCrashReportRescue
istCrashReportEnd
    atomic::FinalizeCrashReporter();
    return 0;
}

