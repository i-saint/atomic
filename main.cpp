#include "stdafx.h"
#include "types.h"
#include "Game/AtomicApplication.h"
#include "FunctionID.h"

namespace atomic {
    void InitializeCrashReporter();
    void FinalizeCrashReporter();
} // namespace atomic

void ExecApp(int argc, char* argv[])
{
    atomic::AtomicApplication app;
    if(app.initialize(argc, argv)) {
        static_cast<atomic::AtomicApplication*>(NULL)->initialize(argc, argv);
        app.mainLoop();
    }
    app.finalize();
}

int istmain(int argc, char* argv[])
{
    //atomic::FunctionIDEachPair([](const ist::EnumStr &es){
    //    istPrint("%d:%s\n", es.num, es.str);
    //});

    atomic::InitializeCrashReporter();
istCrashReportBegin
    ExecApp(argc, argv);
istCrashReportRescue
istCrashReportEnd
    atomic::FinalizeCrashReporter();
    return 0;
}

