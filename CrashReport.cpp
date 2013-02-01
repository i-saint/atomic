#include "stdafx.h"
#include "externals.h"

namespace atomic {

const char g_crash_report_uri[] = "http://primitive-games.jp/atomic/crash_report/post";

using namespace Poco;
using namespace Poco::Net;


class CrashReporter
{
public:
    void report(const char *path_to_dmp);
    void wait();

private:
    HTTPClientSession *m_session;
};

void CrashReporter::report( const char *path_to_dmp )
{
    FilePartSource *dump = new FilePartSource(path_to_dmp);
    // todo:
    //FilePartSource *log = new FilePartSource(path_to_log);
    //FilePartSource *replay = new FilePartSource(path_to_replay);

    URI uri(g_crash_report_uri);
    m_session = new HTTPClientSession(uri.getHost(), uri.getPort());

    HTTPRequest request(HTTPRequest::HTTP_POST, uri.getPathAndQuery(), HTTPMessage::HTTP_1_1);
    HTMLForm form("multipart/form-data");
    form.addPart("dump", dump);
    //form.addPart("log", log);
    //form.addPart("replay", replay);
    form.prepareSubmit(request);
    m_session->sendRequest(request);
}

void CrashReporter::wait()
{
    delete m_session;
    m_session = NULL;
}

static CrashReporter g_crash_reporter;

void InitializeCrashReporter()
{
    istSetDumpFileHanlder(std::bind(&CrashReporter::report, &g_crash_reporter, std::placeholders::_1));
}

void FinalizeCrashReporter()
{

}


} // namespace atomic
