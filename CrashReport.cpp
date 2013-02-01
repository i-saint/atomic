#include "stdafx.h"
#include "externals.h"

namespace atomic {

const char g_crash_report_uri[] = "http://primitive-games.jp/atomic/crash_report/post";

using namespace Poco;
using namespace Poco::Net;


class HTMLFormFileSource : public Poco::Net::PartSource
{
typedef Poco::Net::PartSource super;
public:
    HTMLFormFileSource(const std::string media_type) : super(media_type) {}
    ~HTMLFormFileSource()
    {
        if(!m_filename.empty()) {
            m_stream.close();
            std::remove(m_filename.c_str());
        }
    }

    virtual std::istream& stream() { return m_stream; }
    virtual const std::string& filename() { return m_filename; }

    bool openFile(const char *path)
    {
        m_stream.open(path, std::ios::in|std::ios::binary);
        m_filename = path;
        return true;
    }

private:
    std::string m_filename;
    std::ifstream m_stream;
};

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
    HTMLFormFileSource *src = new HTMLFormFileSource("application/octet-stream");
    if(!src->openFile(path_to_dmp)) {
        delete src;
        return;
    }

    URI uri(g_crash_report_uri);
    m_session = new HTTPClientSession(uri.getHost(), uri.getPort());

    HTTPRequest request(HTTPRequest::HTTP_GET, uri.getPathAndQuery(), HTTPMessage::HTTP_1_1);
    HTMLForm form("multipart/form-data");
    form.addPart("report", src);
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
