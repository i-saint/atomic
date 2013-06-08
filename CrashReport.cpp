#include "stdafx.h"
#include "externals.h"

namespace atm {

using namespace Poco;
using namespace Poco::Net;


void SendReport( const char *path_to_dmp )
{
    URI uri(atm_crash_report_url);
    HTTPRequest request(HTTPRequest::HTTP_POST, uri.getPathAndQuery(), HTTPMessage::HTTP_1_0);

    FilePartSource *dump = new FilePartSource(path_to_dmp);
    // todo:
    //FilePartSource *log = new FilePartSource(path_to_log);
    //FilePartSource *replay = new FilePartSource(path_to_replay);

    HTMLForm form(HTMLForm::ENCODING_MULTIPART);
    form.addPart("dump", dump);
    //form.addPart("log", log);
    //form.addPart("replay", replay);
    form.prepareSubmit(request);

    // HTTP 1.0 の場合 content-length を設定しないといけない。
    // そんなに容量でかくないはずなので一番単純な方法。stream を全部書き出して容量を出す。
    std::string form_content;
    {
        std::stringstream ss;
        form.write(ss);
        form_content = ss.str();
    }
    request.setContentLength(form_content.size());

    HTTPClientSession session(uri.getHost(), uri.getPort());
    std::ostream &oustr = session.sendRequest(request);
    oustr.write(&form_content[0], form_content.size());

    HTTPResponse response;
    session.receiveResponse(response);
}


void InitializeCrashReporter()
{
    istSetDumpFileHanlder(&SendReport);
}

void FinalizeCrashReporter()
{

}


} // namespace atm
