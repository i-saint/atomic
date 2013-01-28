#include "stdafx.h"
#include "LevelEditorServer.h"

namespace atomic {

using namespace Poco;
using namespace Poco::Net;

class LevelEditorRequestHandler: public HTTPRequestHandler
{
public:
    LevelEditorRequestHandler()
    {
    }

    void handleRequest(HTTPServerRequest& request, HTTPServerResponse& response)
    {
        Timestamp now;

        response.setChunkedTransferEncoding(true);
        response.setContentType("text/html");

        std::ostream& ostr = response.send();
        ostr << "<html><head><title>HTTPTimeServer powered by POCO C++ Libraries</title>";
        ostr << "<meta http-equiv=\"refresh\" content=\"1\"></head>";
        ostr << "<body><p style=\"text-align: center; font-size: 48px;\">";
        ostr << now.epochMicroseconds();
        ostr << "</p></body></html>";
    }

private:
};

class LevelEditorRequestHandlerFactory: public HTTPRequestHandlerFactory
{
public:
    LevelEditorRequestHandlerFactory()
    {
    }

    virtual HTTPRequestHandler* createRequestHandler(const HTTPServerRequest& request)
    {
        if(request.getURI() == "/") {
            return new LevelEditorRequestHandler();
        }
        else {
            return 0;
        }
    }

private:
};


LevelEditorServerConfig::LevelEditorServerConfig()
    : port(10050)
    , max_queue(100)
    , max_threads(4)
{
}



LevelEditorServer::LevelEditorServer()
    : m_server(NULL)
{
    m_commands.reserve(128);
}

LevelEditorServer::~LevelEditorServer()
{
    stop();
}

void LevelEditorServer::start()
{
    if(!m_server) {
        HTTPServerParams* pParams = new HTTPServerParams;
        pParams->setMaxQueued(m_conf.max_queue);
        pParams->setMaxThreads(m_conf.max_threads);

        ServerSocket svs(m_conf.port);

        m_server = new Poco::Net::HTTPServer(new LevelEditorRequestHandlerFactory(), svs, pParams);
        m_server->start();
    }
}

void LevelEditorServer::stop()
{
    if(m_server) {
        m_server->stop();
        delete m_server;
        m_server = NULL;
    }
}

void LevelEditorServer::restart()
{
    stop();
    start();
}

void LevelEditorServer::handleCommands( const CommandProcessor &proc )
{

}


LevelEditorServer * LevelEditorServer::s_inst;

void LevelEditorServer::initializeInstance()
{
    if(!s_inst) {
        s_inst = new LevelEditorServer();
        s_inst->start();
    }
}

void LevelEditorServer::finalizeInstance()
{
    if(s_inst) {
        delete s_inst;
        s_inst = NULL;
    }
}

LevelEditorServer* LevelEditorServer::getInstance()
{
    return s_inst;
}

} // namespace atomic
