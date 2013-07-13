#include "stdafx.h"
#include "../FunctionID.h"
#include "WebServer.h"
#include "LevelEditorCommandHandler.h"
#include "Game/AtomicApplication.h"

namespace atm {

#ifdef atm_enable_LevelEditorServer

const char s_fileserver_base_dir[] = "editor";

struct MIMEType { const char *ext; const char *type; };
static const MIMEType s_mime_types[] = {
    {".txt",  "text/plain"},
    {".html", "text/html"},
    {".css",  "text/css"},
    {".js",   "text/javascript"},
    {".png",  "image/png"},
    {".jpg",  "image/jpeg"},
};


class FileRequestHandler: public Poco::Net::HTTPRequestHandler
{
public:
    FileRequestHandler(const std::string &path)
        : m_path(path)
    {
    }

    void handleRequest(HTTPServerRequest &request, HTTPServerResponse &response)
    {
        const char *ext = s_mime_types[0].ext;
        const char *mime = s_mime_types[0].type;
        size_t epos = m_path.find_last_of(".");
        if(epos!=std::string::npos) {
            ext = &m_path[epos];
            for(size_t i=0; i<_countof(s_mime_types); ++i) {
                if(strcmp(ext, s_mime_types[i].ext)==0) {
                    mime = s_mime_types[i].type;
                }
            }
        }
        response.sendFile(m_path, mime);
    }

private:
    std::string m_path;
};


class WebRequestHandlerFactory : public Poco::Net::HTTPRequestHandlerFactory
{
public:
    virtual Poco::Net::HTTPRequestHandler* createRequestHandler(const HTTPServerRequest &request)
    {
        if(request.getURI()=="/") {
            return new FileRequestHandler(std::string(s_fileserver_base_dir)+"/index.html");
        }
        else if( NucleiCommandHandler::findHandler(request.getURI()) ) {
            return new NucleiCommandHandler();
        }
        else {
            std::string path = std::string(s_fileserver_base_dir)+request.getURI();
            Poco::File file(path);
            if(file.exists()) {
                return new FileRequestHandler(path);
            }
            else {
                return 0;
            }
        }
    }
};



WebServer * WebServer::s_inst;

void WebServer::initializeInstance()
{
    if(!s_inst) {
        s_inst = new WebServer();
        s_inst->start();
    }
}

void WebServer::finalizeInstance()
{
    if(s_inst) {
        delete s_inst;
        s_inst = NULL;
    }
}

WebServer* WebServer::getInstance()
{
    return s_inst;
}


WebServerConfig::WebServerConfig()
    : max_queue(100)
    , max_threads(4)
{
}

WebServer::WebServer()
    : m_server(NULL)
    , m_end_flag(false)
{
    m_rand.initialize(0);
    m_commands.reserve(128);
}

WebServer::~WebServer()
{
    stop();
}

void WebServer::start()
{
    if(!m_server) {
        Poco::Net::HTTPServerParams* params = new Poco::Net::HTTPServerParams;
        params->setMaxQueued(m_conf.max_queue);
        params->setMaxThreads(m_conf.max_threads);
        params->setThreadIdleTime(Poco::Timespan(3, 0));

        try {
            Poco::Net::ServerSocket svs(atmGetConfig()->leveleditor_port);
            m_server = new Poco::Net::HTTPServer(new WebRequestHandlerFactory(), svs, params);
            m_server->start();
        }
        catch(Poco::IOException &e) {
            istPrint(e.what());
            istAssert(false);
        }
    }
}

void WebServer::stop()
{
    if(m_server) {
        m_end_flag = true;
        m_server->stopAll(false);
        clearQuery();
        while(m_server->currentConnections()>0 || m_server->currentThreads()>0) {
            ist::MiliSleep(5);
        }
        delete m_server;
        m_server = NULL;
    }
}

void WebServer::restart()
{
    stop();
    start();
}

void WebServer::handleCommands( const CommandHandler &h )
{
    {
        ist::Mutex::ScopedLock lock(m_mutex_commands);
        m_commands_tmp = m_commands;
        m_commands.clear();
    }
    for(size_t i=0; i<m_commands_tmp.size(); ++i) {
        h(m_commands_tmp[i]);
    }
    m_commands_tmp.clear();
}

void WebServer::handleQueries( const QueryHandler &h )
{
    {
        ist::Mutex::ScopedLock lock(m_mutex_queries);
        for(size_t i=0; i<m_queries.size(); ++i) {
            h(*m_queries[i]);
            m_queries[i]->completed = true;
        }
        m_queries.clear();
    }
}

void WebServer::pushCommand( const LevelEditorCommand &cmd )
{
    if(m_end_flag) { return; }
    ist::Mutex::ScopedLock lock(m_mutex_commands);
    m_commands.push_back(cmd);
}

void WebServer::pushQuery( LevelEditorQuery &q )
{
    if(m_end_flag) { return; }
    ist::Mutex::ScopedLock lock(m_mutex_queries);
    m_queries.push_back(&q);
}

void WebServer::clearQuery()
{
    ist::Mutex::ScopedLock lock(m_mutex_queries);
    for(size_t i=0; i<m_queries.size(); ++i) {
        m_queries[i]->completed = true;
    }
    m_queries.clear();
}

vec2 WebServer::randomVec2()
{
    return (vec2(m_rand.genFloat32(), m_rand.genFloat32())-vec2(0.5f, 0.5f)) * 2.0f;
}

#else // atm_enable_LevelEditorServer
#endif // atm_enable_LevelEditorServer


} // namespace atm
