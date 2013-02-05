#include "stdafx.h"
#include "../FunctionID.h"
#include "LevelEditorServer.h"

namespace atomic {

const char s_fileserver_base_dir[] = "editor";

struct MIME { const char *ext; const char *type; };
static const MIME s_mime_types[] = {
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

    void handleRequest(Poco::Net::HTTPServerRequest &request, Poco::Net::HTTPServerResponse &response)
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


void GetDecodedRequestBody(Poco::Net::HTTPServerRequest &request, std::string &out)
{
    if(!request.hasContentLength() || request.getContentLength()>1024*64) {
        return;
    }
    size_t size = (size_t)request.getContentLength();
    std::istream& stream = request.stream();
    std::string encoded_data;
    encoded_data.resize(size);
    stream.read(&encoded_data[0], size);
    Poco::URI::decode(encoded_data, out);
}

template<class Func>
void EachFormData(const std::string &form_fata, const Func &f)
{
    size_t pos = 0;
    size_t sep = 0;
    for(;;) {
        sep = form_fata.find_first_of('&', pos);
        if(sep==std::string::npos) {
            sep = form_fata.size();
            break;
        }
        f(&form_fata[pos], sep-pos);
        pos = sep+1;
    }
    f(&form_fata[pos], sep-pos);
}


class LevelEditorCommandHandler : public Poco::Net::HTTPRequestHandler
{
public:
    LevelEditorCommandHandler()
    {
    }

    void handleRequest(Poco::Net::HTTPServerRequest &request, Poco::Net::HTTPServerResponse &response)
    {
        std::string data;
        GetDecodedRequestBody(request, data);
        if(data.find("command=create")!=std::string::npos) {
            handleCreateRequest(data);
        }
        else if(data.find("command=delete")!=std::string::npos) {
            handleDeleteRequest(data);
        }
        else if(data.find("command=call")!=std::string::npos) {
            handleCallRequest(data);
        }

        response.setContentType("text/plain");
        std::ostream &ostr = response.send();
        ostr.write("ok", 3);

        // todo:
    }

    bool handleCreateRequest(std::string &data)
    {
        vec2 pos;
        EachFormData(data, [&](const char *str, size_t size){
            vec2 t;
            if(sscanf(str, "pos=vec2(%f,%f)", &t.x, &t.y)==2) { pos = t+LevelEditorServer::getInstance()->randomVec2()*0.01f; }
        });

        {
            LevelEditorCommand_Create cmd;
            LevelEditorServer::getInstance()->pushCommand(reinterpret_cast<LevelEditorCommand&>(cmd));
        }
        {
            LevelEditorCommand_Call cmd;
            cmd.entity_id = 0;
            cmd.function_id = FID_setPosition;
            cmd.arg = vec3(pos, 0.0f);
            LevelEditorServer::getInstance()->pushCommand(reinterpret_cast<LevelEditorCommand&>(cmd));
        }
        return true;
    }

    bool handleDeleteRequest(std::string &data)
    {
        return true;
    }

    bool handleCallRequest(std::string &data)
    {
        return true;
    }

private:
};

class LevelEditorQueryHandler : public Poco::Net::HTTPRequestHandler
{
public:
    LevelEditorQueryHandler()
    {
    }

    void handleRequest(Poco::Net::HTTPServerRequest &request, Poco::Net::HTTPServerResponse &response)
    {
        LevelEditorQuery q;
        if(request.getURI() == "/state/entities") {
            q.type = LEQ_Entities;
            LevelEditorServer::getInstance()->pushQuery(q);
            while(!q.completed) {
                ist::Thread::milliSleep(10);
            }
        }

        response.setChunkedTransferEncoding(true);
        response.setContentType("application/json");
        std::ostream &ostr = response.send();
        ostr << q.response;
    }

};

class LevelEditorRequestHandlerFactory : public Poco::Net::HTTPRequestHandlerFactory
{
public:
    virtual Poco::Net::HTTPRequestHandler* createRequestHandler(const Poco::Net::HTTPServerRequest &request)
    {
        if(request.getURI() == "/") {
            return new FileRequestHandler(std::string(s_fileserver_base_dir)+"/index.html");
        }
        else if(request.getURI() == "/command") {
            return new LevelEditorCommandHandler();
        }
        else if(request.getURI().find("/state")==0) {
            return new LevelEditorQueryHandler();
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


LevelEditorServerConfig::LevelEditorServerConfig()
    : port(10050)
    , max_queue(100)
    , max_threads(4)
{
}

LevelEditorServer::LevelEditorServer()
    : m_server(NULL)
    , m_accept_request(true)
{
    m_rand.initialize(0);
    m_commands.reserve(128);
}

LevelEditorServer::~LevelEditorServer()
{
    stop();
}

void LevelEditorServer::start()
{
    if(!m_server) {
        Poco::Net::HTTPServerParams* params = new Poco::Net::HTTPServerParams;
        params->setMaxQueued(m_conf.max_queue);
        params->setMaxThreads(m_conf.max_threads);
        params->setThreadIdleTime(Poco::Timespan(3, 0));

        try {
            Poco::Net::ServerSocket svs(m_conf.port);
            m_server = new Poco::Net::HTTPServer(new LevelEditorRequestHandlerFactory(), svs, params);
            m_server->start();
        }
        catch(Poco::IOException &e) {
            istAssert(e.what());
        }
    }
}

void LevelEditorServer::stop()
{
    if(m_server) {
        m_accept_request = false;
        m_server->stopAll(false);
        clearQuery();
        while(m_server->currentConnections()>0 || m_server->currentThreads()>0) {
            ist::Thread::milliSleep(5);
        }
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
    {
        ist::Mutex::ScopedLock lock(m_mutex_commands);
        m_commands_tmp = m_commands;
        m_commands.clear();
    }
    for(size_t i=0; i<m_commands_tmp.size(); ++i) {
        proc(m_commands_tmp[i]);
    }
    m_commands_tmp.clear();
}

void LevelEditorServer::handleQueries( const QueryProcessor &proc )
{
    {
        ist::Mutex::ScopedLock lock(m_mutex_queries);
        for(size_t i=0; i<m_queries.size(); ++i) {
            proc(*m_queries[i]);
            m_queries[i]->completed = true;
        }
        m_queries.clear();
    }
}

void LevelEditorServer::pushCommand( const LevelEditorCommand &cmd )
{
    if(!m_accept_request) { return; }
    ist::Mutex::ScopedLock lock(m_mutex_commands);
    m_commands.push_back(cmd);
}

void LevelEditorServer::pushQuery( LevelEditorQuery &q )
{
    if(!m_accept_request) { return; }
    ist::Mutex::ScopedLock lock(m_mutex_queries);
    m_queries.push_back(&q);
}

void LevelEditorServer::clearQuery()
{
    ist::Mutex::ScopedLock lock(m_mutex_queries);
    for(size_t i=0; i<m_queries.size(); ++i) {
        m_queries[i]->completed = true;
    }
    m_queries.clear();
}

vec2 LevelEditorServer::randomVec2()
{
    return (vec2(m_rand.genFloat32(), m_rand.genFloat32())-vec2(0.5f, 0.5f)) * 2.0f;
}


} // namespace atomic
