#include "stdafx.h"
#include "../FunctionID.h"
#include "LevelEditorServer.h"
#include "Game/AtomicApplication.h"
#include "Game/EntityClass.h"

namespace atm {

#ifdef atm_enable_LevelEditorServer

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

inline bool ParseArg(variant &out, const std::string &str)
{
    ivec4 iv;
    uvec4 uv;
    vec4 fv;
    if(sscanf(str.c_str(), "int(%d)", &iv.x)==1) {
        out=iv; return true;
    }
    else if(sscanf(str.c_str(), "uint(%u)", &uv.x)==1) {
        out=uv; return true;
    }
    else if(sscanf(str.c_str(), "float(%f)", &fv.x)==1) {
        out=fv; return true;
    }
    else if(sscanf(str.c_str(), "vec2(%f,%f)", &fv.x, &fv.y)==2) {
        out=fv; return true;
    }
    else if(sscanf(str.c_str(), "vec3(%f,%f,%f)", &fv.x, &fv.y, &fv.z)==3) {
        out=fv; return true;
    }
    else if(sscanf(str.c_str(), "instruction(%f,%f,%f,%u)", &fv.x, &fv.y, &fv.z, &uv.x)==4) {
        out=ist::MakeValueList(vec3(fv),uv.x); return true;
    }
    return false;
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
        if(request.getURI()=="/call") {
            handleCallRequest(data);
        }
        else if(request.getURI()=="/create") {
            handleCreateRequest(data);
        }
        else if(request.getURI()=="/delete") {
            handleDeleteRequest(data);
        }

        response.setContentType("text/plain");
        response.setContentLength(2);
        std::ostream &ostr = response.send();
        ostr.write("ok", 2);
    }

    bool handleCreateRequest(std::string &data)
    {
        static stl::set<EntityClassID> s_table;
        if(s_table.empty()) {
#define RegisterEntityClass(f) s_table.insert(EC_##f)
            RegisterEntityClass(Enemy_Test);
            RegisterEntityClass(SmallFighter);
            RegisterEntityClass(MediumFighter);
            RegisterEntityClass(LargeFighter);
#undef RegisterEntityClass
        }

        std::smatch m1;
        if(std::regex_search(data, m1, std::regex("classid=(\\d+),pos=(.+)"))) {
            EntityClassID cid = (EntityClassID)_atoi64(m1[1].str().c_str());
            if(s_table.find(cid)==s_table.end()) { return false; }

            variant vpos;
            if(ParseArg(vpos, m1[2].str())) {
                vec2 pos = (vec2&)vpos + LevelEditorServer::getInstance()->randomVec2()*0.01f;
                {
                    LevelEditorCommand_Create cmd;
                    LevelEditorServer::getInstance()->pushCommand((LevelEditorCommand&)cmd);
                }
                {
                    LevelEditorCommand_Call cmd;
                    cmd.entity = 0;
                    cmd.function = FID_setPosition;
                    cmd.arg = vec3(pos, 0.0f);
                    LevelEditorServer::getInstance()->pushCommand((LevelEditorCommand&)cmd);
                }
                return true;
            }
        }

        return false;
    }

    bool handleDeleteRequest(std::string &data)
    {
        return true;
    }

    bool handleCallRequest(std::string &data)
    {
        static stl::map<stl::string, FunctionID> s_table;
        if(s_table.empty()) {
#define RegisterFunction(f) s_table[#f]=FID_##f
            RegisterFunction(move);
            RegisterFunction(orient);
            RegisterFunction(instruct);
            RegisterFunction(setPosition);
            RegisterFunction(setScale);
            RegisterFunction(setParent);
#undef RegisterFunction
        }

        std::smatch m1;
        if(std::regex_search(data, m1, std::regex("entity=(\\d+),func=(\\w+),arg=(.+)"))) {
            variant arg;
            EntityHandle entity = (EntityHandle)_atoi64(m1[1].str().c_str());
            auto i = s_table.find(m1[2].str());
            if(i!=s_table.end() && ParseArg(arg, m1[3].str())) {
                LevelEditorCommand_Call cmd;
                cmd.entity = entity;
                cmd.function = i->second;
                cmd.arg = arg;
                LevelEditorServer::getInstance()->pushCommand((LevelEditorCommand&)cmd);
                return true;
            }
            else {
                istPrint("handleCallRequest(): invalid command %s\n", data.c_str());
            }
        }
        return false;
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
            while(!q.completed && !LevelEditorServer::getInstance()->endFlag()) {
                ist::MiliSleep(10);
            }
        }

        response.setChunkedTransferEncoding(true);
#ifdef atm_enable_WebGL
        response.setContentType("application/octet-stream");
#else // atm_enable_WebGL
        response.setContentType("application/json");
#endif // atm_enable_WebGL
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
        else if(request.getURI()=="/create" || request.getURI()=="/delete" || request.getURI()=="/call") {
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
    : max_queue(100)
    , max_threads(4)
{
}

LevelEditorServer::LevelEditorServer()
    : m_server(NULL)
    , m_end_flag(false)
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
            Poco::Net::ServerSocket svs(atmGetConfig()->leveleditor_port);
            m_server = new Poco::Net::HTTPServer(new LevelEditorRequestHandlerFactory(), svs, params);
            m_server->start();
        }
        catch(Poco::IOException &e) {
            istAssert(false);
        }
    }
}

void LevelEditorServer::stop()
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

void LevelEditorServer::restart()
{
    stop();
    start();
}

void LevelEditorServer::handleCommands( const CommandHandler &h )
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

void LevelEditorServer::handleQueries( const QueryHandler &h )
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

void LevelEditorServer::pushCommand( const LevelEditorCommand &cmd )
{
    if(m_end_flag) { return; }
    ist::Mutex::ScopedLock lock(m_mutex_commands);
    m_commands.push_back(cmd);
}

void LevelEditorServer::pushQuery( LevelEditorQuery &q )
{
    if(m_end_flag) { return; }
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

#else // atm_enable_LevelEditorServer
#endif // atm_enable_LevelEditorServer


} // namespace atm
