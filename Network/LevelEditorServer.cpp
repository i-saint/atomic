#include "stdafx.h"
#include "../FunctionID.h"
#include "LevelEditorServer.h"
#include "Game/AtomicApplication.h"
#include "Game/EntityClass.h"

namespace atm {

#ifdef atm_enable_LevelEditorServer

using Poco::Net::HTTPServerRequest;
using Poco::Net::HTTPServerResponse;

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


void GetDecodedRequestBody(HTTPServerRequest &request, std::string &out)
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


class NucleiCommandHandler : public Poco::Net::HTTPRequestHandler
{
public:
    enum ReturnCode {
        RC_Ok = 0,
        RC_InvalidCommand = -1,
        RC_InvalicParam = -2,
    };
    typedef std::function<void (NucleiCommandHandler*, HTTPServerRequest&, HTTPServerResponse&)> Handler;
    typedef stl::map<stl::string, Handler> HandlerTable;
    static HandlerTable&    getHandlerTable();
    static Handler*         findHandler(const stl::string &path);
    static FunctionID       getValidFID(const stl::string &name);
    static bool             isDeployable(EntityClassID ecid);

public:
    NucleiCommandHandler();
    void handleRequest(HTTPServerRequest &request, HTTPServerResponse &response);
    void respondCode(HTTPServerResponse &response, int32 code);

    void handleCreateRequest(HTTPServerRequest &request, HTTPServerResponse &response);
    void handleDeleteRequest(HTTPServerRequest &request, HTTPServerResponse &response);
    void handleCallRequest(HTTPServerRequest &request, HTTPServerResponse &response);
    void handleEntities(HTTPServerRequest &request, HTTPServerResponse &response);
    void handleEntity(HTTPServerRequest &request, HTTPServerResponse &response);
    void handleConst(HTTPServerRequest &request, HTTPServerResponse &response);
};

NucleiCommandHandler::HandlerTable& NucleiCommandHandler::getHandlerTable()
{
    static HandlerTable s_table;
    if(s_table.empty()) {
        s_table["/nuclei/call"]     = [](NucleiCommandHandler *o, HTTPServerRequest &req, HTTPServerResponse &res){ o->handleCallRequest(req,res);  };
        s_table["/nuclei/create"]   = [](NucleiCommandHandler *o, HTTPServerRequest &req, HTTPServerResponse &res){ o->handleCreateRequest(req,res);};
        s_table["/nuclei/delete"]   = [](NucleiCommandHandler *o, HTTPServerRequest &req, HTTPServerResponse &res){ o->handleDeleteRequest(req,res);};
        s_table["/nuclei/entity"]   = [](NucleiCommandHandler *o, HTTPServerRequest &req, HTTPServerResponse &res){ o->handleEntity(req,res);       };
        s_table["/nuclei/entities"] = [](NucleiCommandHandler *o, HTTPServerRequest &req, HTTPServerResponse &res){ o->handleEntities(req,res);     };
        s_table["/nuclei/const"]    = [](NucleiCommandHandler *o, HTTPServerRequest &req, HTTPServerResponse &res){ o->handleConst(req,res);        };
    }
    return s_table;
}

NucleiCommandHandler::Handler* NucleiCommandHandler::findHandler(const stl::string &path)
{
    auto &table = getHandlerTable();
    auto i = table.find(path);
    return i==table.end() ? nullptr : &i->second;
}

FunctionID NucleiCommandHandler::getValidFID(const stl::string &name)
{
    if(name.size()<64) {
        if(atmIsEditMode()) {
            char fid[128];
            istSPrintf(fid, "FID_%s", name.c_str());
            return GetFunctionIDNum(fid);
        }
        else {
            if(name=="instruct") { return FID_instruct; }
        }
    }
    return FID_unknown;
}

bool NucleiCommandHandler::isDeployable(EntityClassID ecid)
{
    if(EntityClassInfo *eci=GetEntityClassInfo(ecid)) {
        if(atmIsEditMode()) { return true; }
        else if(eci->deploy==DF_RTS) { return true; }
    }
    return false;
}


NucleiCommandHandler::NucleiCommandHandler()
{
}

void NucleiCommandHandler::handleRequest(HTTPServerRequest &request, HTTPServerResponse &response)
{
    if(Handler *hanlder = findHandler(request.getURI())) {
        (*hanlder)(this, request, response);
    }
}

void NucleiCommandHandler::respondCode(HTTPServerResponse &response, int32 code)
{
    char str[16];
    size_t len;
    istSPrintf(str, "%d", code);
    len = strlen(str);

    response.setContentType("text/plain");
    response.setContentLength(len);
    std::ostream &ostr = response.send();
    ostr.write(str, len);
}

void NucleiCommandHandler::handleCreateRequest(HTTPServerRequest &request, HTTPServerResponse &response)
{
    std::string data;
    GetDecodedRequestBody(request, data);
    int32 code = 0;

    std::smatch m1;
    if(std::regex_search(data, m1, std::regex("classid=(\\d+),pos=(.+)"))) {
        EntityClassID ecid = (EntityClassID)_atoi64(m1[1].str().c_str());
        if(!isDeployable(ecid)) {
            code=RC_InvalidCommand; goto RESPOND;
        }

        variant vpos;
        if(ParseArg(vpos, m1[2].str())) {
            vec2 pos = (vec2&)vpos + WebServer::getInstance()->randomVec2()*0.01f;
            {
                LevelEditorCommand_Create cmd;
                cmd.classid = ecid;
                WebServer::getInstance()->pushCommand((LevelEditorCommand&)cmd);
            }
            {
                LevelEditorCommand_Call cmd;
                cmd.entity = 0;
                cmd.function = FID_move;
                cmd.arg = vec3(pos, 0.0f);
                WebServer::getInstance()->pushCommand((LevelEditorCommand&)cmd);
            }
        }
    }
RESPOND:
    respondCode(response, code);
}

void NucleiCommandHandler::handleDeleteRequest(HTTPServerRequest &request, HTTPServerResponse &response)
{
    std::string data;
    GetDecodedRequestBody(request, data);
    int32 code = 0;

    respondCode(response, code);
}

void NucleiCommandHandler::handleCallRequest(HTTPServerRequest &request, HTTPServerResponse &response)
{
    std::string data;
    GetDecodedRequestBody(request, data);
    int32 code = 0;

    std::smatch m1;
    if(std::regex_search(data, m1, std::regex("entity=(\\d+),func=(\\w+),arg=(.+)"))) {
        variant arg;
        EntityHandle entity = (EntityHandle)_atoi64(m1[1].str().c_str());
        FunctionID fid = getValidFID(m1[2].str());
        if(fid!=0 && ParseArg(arg, m1[3].str())) {
            LevelEditorCommand_Call cmd;
            cmd.entity = entity;
            cmd.function = fid;
            cmd.arg = arg;
            WebServer::getInstance()->pushCommand((LevelEditorCommand&)cmd);
        }
        else {
            code = RC_InvalidCommand;
        }
    }
    respondCode(response, code);
}

void NucleiCommandHandler::handleEntities(HTTPServerRequest &request, HTTPServerResponse &response)
{
    LevelEditorQuery q;
    q.type = LEQ_Entities;
    WebServer::getInstance()->pushQuery(q);
    while(!q.completed && !WebServer::getInstance()->endFlag()) {
        ist::MiliSleep(5);
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


void NucleiCommandHandler::handleEntity(HTTPServerRequest &request, HTTPServerResponse &response)
{
    std::string data;
    GetDecodedRequestBody(request, data);

    uint32 h = 0;
    sscanf(data.c_str(), "entity=%u", &h);

    LevelEditorQuery q;
    q.type = LEQ_Entity;
    q.optional =  h;
    WebServer::getInstance()->pushQuery(q);
    while(!q.completed && !WebServer::getInstance()->endFlag()) {
        ist::MiliSleep(5);
    }

    response.setContentType("application/json");
    response.setContentLength(q.response.size());
    std::ostream &ostr = response.send();
    ostr << q.response;
}

void NucleiCommandHandler::handleConst(HTTPServerRequest &request, HTTPServerResponse &response)
{
    stl::string ret;
    ret += "{";
    {
        bool first = true;
        ret += "\"entityTypes\":{";
        EntityClassIDEachPair([&](const ist::EnumStr &es){
            EntityClassID ecid = (EntityClassID)es.num;
            const EntityClassInfo *eci = GetEntityClassInfo(ecid);
            bool deployable = false;
            float32 cost = 0.0f;
            if(eci) {
                if(eci->deploy==DF_RTS || (eci->deploy==DF_Editor && atmIsEditMode())) { deployable=true; }
                cost = eci->cost;
            }
            if(!first){ret+=",";} first=false;
            ret += ist::Format(
                "\"%s\":{\"id\":%d,\"deployable\":%d,\"cost\":%f}",
                es.str+3, es.num, deployable, cost );

        });
        ret += "},";
    }
    {
        bool first = true;
        ret += "\"functions\":{";
        FunctionIDEachPair([&](const ist::EnumStr &es){
            if(!first){ret+=",";} first=false;
            ret += ist::Format("\"%s\":%d", es.str+4, es.num);
        });
        ret += "}";
    }
    ret += "}";

    response.setContentType("application/json");
    response.setContentLength(ret.size());
    std::ostream &ostr = response.send();
    ostr << ret;
}


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
