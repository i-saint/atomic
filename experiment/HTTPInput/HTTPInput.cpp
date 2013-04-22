#define POCO_STATIC
#include "Poco/Path.h"
#include "Poco/File.h"
#include "Poco/FileStream.h"
#include "Poco/Timestamp.h"
#include "Poco/URI.h"
#include "Poco/Net/TCPServer.h"
#include "Poco/Net/TCPServerParams.h"
#include "Poco/Net/HTTPServer.h"
#include "Poco/Net/HTTPRequestHandler.h"
#include "Poco/Net/HTTPRequestHandlerFactory.h"
#include "Poco/Net/HTTPServerParams.h"
#include "Poco/Net/HTTPServerRequest.h"
#include "Poco/Net/HTTPServerResponse.h"
#include "Poco/Net/HTTPServerParams.h"
#include "Poco/Net/HTTPClientSession.h"
#include "Poco/Net/HTMLForm.h"
#include "Poco/Net/PartHandler.h"
#include "Poco/Net/PartSource.h"
#include "Poco/Net/FilePartSource.h"
#include "Poco/Net/MessageHeader.h"
#include "Poco/Net/ServerSocket.h"
#include "Poco/Net/StreamSocket.h"
#include "Poco/Net/SocketStream.h"
#include "Poco/Net/SocketAddress.h"
#include "HTTPInput.h"
#include "HTTPInput_Internal.h"
#include <functional>


struct InputServerConfig
{
    uint16 port;
    uint16 max_queue;
    uint16 max_threads;

    InputServerConfig();
};



class InputServer
{
public:
    static void initializeInstance();
    static void finalizeInstance();
    static InputServer* getInstance();

    InputServer();
    ~InputServer();
    void start();
    void stop();
    HTTPInputData& getState();

    bool endFlag() const { return m_end_flag; }

private:

    static InputServer *s_inst;
    Poco::Net::HTTPServer *m_server;
    InputServerConfig m_conf;
    bool m_end_flag;

    HTTPInputData m_state;
};




const char s_root_dir[] = "html";

struct MIME { const char *ext; const char *type; };
static const MIME s_mime_types[] = {
    {".txt",  "text/plain"},
    {".html", "text/html"},
    {".css",  "text/css"},
    {".js",   "text/javascript"},
    {".png",  "image/png"},
    {".jpg",  "image/jpeg"},
};

static const char* GetModulePath()
{
    static char s_path[MAX_PATH] = {0};
    if(s_path[0]==0) {
        HMODULE mod = 0;
        ::GetModuleHandleEx(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, (LPCTSTR)&GetModulePath, &mod);
        DWORD size = ::GetModuleFileNameA(mod, s_path, MAX_PATH);
        while(size>0) {
            if(s_path[size]=='\\') {
                s_path[size+1] = '\0';
                break;
            }
            --size;
        }
    }
    return s_path;
}


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


template<class F>
void EachInputValue(Poco::Net::HTTPServerRequest &request, const F &f)
{
    if(!request.hasContentLength() || request.getContentLength()>1024*64) {
        return;
    }
    size_t size = (size_t)request.getContentLength();
    std::istream& stream = request.stream();
    std::string encoded_content;
    std::string content;
    encoded_content.resize(size);
    stream.read(&encoded_content[0], size);
    Poco::URI::decode(encoded_content, content);
    for(size_t i=5; i<content.size(); ++i) {
        if(content[i-1]==',' || content[i-1]=='=') {
            f(&content[i]);
        }
    }
}

class InputCommandHandler : public Poco::Net::HTTPRequestHandler
{
public:
    InputCommandHandler()
    {
    }

    void handleRequest(Poco::Net::HTTPServerRequest &request, Poco::Net::HTTPServerResponse &response)
    {
        HTTPInputData *input = HTTPInput_GetData();
        if(request.getURI()=="/input") {
            EachInputValue(request, [&](const char *value){
                int pad, button, v;
                if     (sscanf(value, "pad%d_x1:%x", &pad, &v)==2) { input->pad[pad].x1=v; }
                else if(sscanf(value, "pad%d_y1:%x", &pad, &v)==2) { input->pad[pad].y1=v; }
                else if(sscanf(value, "pad%d_x2:%x", &pad, &v)==2) { input->pad[pad].x2=v; }
                else if(sscanf(value, "pad%d_y2:%x", &pad, &v)==2) { input->pad[pad].y2=v; }
                else if(sscanf(value, "pad%d_RT:%x", &pad, &v)==2) { input->pad[pad].trigger1=v; }
                else if(sscanf(value, "pad%d_LT:%x", &pad, &v)==2) { input->pad[pad].trigger2=v; }
                else if(sscanf(value, "pad%d_button%d:%x", &pad, &button, &v)==3) {
                    if(button>=0 && button<HTTPInputData::Pad::MaxButtons) {
                        input->pad[pad].buttons[button] = v;
                    }
                }
            });
        }

        response.setContentType("text/plain");
        response.setContentLength(2);
        std::ostream &ostr = response.send();
        ostr.write("ok", 3);
    }
};

class InputRequestHandlerFactory : public Poco::Net::HTTPRequestHandlerFactory
{
public:
    virtual Poco::Net::HTTPRequestHandler* createRequestHandler(const Poco::Net::HTTPServerRequest &request)
    {
        if(request.getURI() == "/") {
            return new FileRequestHandler(std::string(GetModulePath())+std::string(s_root_dir)+"/index.html");
        }
        else if(request.getURI()=="/input") {
            return new InputCommandHandler();
        }
        else {
            std::string path = std::string(GetModulePath())+std::string(s_root_dir)+request.getURI();
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




InputServer * InputServer::s_inst;

void InputServer::initializeInstance()
{
    if(!s_inst) {
        s_inst = new InputServer();
        s_inst->start();
    }
}

void InputServer::finalizeInstance()
{
    if(s_inst) {
        delete s_inst;
        s_inst = NULL;
    }
}

InputServer* InputServer::getInstance()
{
    return s_inst;
}


InputServerConfig::InputServerConfig()
    : port(10001)
    , max_queue(100)
    , max_threads(4)
{
}

InputServer::InputServer()
    : m_server(NULL)
    , m_end_flag(false)
{
    memset(&m_state, 0, sizeof(m_state));
    for(int i=0; i<HTTPInputData::MaxPads; ++i) {
        m_state.pad[i].x1 = m_state.pad[i].y1 = m_state.pad[i].x2 = m_state.pad[i].y2 = INT16_MAX;
    }
}

InputServer::~InputServer()
{
    stop();
}

void InputServer::start()
{
    if(!m_server) {
        Poco::Net::HTTPServerParams* params = new Poco::Net::HTTPServerParams;
        params->setMaxQueued(m_conf.max_queue);
        params->setMaxThreads(m_conf.max_threads);
        params->setThreadIdleTime(Poco::Timespan(3, 0));

        try {
            Poco::Net::ServerSocket svs(m_conf.port);
            m_server = new Poco::Net::HTTPServer(new InputRequestHandlerFactory(), svs, params);
            m_server->start();
        }
        catch(Poco::IOException &) {
        }
    }
}

void InputServer::stop()
{
    if(m_server) {
        m_end_flag = true;
        m_server->stopAll(false);
        while(m_server->currentConnections()>0 || m_server->currentThreads()>0) {
            ::Sleep(1);
        }
        delete m_server;
        m_server = NULL;
    }
}

HTTPInputData& InputServer::getState()
{
    return m_state;
}

static HTTPInputConfig g_conf = {true, true, true, true};

extern "C" {

__declspec(dllexport) HTTPInputConfig*  HTTPInput_GetConfig()
{
    return &g_conf;
}

__declspec(dllexport) bool HTTPInput_StartServer()
{
    InputServer::initializeInstance();
    return true;
}

__declspec(dllexport) bool HTTPInput_StopServer()
{
    InputServer::finalizeInstance();
    return true;
}

__declspec(dllexport) HTTPInputData* HTTPInput_GetData()
{
    if(InputServer *server = InputServer::getInstance()) {
        return &server->getState();
    }
    return NULL;
}

} // extern "C"
