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
#include <functional>


struct InputServerConfig
{
    uint16 port;
    uint16 max_queue;
    uint16 max_threads;

    InputServerConfig();
};

struct InputCommand
{

};

class InputCommandHandler;
class InputRequestHandlerFactory;

class InputServer
{
friend class InputCommandHandler;
friend class InputQueryHandler;
friend class InputRequestHandlerFactory;
public:
    static void initializeInstance();
    static void finalizeInstance();
    static InputServer* getInstance();

    InputServer();
    ~InputServer();
    void start();
    void stop();
    HTTPInputData& getState();

    void pushCommand(const InputCommand &cmd);
    bool endFlag() const { return m_end_flag; }

private:
    typedef std::vector<InputCommand> CommandCont;

    static InputServer *s_inst;
    Poco::Net::HTTPServer *m_server;
    InputServerConfig m_conf;
    bool m_end_flag;

    Poco::Mutex m_mutex_commands;
    CommandCont m_commands;
    CommandCont m_commands_tmp;
    HTTPInputData m_state;
};




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



class InputCommandHandler : public Poco::Net::HTTPRequestHandler
{
public:
    InputCommandHandler()
    {
    }

    void handleRequest(Poco::Net::HTTPServerRequest &request, Poco::Net::HTTPServerResponse &response)
    {
        std::istream& stream = request.stream();
        if(request.getURI()=="/keyboard") {
            // todo:
        }
        else if(request.getURI()=="/mouse") {

        }
        else if(request.getURI()=="/pad") {

        }

        response.setContentType("text/plain");
        std::ostream &ostr = response.send();
        ostr.write("ok", 3);

        // todo:
    }
};

class InputRequestHandlerFactory : public Poco::Net::HTTPRequestHandlerFactory
{
public:
    virtual Poco::Net::HTTPRequestHandler* createRequestHandler(const Poco::Net::HTTPServerRequest &request)
    {
        if(request.getURI() == "/") {
            return new FileRequestHandler(std::string(s_fileserver_base_dir)+"/index.html");
        }
        else if(request.getURI()=="/keyboard" || request.getURI()=="/mouse" || request.getURI()=="/pad") {
            return new InputCommandHandler();
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
    : port(10000)
    , max_queue(100)
    , max_threads(4)
{
}

InputServer::InputServer()
    : m_server(NULL)
    , m_end_flag(false)
{
    memset(&m_state, 0, sizeof(m_state));
    m_commands.reserve(128);
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

void InputServer::pushCommand( const InputCommand &cmd )
{
    if(m_end_flag) { return; }
    Poco::Mutex::ScopedLock lock(m_mutex_commands);
    m_commands.push_back(cmd);
}

HTTPInputData& InputServer::getState()
{
    {
        Poco::Mutex::ScopedLock lock(m_mutex_commands);
        m_commands_tmp = m_commands;
        m_commands.clear();
    }
    for(size_t i=0; i<m_commands_tmp.size(); ++i) {
        // todo:
    }
    return m_state;
}



bool StartHTTPInputServer()
{
    InputServer::initializeInstance();
    return true;
}

bool StopHTTPInputServer()
{
    InputServer::finalizeInstance();
    return true;
}

const HTTPInputData* GetHTTPInputData()
{
    return &InputServer::getInstance()->getState();
}
