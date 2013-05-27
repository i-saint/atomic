#define POCO_STATIC
#include "Poco/URI.h"
#include "Poco/Net/TCPServer.h"
#include "Poco/Net/ServerSocket.h"
#include "Poco/Net/StreamSocket.h"
#include "Poco/Net/SocketStream.h"
#include "Poco/Net/SocketAddress.h"


class TestSession : public Poco::Net::TCPServerConnection
{
typedef Poco::Net::TCPServerConnection super;
public:
    TestSession(const Poco::Net::StreamSocket &ss)
        : super(ss)
    {}

    virtual void run()
    {
        Poco::Net::StreamSocket &ss = socket();
        for(;;) {
            char message[8];
            ss.receiveBytes(message, 5);
            ss.receiveBytes(message, 5);
            ss.sendBytes("pong", 5);
            printf("%s\n", message);
            ::Sleep(200);
        }
    }
};


class TestSessionFactory : public Poco::Net::TCPServerConnectionFactory
{
public:
    virtual Poco::Net::TCPServerConnection* createConnection(const Poco::Net::StreamSocket &ss)
    {
        return new TestSession(ss);
    }
};


int main()
{
    Poco::Net::TCPServerParams* params = new Poco::Net::TCPServerParams();
    params->setMaxQueued(10);
    params->setMaxThreads(8);
    params->setThreadIdleTime(Poco::Timespan(3, 0));

    Poco::Net::ServerSocket svs(10045);
    Poco::Net::TCPServer *server = new Poco::Net::TCPServer(new TestSessionFactory(), svs, params);
    server->start();
    for(;;) { ::Sleep(100); }
}