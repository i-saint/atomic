#define POCO_STATIC
#include "Poco/URI.h"
#include "Poco/Net/TCPServer.h"
#include "Poco/Net/ServerSocket.h"
#include "Poco/Net/StreamSocket.h"
#include "Poco/Net/SocketStream.h"
#include "Poco/Net/SocketAddress.h"

int main()
{
    Poco::Net::SocketAddress address = Poco::Net::SocketAddress("localhost", 10045);

    Poco::Net::StreamSocket *sock = new Poco::Net::StreamSocket(address);
    sock->setNoDelay(true);
    sock->setBlocking(true);
    sock->setReceiveTimeout(Poco::Timespan(3, 0));
    sock->setSendTimeout(Poco::Timespan(3, 0));

    for(;;) {
        char message[8];
        sock->sendBytes("ping", 5);
        sock->sendBytes("ping", 5);
        sock->receiveBytes(message, 5);
        printf("%s\n", message);
    }
}