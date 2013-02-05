#include "stdafx.h"
#include "GameServer.h"
#include "GameSession.h"

namespace atomic {


GameSession::GameSession( const Poco::Net::StreamSocket& s )
    : super(s)
{

}

void GameSession::run()
{

}

} // namespace atomic
