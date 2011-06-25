#include "stdafx.h"
#include "../types.h"
#include "World.h"
#include "AtomicGame.h"
#include "AtomicApplication.h"


namespace atomic {


AtomicApplication::AtomicApplication()
    : m_request_exit(false)
    , m_game(NULL)
{
}


bool AtomicApplication::Initialize(size_t x, size_t y, size_t width, size_t height, const wchar_t *title, bool fullscreen)
{
    if(!super::Initialize(x,y, width, height, title, fullscreen))
    {
        return false;
    }
    TaskScheduler::initializeSingleton(11);

    m_game = EA_ALIGNED_NEW(AtomicGame, 16) AtomicGame();
    return true;
}

void AtomicApplication::Finalize()
{
    EA_DELETE(m_game);

    TaskScheduler::finalizeSingleton();
    super::Finalize();
}

void AtomicApplication::mainLoop()
{
    while(!m_request_exit)
    {
        translateMessage();

        if(m_game) {
            m_game->update();
            m_game->draw();
        }
    }
}

int AtomicApplication::handleWindowMessage(const ist::WindowMessage& wm)
{
    switch(wm.type)
    {
    case ist::WindowMessage::MES_CLOSE:
        {
            m_request_exit = true;
        }
        return 0;

    case ist::WindowMessage::MES_KEYBOARD:
        {
            ist::WM_Keyboard m = (ist::WM_Keyboard&)wm;
            if(m.action==ist::WM_Keyboard::ACT_KEYUP && m.key==ist::WM_Keyboard::KEY_ESCAPE) {
                m_request_exit = true;
            }
        }
    }

    return 0;
}


} // namespace atomic