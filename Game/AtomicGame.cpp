#include "stdafx.h"
#include "../types.h"
#include "Message.h"
#include "World.h"
#include "AtomicApplication.h"
#include "AtomicGame.h"
#include "../Graphics/GraphicResourceManager.h"
#include "../Graphics/Renderer.h"

namespace atomic {


AtomicGame::AtomicGame()
: m_current_world(0)
, m_draw_target(NULL)
{
    AtomicRenderer::initializeInstance();
    MessageRouter::initializeInstance();
    World::initializeInterframe();

    m_worlds.resize(MAX_WORLDS);
    m_worlds[0] = AT_ALIGNED_NEW(World, 16)();
    m_worlds[0]->initialize(NULL, *stl::get_default_allocator(NULL));
}

AtomicGame::~AtomicGame()
{
    for(uint32 i=0; i<MAX_WORLDS; ++i) {
        AT_DELETE(m_worlds[i]);
    }

    World::finalizeInterframe();
    MessageRouter::finalizeInstance();
    AtomicRenderer::finalizeInstance();
}


void AtomicGame::update()
{
    World *w = m_worlds[m_current_world];
    w->update();
    m_draw_target = w;

    // todo: —v‹ƒƒ‚ƒŠŽZo «‚Ý‚½‚¢‚ÈB
    // uint32 required_memory = w->getRequiredMemoryOnNextFrame();
    // m_frame_allocator.reserve(required_memory);

    uint32 next_world = (m_current_world+1) % MAX_WORLDS;
    World *n = AT_ALIGNED_NEW(World, 16)();
    n->initialize(w, *stl::get_default_allocator(NULL));
    AT_DELETE(m_worlds[next_world]);
    m_worlds[next_world] = n;
    m_current_world = next_world;
}


void AtomicGame::draw()
{
    WaitForDrawComplete();
    AtomicRenderer::getInstance()->beforeDraw();
    if(m_draw_target) {
        m_draw_target->draw();
    }
    KickDraw();
}



} // namespace atomic