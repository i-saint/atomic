#ifndef atm_Game_VFX_h
#define atm_Game_VFX_h
#include "Game/VFX/VFXInterfaces.h"

namespace atm {

class IVFXComponent
{
    istSerializeBlock(
    )
public:
    virtual ~IVFXComponent() {}
    virtual void release() { istDelete(this); }

    virtual void frameBegin() {}
    virtual void update(float32 dt) {}
    virtual void asyncupdate(float32 dt) {}
    virtual void draw() {}
    virtual void frameEnd() {}
};

class VFXModule : public IAtomicGameModule
{
typedef IAtomicGameModule super;
private:
    ist::vector<IVFXComponent*> m_components;
    IVFXComponent *m_scintilla;
    IVFXComponent *m_light;
    IVFXComponent *m_shockwave;
    IVFXComponent *m_feedbackblur;

    istSerializeBlockDecl();

public:
    VFXScintilla*       getScintilla()    { return (VFXScintilla*)m_scintilla; }
    VFXLight*           getLight()        { return (VFXLight*)m_light; }
    VFXShockwave*       getShockwave()    { return (VFXShockwave*)m_shockwave; }
    VFXFeedbackBlur*    getFeedbackBlur() { return (VFXFeedbackBlur*)m_feedbackblur; }

public:
    VFXModule();
    ~VFXModule();

    void initialize();
    void frameBegin();
    void update(float32 dt);
    void asyncupdate(float32 dt);
    void draw();
    void frameEnd();
};

} // namespace atm
#endif // atm_Game_VFX_h
