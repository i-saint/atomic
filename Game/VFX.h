#ifndef __atomic_Game_VFX__
#define __atomic_Game_VFX__

namespace atomic
{


class VFXSet
{
private:
    VFXSet *m_prev, *m_next;

public:
    VFXSet(VFXSet* prev);
    ~VFXSet();

    void update();
    void draw();

};


class Task_WorldUpdate;
class Task_WorldDraw;


} // namespace atomic
#endif // __atomic_Game_VFX__
