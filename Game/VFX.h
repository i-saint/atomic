#ifndef __atomic_Game_VFX__
#define __atomic_Game_VFX__

namespace atomic
{


class VFXSet
{
private:
    VFXSet *m_prev;

public:
    VFXSet();
    ~VFXSet();

    void initialize(VFXSet* prev);

    void update();
    void draw();
    void sync();

public:
};


class Task_WorldUpdate;
class Task_WorldDraw;


} // namespace atomic
#endif // __atomic_Game_VFX__
