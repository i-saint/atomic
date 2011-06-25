#ifndef __atomic_Game_VFX__
#define __atomic_Game_VFX__

namespace atomic
{


class ForceSet
{
private:
    ForceSet *m_prev, *m_next

public:
    ForceSet(ForceSet* prev);
    ~ForceSet();

    void update();
    void sync();
    void flushMessage();
    void processMessage();
    void draw();

};


class Task_WorldUpdate;
class Task_WorldDraw;


} // namespace atomic
#endif // __atomic_Game_VFX__
