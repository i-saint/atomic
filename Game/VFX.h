#ifndef __atomic_Game_VFX_h__
#define __atomic_Game_VFX_h__

namespace atomic {


class VFXSet
{
private:
    VFXSet *m_prev;

public:
    VFXSet();
    ~VFXSet();

    void initialize();
    void serialize(Serializer& s) const;
    void deserialize(Deserializer& s);

    void update();
    void draw() const;
    void sync() const;

public:
    void taskBeforeDraw();
    void taskAfterDraw();
    void taskDraw() const;
    void taskCopy(VFXSet *dst) const;
};


} // namespace atomic
#endif // __atomic_Game_VFX_h__
