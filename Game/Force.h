#ifndef __atomic_Game_Force_h__
#define __atomic_Game_Force_h__

namespace atomic {


class IForce
{
public:
    virtual ~IForce() {}
    virtual void taskBeforeDraw()=0;
    virtual void taskAfterDraw()=0;
    virtual void taskDraw() const=0;
    virtual void taskCopy() const=0;
};


class ForceSet
{
private:
    typedef stl::vector<IForce*> ForceCont;

    const ForceSet *m_prev;
    ForceSet *m_next;
    ForceCont m_forces;


public:
    ForceSet(ForceSet* prev);
    ~ForceSet();

    void initialize();
    void serialize(Serializer& s) const;
    void deserialize(Deserializer& s);

    void update();
    void draw() const;
    void sync() const;

    void setNext(ForceSet *next);
    ForceSet* getNext() { return m_next; }
    const ForceSet* getPrev() const { return m_prev; }

public:
};




} // namespace atomic
#endif // __atomic_Game_Force_h__
