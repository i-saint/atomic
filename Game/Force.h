#ifndef __atomic_Game_Force_h__
#define __atomic_Game_Force_h__

namespace atomic {


class IForce
{
public:
    virtual ~IForce() {}
    virtual void update()=0;
    virtual void draw()=0;
    virtual void updateAsync()=0;
};


class ForceSet
{
private:
    typedef stl::vector<IForce*> ForceCont;


public:
    ForceSet(ForceSet* prev);
    ~ForceSet();

    void initialize();
    void serialize(Serializer& s) const;
    void deserialize(Deserializer& s);

    void update();
    void draw() const;
    void sync() const;
    void updateAsync();

public:
};




} // namespace atomic
#endif // __atomic_Game_Force_h__
