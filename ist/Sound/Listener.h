#ifndef __ist_Sound_Listener__
#define __ist_Sound_Listener__

namespace ist {
namespace sound {

 
class Listener
{
private:

protected:
    int     getI(ALenum param) const;
    float   getF(ALenum param) const;
    float4  get3F(ALenum param) const;
    void setI(ALenum param, int v);
    void setF(ALenum param, float v);
    void set3F(ALenum param, const vec3& v);

public:
    Listener();
    virtual ~Listener();
    float getGain() const;
    float4 getPosition() const;
    float4 getVelocity() const;

    void setGain(float v);
    void setPosition(const vec3& v);
    void setVelocity(const vec3& v);
};

} // namespace sound
} // namespace ist

#endif // __ist_Sound_Listener__
