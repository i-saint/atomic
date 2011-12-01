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
    void set3F(ALenum param, const float4& v);

public:
    Listener();
    virtual ~Listener();
    float getGain() const;
    float4 getPosition() const;
    float4 getVelocity() const;

    void setGain(float v);
    void setPosition(const float4& v);
    void setVelocity(const float4& v);
};

} // namespace sound
} // namespace ist

#endif // __ist_Sound_Listener__
