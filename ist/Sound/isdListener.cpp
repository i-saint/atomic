#include "stdafx.h"
#include "../Sound.h"

namespace ist {
namespace isd {

int Listener::getI(ALenum param) const
{
    int v = 0;
    alGetListeneri(param, &v);
    return v;
}

float Listener::getF(ALenum param) const
{
    float r;
    alGetListenerf(param, &r);
    return r;
}

vec3 Listener::get3F(ALenum param) const
{
    vec3 r;
    alGetListenerfv(param, (ALfloat*)&r);
    return r;
}

void Listener::setI(ALenum param, int v)
{
    alListeneri(param, v);
}

void Listener::setF(ALenum param, float v)
{
    alListenerf(param, v);
}

void Listener::set3F(ALenum param, const vec3& v)
{
    alListenerfv(param, (ALfloat*)&v);
}


Listener::Listener(){}
Listener::~Listener() {}

float Listener::getGain() const { return getF(AL_GAIN); }
vec3 Listener::getPosition() const { return get3F(AL_POSITION); }
vec3 Listener::getVelocity() const { return get3F(AL_VELOCITY); }

void Listener::setGain(float v) { setF(AL_GAIN, v); }
void Listener::setPosition(const vec3 &v) { set3F(AL_POSITION, v); }
void Listener::setVelocity(const vec3 &v) { set3F(AL_VELOCITY, v); }


} // namespace isd
} // namespace ist
