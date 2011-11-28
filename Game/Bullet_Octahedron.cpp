#include "stdafx.h"
#include "ist/ist.h"
#include "types.h"
#include "Game/Message.h"
#include "Game/Bullet.h"
#include "Game/Bullet_Octahedron.h"

namespace atomic {


Bullet_OctahedronSet::Bullet_OctahedronSet()
{
}

Bullet_OctahedronSet::~Bullet_OctahedronSet()
{
}

void Bullet_OctahedronSet::update()
{
    uint32 num_data = m_data.size();
    uint32 begin = 0;
    uint32 end = num_data;
    for(uint32 i=begin; i<end; i+=4) {
        uint32 e = std::min<uint32>(4, end-i);
        BulletData *data = &m_data[i];

        SOAVECTOR3 pos4         = SOAVectorTranspose3(data[0].pos, data[1].pos, data[2].pos, data[3].pos);
        SOAVECTOR3 vel4         = SOAVectorTranspose3(data[0].vel, data[1].vel, data[2].vel, data[3].vel);
        pos4 = SOAVectorAdd3(pos4, vel4);

        // todo: 円やボックスの接線方向に位置/速度を矯正

        SOAVECTOR4 pos_nextv  = SOAVectorTranspose4(pos4.x, pos4.y, pos4.z);
        SOAVECTOR4 vel_nextv  = SOAVectorTranspose4(vel4.x, vel4.y, vel4.z);
        for(uint32 i=0; i<e; ++i) {
            data[i].pos = pos_nextv.v[i];
            data[i].vel = vel_nextv.v[i];
        }

    }
}

void Bullet_OctahedronSet::draw()
{
}

void Bullet_OctahedronSet::updateAsync() const
{
}


} // namespace atomic
