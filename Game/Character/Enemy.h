#ifndef __atomic_Game_Character_Enemy__
#define __atomic_Game_Character_Enemy__

#include "Game/Entity.h"
#include "Game/EntityQuery.h"
#include "Attributes.h"

namespace atomic {

    class IRoutine
    {
    protected:
        IEntity *m_obj;

    public:
        IRoutine()  : m_obj(NULL) {}
        virtual ~IRoutine() {}
        IEntity* getEntity() { return m_obj; }
        void setEntity(IEntity *v) { m_obj=v; }

        virtual void update(float32 dt)=0;
        virtual void asyncupdate(float32 dt)=0;
        virtual void draw() {}

        virtual bool call(uint32 call_id, const variant &v) { return false; }
        virtual bool query(uint32 query_id, variant &v) const { return false; }
    };


    class Breakable : public IEntity
    {
    typedef IEntity super;
    private:
        mat4        m_transform;
        IRoutine    *m_routine;
        float32     m_health;

    public:
        Breakable() : m_transform(), m_routine(NULL), m_health(1.0f) {}

        float32     getHealth() const   { return m_health; }
        const mat4& getTransform() const{ return m_transform; }
        IRoutine*   getRoutine()        { return m_routine; }

        void setHealth(float32 v)        { m_health=v; }
        void setTransform(const mat4& v){ m_transform=v; }
        void setRoutine(IRoutine *v)    { m_routine=v; }

        virtual void update(float32 dt)
        {
            if(m_routine) { m_routine->update(dt); }
        }

        virtual void asyncupdate(float32 dt)
        {
            if(m_routine) { m_routine->asyncupdate(dt); }
        }

        virtual void onDamage(const DamageMessage &m)
        {
        }

        virtual void damage(float32 d)
        {
            if(m_health > 0.0f) {
                m_health -= d;
                if(m_health <= 0.0f) {
                    destroy();
                }
            }
        }

        virtual void destroy()
        {
            atomicGetEntitySet()->deleteEntity(getHandle());
        }

        virtual bool call(uint32 call_id, const variant &v)
        {
            switch(call_id) {
            DEFINE_ECALL1(setHealth, float32);
            DEFINE_ECALL1(damage, float32);
            default: return super::call(call_id, v);
            }
        }

        virtual bool query(uint32 query_id, variant &v) const
        {
            switch(query_id) {
                DEFINE_EQUERY(getHealth);
            }
            return false;
        }
    };


} // namespace atomic
#endif // __atomic_Game_Character_Enemy__
