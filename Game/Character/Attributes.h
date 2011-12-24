#ifndef __atomic_Game_Character_Attributes__
#define __atomic_Game_Character_Attributes__
namespace atomic {

    class Attr_RefCount
    {
    private:
        uint32 m_ref_count;

    protected:
        void setRefCount(uint32 v) { m_ref_count=v; }

    public:
        Attr_RefCount() : m_ref_count(1) {}
        uint32 getRefCount() const { return m_ref_count; }
        void addRefCount() { ++m_ref_count; }
        void release() { --m_ref_count; }

        bool call(uint32 call_id, const variant &v)
        {
            switch(call_id) {
                DEFINE_ECALL1(setRefCount, uint32);
                DEFINE_ECALL0(addRefCount);
                DEFINE_ECALL0(release);
            }
            return false;
        }

        bool query(uint32 query_id, variant &v) const
        {
            switch(query_id) {
                DEFINE_EQUERY(getRefCount);
            }
            return false;
        }
    };

    class Attr_Translate
    {
    private:
        vec4 m_pos;

    public:
        Attr_Translate() {}
        const vec4& getPosition() const { return m_pos; }
        void setPosition(const vec4& v) { m_pos=v; }

        mat4 computeMatrix() const
        {
            mat4 mat;
            mat = glm::translate(mat, reinterpret_cast<const vec3&>(m_pos));
            return mat;
        }

        bool call(uint32 call_id, const variant &v)
        {
            switch(call_id) {
                DEFINE_ECALL1(setPosition, vec4);
            }
            return false;
        }

        bool query(uint32 query_id, variant &v) const
        {
            switch(query_id) {
                DEFINE_EQUERY(getPosition);
            }
            return false;
        }
    };

    class Attr_Transform
    {
    private:
        vec4 m_pos;
        vec4 m_scale;
        vec4 m_axis;
        float32 m_rot;

    public:
        Attr_Transform() : m_rot(0.0f) {}

        const vec4& getPosition() const { return m_pos; }
        const vec4& getScale() const    { return m_scale; }
        const vec4& getAxis() const     { return m_axis; }
        float32 getRotate() const     { return m_rot; }

        void setPosition(const vec4& v) { m_pos=v; }
        void setScale(const vec4& v)    { m_scale=v; }
        void setAxis(const vec4& v)     { m_axis=v; }
        void setRotate(float32 v)     { m_rot=v; }

        mat4 computeMatrix() const
        {
            mat4 mat;
            mat = glm::translate(mat, reinterpret_cast<const vec3&>(m_pos));
            mat = glm::rotate(mat, m_rot, reinterpret_cast<const vec3&>(m_axis));
            mat = glm::scale(mat, reinterpret_cast<const vec3&>(m_scale));
            return mat;
        }

        bool call(uint32 call_id, const variant &v)
        {
            switch(call_id) {
                DEFINE_ECALL1(setPosition, vec4);
                DEFINE_ECALL1(setScale, vec4);
                DEFINE_ECALL1(setAxis, vec4);
                DEFINE_ECALL1(setRotate, float32);
            }
            return false;
        }

        bool query(uint32 query_id, variant &v) const
        {
            switch(query_id) {
                DEFINE_EQUERY(getPosition);
                DEFINE_EQUERY(getScale);
                DEFINE_EQUERY(getAxis);
                DEFINE_EQUERY(getRotate);
            }
            return false;
        }

        void update(float32 dt) {}
        void updateAsync(float32 dt) {}
    };

    class Attr_DoubleAxisRotation
    {
    private:
        vec4 m_pos;
        vec4 m_scale;
        vec4 m_axis1;
        vec4 m_axis2;
        float32 m_rot1;
        float32 m_rot2;

    public:
        Attr_DoubleAxisRotation()
            : m_scale(1.0f, 1.0f, 1.0f, 0.0f)
            , m_axis1(0.0f, 1.0f, 0.0f, 0.0f)
            , m_axis2(0.0f, 1.0f, 0.0f, 0.0f)
            , m_rot1(0.0f), m_rot2(0.0f) {}

        const vec4& getPosition() const { return m_pos; }
        const vec4& getScale() const    { return m_scale; }
        const vec4& getAxis1() const    { return m_axis1; }
        const vec4& getAxis2() const    { return m_axis2; }
        float32 getRotate1() const    { return m_rot1; }
        float32 getRotate2() const    { return m_rot2; }

        void setPosition(const vec4& v) { m_pos=v; }
        void setScale(const vec4& v)    { m_scale=v; }
        void setAxis1(const vec4& v)    { m_axis1=v; }
        void setAxis2(const vec4& v)    { m_axis2=v; }
        void setRotate1(float32 v)    { m_rot1=v; }
        void setRotate2(float32 v)    { m_rot2=v; }

        mat4 computeMatrix() const
        {
            mat4 mat;
            mat = glm::translate(mat, reinterpret_cast<const vec3&>(m_pos));
            mat = glm::rotate(mat, m_rot2, reinterpret_cast<const vec3&>(m_axis2));
            mat = glm::rotate(mat, m_rot1, reinterpret_cast<const vec3&>(m_axis1));
            mat = glm::scale(mat, reinterpret_cast<const vec3&>(m_scale));
            return mat;
        }

        bool call(uint32 call_id, const variant &v)
        {
            switch(call_id) {
                DEFINE_ECALL1(setPosition, vec4);
                DEFINE_ECALL1(setScale, vec4);
                DEFINE_ECALL1(setAxis1, vec4);
                DEFINE_ECALL1(setAxis2, vec4);
                DEFINE_ECALL1(setRotate1, float32);
                DEFINE_ECALL1(setRotate2, float32);
            }
            return false;
        }

        bool query(uint32 query_id, variant &v) const
        {
            switch(query_id) {
                DEFINE_EQUERY(getPosition);
                DEFINE_EQUERY(getScale);
                DEFINE_EQUERY(getAxis1);
                DEFINE_EQUERY(getAxis2);
                DEFINE_EQUERY(getRotate1);
                DEFINE_EQUERY(getRotate2);
            }
            return false;
        }

        void update(float32 dt) {}
        void updateAsync(float32 dt) {}
    };

    template<class T>
    class TAttr_RotateSpeed : public T
    {
    typedef T super;
    private:
        float32 m_rspeed1;
        float32 m_rspeed2;

    public:
        TAttr_RotateSpeed() : m_rspeed1(0.0f), m_rspeed2(0.0f)
        {}

        float32 getRotateSpeed1() const { return m_rspeed1; }
        float32 getRotateSpeed2() const { return m_rspeed2; }
        void setRotateSpeed1(float32 v) { m_rspeed1=v; }
        void setRotateSpeed2(float32 v) { m_rspeed2=v; }

        bool call(uint32 call_id, const variant &v)
        {
            switch(call_id) {
                DEFINE_ECALL1(setRotateSpeed1, float32);
                DEFINE_ECALL1(setRotateSpeed2, float32);
            }
            return super::call(call_id, v);
        }

        bool query(uint32 query_id, variant &v) const
        {
            switch(query_id) {
                DEFINE_EQUERY(getRotateSpeed1);
                DEFINE_EQUERY(getRotateSpeed2);
            }
            return super::query(query_id, v);
        }

        void update(float32 dt)
        {
            this->setRotate1(this->getRotate1()+getRotateSpeed1());
            this->setRotate2(this->getRotate2()+getRotateSpeed2());
        }

        void updateAsync(float32 dt) {}
    };

} // namespace atomic
#endif // __atomic_Game_Character_Attributes__
