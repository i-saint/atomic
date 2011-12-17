#ifndef __atomic_Game_Character_Attributes__
#define __atomic_Game_Character_Attributes__
namespace atomic {

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
        float32 getRotation() const     { return m_rot; }

        void setPosition(const vec4& v) { m_pos=v; }
        void setScale(const vec4& v)    { m_scale=v; }
        void setAxis(const vec4& v)     { m_axis=v; }
        void setRotation(float32 v)     { m_rot=v; }

        mat4 computeMatrix() const
        {
            mat4 mat;
            mat = glm::translate(mat, reinterpret_cast<const vec3&>(m_pos));
            mat = glm::rotate(mat, m_rot, reinterpret_cast<const vec3&>(m_axis));
            mat = glm::scale(mat, reinterpret_cast<const vec3&>(m_scale));
            return mat;
        }
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
        Attr_DoubleAxisRotation() : m_rot1(0.0f), m_rot2(0.0f) {}

        const vec4& getPosition() const { return m_pos; }
        const vec4& getScale() const    { return m_scale; }
        const vec4& getAxis1() const    { return m_axis1; }
        const vec4& getAxis2() const    { return m_axis2; }
        float32 getRotation1() const    { return m_rot1; }
        float32 getRotation2() const    { return m_rot2; }

        void setPosition(const vec4& v) { m_pos=v; }
        void setScale(const vec4& v)    { m_scale=v; }
        void setAxis1(const vec4& v)    { m_axis1=v; }
        void setAxis2(const vec4& v)    { m_axis2=v; }
        void setRotation1(float32 v)    { m_rot1=v; }
        void setRotation2(float32 v)    { m_rot2=v; }

        mat4 computeMatrix() const
        {
            mat4 mat;
            mat = glm::translate(mat, reinterpret_cast<const vec3&>(m_pos));
            mat = glm::rotate(mat, m_rot2, reinterpret_cast<const vec3&>(m_axis2));
            mat = glm::rotate(mat, m_rot1, reinterpret_cast<const vec3&>(m_axis1));
            mat = glm::scale(mat, reinterpret_cast<const vec3&>(m_scale));
            return mat;
        }
    };


} // namespace atomic
#endif // __atomic_Game_Character_Attributes__
