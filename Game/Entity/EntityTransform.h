#ifndef atm_Game_Entity_EntityTransform_h
#define atm_Game_Entity_EntityTransform_h
namespace atm {


class Attr_Translate
{
protected:
    vec3 m_pos;

    istSerializeBlock(
        istSerialize(m_pos)
    )
public:
    atmECallBlock(
        atmMethodBlock(
        atmECall(getPosition)
        atmECall(setPosition)
        )
    )

    wdmScope(
    void addDebugNodes(const wdmString &path)
    {
        wdmAddNode(path+"/m_pos",   &m_pos,   -3.0f, 3.0f);
    }
    )

public:
    Attr_Translate() {}
    const vec3& getPosition() const { return m_pos; }
    void setPosition(const vec3& v) { m_pos=v; }

    mat4 computeMatrix() const
    {
        mat4 mat;
        mat = glm::translate(mat, m_pos);
        return mat;
    }
};

class Attr_Transform
{
private:
    vec3 m_pivot;
    vec3 m_pos;
    vec3 m_scale;
    vec3 m_axis;
    float32 m_rot;

    istSerializeBlock(
        istSerialize(m_pivot)
        istSerialize(m_scale)
        istSerialize(m_axis)
        istSerialize(m_rot)
    )
public:
    atmECallBlock(
        atmMethodBlock(
        atmECall(getPivot)
        atmECall(setPivot)
        atmECall(getPosition)
        atmECall(setPosition)
        atmECall(getScale)
        atmECall(setScale)
        atmECall(getAxis)
        atmECall(setAxis)
        atmECall(getRotate)
        atmECall(setRotate)
        )
    )

    wdmScope(
    void addDebugNodes(const wdmString &path)
    {
        wdmAddNode(path+"/m_pivot", &m_pivot, -3.0f, 3.0f);
        wdmAddNode(path+"/m_pos",   &m_pos,   -3.0f, 3.0f);
        wdmAddNode(path+"/m_scale", &m_scale,  0.001f, 4.0f);
        wdmAddNode(path+"/m_rot",   &m_rot,    0.0f, 360.0f);
    }
    )

public:
    Attr_Transform()
        : m_scale(1.0f, 1.0f, 1.0f)
        , m_axis(0.0f, 0.0f, 1.0f)
        , m_rot(0.0f)
    {}

    const vec3& getPivot() const    { return m_pivot; }
    const vec3& getPosition() const { return m_pos; }
    const vec3& getScale() const    { return m_scale; }
    const vec3& getAxis() const     { return m_axis; }
    float32 getRotate() const       { return m_rot; }
    void setPivot(const vec3& v)    { m_pivot=v; }
    void setPosition(const vec3& v) { m_pos=v; }
    void setScale(const vec3& v)    { m_scale=v; }
    void setAxis(const vec3& v)     { m_axis=v; }
    void setRotate(float32 v)       { m_rot=v; }

    mat4 computeMatrix() const
    {
        mat4 mat;
        mat = glm::translate(mat, m_pos);
        mat = glm::rotate(mat, m_rot, m_axis);
        mat = glm::scale(mat, m_scale);
        mat = glm::translate(mat, -m_pivot);
        return mat;
    }

    void update(float32 dt) {}
    void asyncupdate(float32 dt) {}
};

class Attr_Orientation
{
private:
    vec3 m_pivot;
    vec3 m_pos;
    vec3 m_scale;
    vec3 m_oriantation;
    vec3 m_up;

    istSerializeBlock(
        istSerialize(m_pivot)
        istSerialize(m_pos)
        istSerialize(m_scale)
        istSerialize(m_oriantation)
        istSerialize(m_up)
    )
public:
    atmECallBlock(
        atmMethodBlock(
        atmECall(getPivot)
        atmECall(setPivot)
        atmECall(getPosition)
        atmECall(setPosition)
        atmECall(getScale)
        atmECall(setScale)
        atmECall(getOrientation)
        atmECall(setOrientation)
        atmECall(getUpVector)
        atmECall(setUpVector)
        )
    )

    wdmScope(
    void addDebugNodes(const wdmString &path)
    {
        wdmAddNode(path+"/m_pivot", &m_pivot, -3.0f, 3.0f);
        wdmAddNode(path+"/m_pos",   &m_pos, -3.0f, 3.0f);
        wdmAddNode(path+"/m_scale", &m_scale, 0.001f, 4.0f);
        wdmAddNode(path+"/m_oriantation", this, &Attr_Orientation::getOrientation, &Attr_Orientation::setOrientation);
        wdmAddNode(path+"/m_up", &m_up, 0.0f, 360.0f);
    }
    )

public:
    Attr_Orientation()
        : m_scale(1.0f, 1.0f, 1.0f)
        , m_oriantation(1.0f, 0.0f, 0.0f)
        , m_up(1.0f, 0.0f, 0.0f)
    {}

    const vec3& getPivot() const        { return m_pivot; }
    const vec3& getPosition() const     { return m_pos; }
    const vec3& getScale() const        { return m_scale; }
    const vec3& getOrientation() const  { return m_oriantation; }
    const vec3& getUpVector() const     { return m_up; }
    void setPivot(const vec3& v)        { m_pivot=v; }
    void setPosition(const vec3& v)     { m_pos=v; }
    void setScale(const vec3& v)        { m_scale=v; }
    void setOrientation(const vec3& v)  { m_oriantation=glm::normalize(v); }
    void setUpVector(const vec3& v)     { m_up=v; }

    mat4 computeMatrix() const
    {
        mat4 mat;
        mat = glm::translate(mat, m_pos);
        mat *= glm::orientation(m_oriantation, m_up);
        mat = glm::scale(mat, m_scale);
        mat = glm::translate(mat, -m_pivot);
        return mat;
    }

    void update(float32 dt) {}
    void asyncupdate(float32 dt) {}
};

class Attr_DoubleAxisRotation
{
private:
    vec3 m_pivot;
    vec3 m_pos;
    vec3 m_scale;
    vec3 m_axis1;
    vec3 m_axis2;
    float32 m_rot1;
    float32 m_rot2;

    istSerializeBlock(
        istSerialize(m_pivot)
        istSerialize(m_pos)
        istSerialize(m_scale)
        istSerialize(m_axis1)
        istSerialize(m_axis2)
        istSerialize(m_rot1)
        istSerialize(m_rot2)
    )

public:
    atmECallBlock(
        atmMethodBlock(
            atmECall(getPivot)
            atmECall(setPivot)
            atmECall(getPosition)
            atmECall(setPosition)
            atmECall(getScale)
            atmECall(setScale)
            atmECall(getAxis1)
            atmECall(setAxis1)
            atmECall(getAxis2)
            atmECall(setAxis2)
            atmECall(getRotate1)
            atmECall(setRotate1)
            atmECall(getRotate2)
            atmECall(setRotate2)
        )
    )

    wdmScope(
    void addDebugNodes(const wdmString &path)
    {
        wdmAddNode(path+"/m_pivot", &m_pivot, -3.0f, 3.0f);
        wdmAddNode(path+"/m_pos", &m_pos, -3.0f, 3.0f);
        wdmAddNode(path+"/m_scale", &m_scale, 0.001f, 4.0f);
        wdmAddNode(path+"/m_rot1", &m_rot1, 0.0f, 360.0f);
        wdmAddNode(path+"/m_rot2", &m_rot2, 0.0f, 360.0f);
    }
    )

public:
    Attr_DoubleAxisRotation()
        : m_scale(1.0f, 1.0f, 1.0f)
        , m_axis1(0.0f, 1.0f, 0.0f)
        , m_axis2(0.0f, 0.0f, 1.0f)
        , m_rot1(0.0f), m_rot2(0.0f)
    {
    }

    const vec3& getPivot() const    { return m_pivot; }
    const vec3& getPosition() const { return m_pos; }
    const vec3& getScale() const    { return m_scale; }
    const vec3& getAxis1() const    { return m_axis1; }
    const vec3& getAxis2() const    { return m_axis2; }
    float32 getRotate1() const      { return m_rot1; }
    float32 getRotate2() const      { return m_rot2; }

    void setPivot(const vec3& v)    { m_pivot=v; }
    void setPosition(const vec3& v) { m_pos=v; }
    void setScale(const vec3& v)    { m_scale=v; }
    void setAxis1(const vec3& v)    { m_axis1=v; }
    void setAxis2(const vec3& v)    { m_axis2=v; }
    void setRotate1(float32 v)      { m_rot1=v; }
    void setRotate2(float32 v)      { m_rot2=v; }

    mat4 computeMatrix() const
    {
        mat4 mat;
        mat = glm::translate(mat, m_pos);
        mat = glm::rotate(mat, m_rot2, m_axis2);
        mat = glm::rotate(mat, m_rot1, m_axis1);
        mat = glm::scale(mat, m_scale);
        mat = glm::translate(mat, -m_pivot);
        return mat;
    }
};

template<class T>
class TAttr_RotateSpeed : public T
{
typedef T super;
private:
    float32 m_rspeed1;
    float32 m_rspeed2;

    istSerializeBlock(
        istSerializeBase(super)
        istSerialize(m_rspeed1)
        istSerialize(m_rspeed2)
    )

public:
    atmECallBlock(
        atmMethodBlock(
            atmECall(getRotateSpeed1)
            atmECall(setRotateSpeed1)
            atmECall(getRotateSpeed2)
            atmECall(setRotateSpeed2)
        )
        atmECallSuper(super)
    )

    wdmScope(
    void addDebugNodes(const wdmString &path)
    {
        T::addDebugNodes(path);
        wdmAddNode(path+"/m_rspeed1", &m_rspeed1, -3.0f, 3.0f);
        wdmAddNode(path+"/m_rspeed2", &m_rspeed2, -3.0f, 3.0f);
    }
    )

public:
    TAttr_RotateSpeed() : m_rspeed1(0.0f), m_rspeed2(0.0f)
    {}
    float32 getRotateSpeed1() const { return m_rspeed1; }
    float32 getRotateSpeed2() const { return m_rspeed2; }
    void setRotateSpeed1(float32 v) { m_rspeed1=v; }
    void setRotateSpeed2(float32 v) { m_rspeed2=v; }

    void updateRotate(float32 dt)
    {
        this->setRotate1(this->getRotate1()+getRotateSpeed1());
        this->setRotate2(this->getRotate2()+getRotateSpeed2());
    }
};


template<class T>
class TAttr_HaveParent : public T
{
typedef T super;
private:
    EntityHandle m_parent;

    istSerializeBlock(
        istSerializeBase(super)
        istSerialize(m_parent)
    )

public:
    atmECallBlock(
        atmMethodBlock(
            atmECall(getParent)
            atmECall(setParent)
        )
        atmECallSuper(super)
    )

    wdmScope(
    void addDebugNodes(const wdmString &path)
    {
        super::addDebugNodes(path);
        wdmAddNode(path+"/m_parent", (const uint32*)&m_parent);
    }
    )

public:
    TAttr_HaveParent() : m_parent(0)
    {}
    EntityHandle getParent() const { return m_parent; }
    void setParent(EntityHandle v) { m_parent=v; }

    mat4 computeMatrix() const
    {
        mat4 mat = super::computeMatrix();
        mat4 pmat;
        if(atmQuery(getParent(), getTransform, pmat)) {
            mat = pmat * mat;
        }
        return mat;
    }
};


template<class T>
class TAttr_TransformMatrix : public T
{
typedef T super;
private:
    mat4 m_transform;

    istSerializeBlock(
        istSerializeBase(super)
        istSerialize(m_transform)
    )

public:
    atmECallBlock(
        atmMethodBlock(
            atmECall(getTransform)
            atmECall(setTransform)
            atmECall(updateTransformMatrix)
        )
        atmECallSuper(super)
    )

    wdmScope(
    void addDebugNodes(const wdmString &path) {
        super::addDebugNodes(path);
    }
    )

public:
    const mat4& getTransform() const { return m_transform; }
    void setTransform(const mat4 &v) { m_transform=v; }
    void updateTransformMatrix()     { setTransform(super::computeMatrix()); }
};

template<class T>
class TAttr_TransformMatrixI : public T
{
typedef T super;
private:
    mat4 m_transform;
    mat4 m_itransform;

    istSerializeBlock(
        istSerializeBase(super)
        istSerialize(m_transform)
        istSerialize(m_itransform)
    )

public:
    atmECallBlock(
        atmMethodBlock(
            atmECall(getTransform)
            atmECall(setTransform)
            atmECall(getInverseTransform)
            atmECall(updateTransformMatrix)
        )
        atmECallSuper(super)
    )
    wdmScope(void addDebugNodes(const wdmString &path) {
        T::addDebugNodes(path);
    })

public:
    const mat4& getTransform() const        { return m_transform; }
    const mat4& getInverseTransform() const { return m_itransform; }

    void setTransform(const mat4 &v)
    {
        m_transform = v;
        m_itransform = glm::inverse(v);
    }

    void updateTransformMatrix()
    {
        setTransform(super::computeMatrix());
    }
};

} // namespace atm
#endif // atm_Game_Entity_EntityTransform_h
