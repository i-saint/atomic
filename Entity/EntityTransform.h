#ifndef atm_Game_Entity_EntityTransform_h
#define atm_Game_Entity_EntityTransform_h
namespace atm {


class Attr_Position
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
            atmECall(move)
            atmECall(getPositionAbs)
            atmECall(setPositionAbs)
        )
    )

    wdmScope(
    void addDebugNodes(const wdmString &path)
    {
        wdmAddNode(path+"/m_pos",   &m_pos,   -3.0f, 3.0f);
    }
    )
    atmJsonizeBlock(
        atmJsonizeMember(m_pos, getPosition, setPosition)
        atmJsonizeCall(move)
    )


public:
    Attr_Position() {}
    const vec3& getPosition() const { return m_pos; }
    void setPosition(const vec3& v) { m_pos=v; }
    void move(const vec3 &v)        { setPosition(getPosition()+v); }
    const vec3& getPositionAbs() const { return m_pos; }
    void setPositionAbs(const vec3& v) { m_pos=v; }

    mat4 computeTransformMatrix() const
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
        istSerialize(m_pos)
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
            atmECall(move)
            atmECall(computeTransformMatrix)
            atmECall(computeRotationMatrix)
            atmECall(getPositionAbs)
            atmECall(setPositionAbs)
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

    atmJsonizeBlock(
        atmJsonizeMember(m_pivot, getPivot, setPivot)
        atmJsonizeMember(m_pos, getPosition, setPosition)
        atmJsonizeMember(m_scale, getScale, setScale)
        atmJsonizeMember(m_axis, getAxis, setAxis)
        atmJsonizeMember(m_rot, getRotate, setRotate)
        atmJsonizeCall( move)
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
    void move(const vec3 &v)        { setPosition(getPosition()+v); }
    const vec3& getPositionAbs() const { return m_pos; }
    void setPositionAbs(const vec3& v) { m_pos=v; }

    mat4 computeTransformMatrix() const
    {
        mat4 mat;
        mat = glm::translate(mat, m_pos);
        mat = glm::rotate(mat, m_rot, m_axis);
        mat = glm::scale(mat, m_scale);
        mat = glm::translate(mat, -m_pivot);
        return mat;
    }

    mat4 computeRotationMatrix() const
    {
        return glm::rotate(m_rot, m_axis);
    }

    void update(float32 dt) {}
    void asyncupdate(float32 dt) {}
};

class Attr_Direction
{
private:
    vec3 m_pivot;
    vec3 m_pos;
    vec3 m_scale;
    vec3 m_dir;
    vec3 m_up;

    istSerializeBlock(
        istSerialize(m_pivot)
        istSerialize(m_pos)
        istSerialize(m_scale)
        istSerialize(m_dir)
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
            atmECall(getDirection)
            atmECall(setDirection)
            atmECall(getUpVector)
            atmECall(setUpVector)
            atmECall(move)
            atmECall(orient)
            atmECall(computeTransformMatrix)
            atmECall(computeRotationMatrix)
            atmECall(getPositionAbs)
            atmECall(setPositionAbs)
        )
    )

    wdmScope(
    void addDebugNodes(const wdmString &path)
    {
        wdmAddNode(path+"/m_pivot", &m_pivot, -3.0f, 3.0f);
        wdmAddNode(path+"/m_pos",   &m_pos, -3.0f, 3.0f);
        wdmAddNode(path+"/m_scale", &m_scale, 0.001f, 4.0f);
        wdmAddNode(path+"/m_orient", this, &Attr_Direction::getDirection, &Attr_Direction::setDirection);
        wdmAddNode(path+"/m_up", &m_up, 0.0f, 360.0f);
    }
    )

    atmJsonizeBlock(
        atmJsonizeMember(m_pivot, getPivot, setPivot)
        atmJsonizeMember(m_pos, getPosition, setPosition)
        atmJsonizeMember(m_scale, getScale, setScale)
        atmJsonizeMember(m_dir, getDirection, setDirection)
        atmJsonizeMember(m_up, getUpVector, setUpVector)
        atmJsonizeCall(move)
        atmJsonizeCall(orient)
    )

public:
    Attr_Direction()
        : m_scale(1.0f, 1.0f, 1.0f)
        , m_dir(1.0f, 0.0f, 0.0f)
        , m_up(1.0f, 0.0f, 0.0f)
    {}

    const vec3& getPivot() const        { return m_pivot; }
    const vec3& getPosition() const     { return m_pos; }
    const vec3& getScale() const        { return m_scale; }
    const vec3& getDirection() const    { return m_dir; }
    const vec3& getUpVector() const     { return m_up; }
    void setPivot(const vec3& v)        { m_pivot=v; }
    void setPosition(const vec3& v)     { m_pos=v; }
    void setScale(const vec3& v)        { m_scale=v; }
    void setDirection(const vec3& v)    { if(glm::dot(v,v)>0.001f) m_dir=glm::normalize(v); }
    void setUpVector(const vec3& v)     { m_up=v; }
    void move(const vec3 &v)            { setPosition(getPosition()+v); }
    void orient(const vec3 &v)          { setDirection(v); }
    const vec3& getPositionAbs() const { return m_pos; }
    void setPositionAbs(const vec3& v) { m_pos=v; }

    mat4 computeTransformMatrix() const
    {
        mat4 mat;
        mat = glm::translate(mat, m_pos);
        mat *= glm::orientation(m_dir, m_up);
        mat = glm::scale(mat, m_scale);
        mat = glm::translate(mat, -m_pivot);
        return mat;
    }

    mat4 computeRotationMatrix() const
    {
        return glm::orientation(m_dir, m_up);
    }
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
            atmECall(move)
            atmECall(computeTransformMatrix)
            atmECall(computeRotationMatrix)
            atmECall(getPositionAbs)
            atmECall(setPositionAbs)
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

    atmJsonizeBlock(
        atmJsonizeMember(m_pivot, getPivot, setPivot)
        atmJsonizeMember(m_pos, getPosition, setPosition)
        atmJsonizeMember(m_scale, getScale, setScale)
        atmJsonizeMember(m_axis1, getAxis1, setAxis1)
        atmJsonizeMember(m_axis2, getAxis2, setAxis2)
        atmJsonizeMember(m_rot1, getRotate1, setRotate1)
        atmJsonizeMember(m_rot2, getRotate2, setRotate2)
        atmJsonizeCall(move)
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
    void move(const vec3 &v)        { setPosition(getPosition()+v); }
    const vec3& getPositionAbs() const { return m_pos; }
    void setPositionAbs(const vec3& v) { m_pos=v; }

    mat4 computeTransformMatrix() const
    {
        mat4 mat;
        mat = glm::translate(mat, m_pos);
        mat = glm::rotate(mat, m_rot2, m_axis2);
        mat = glm::rotate(mat, m_rot1, m_axis1);
        mat = glm::scale(mat, m_scale);
        mat = glm::translate(mat, -m_pivot);
        return mat;
    }

    mat4 computeRotationMatrix() const
    {
        mat4 mat;
        mat = glm::rotate(mat, m_rot2, m_axis2);
        mat = glm::rotate(mat, m_rot1, m_axis1);
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

    atmJsonizeBlock(
        atmJsonizeSuper(super)
        atmJsonizeMember(m_rspeed1, getRotateSpeed1, setRotateSpeed1)
        atmJsonizeMember(m_rspeed2, getRotateSpeed2, setRotateSpeed2)
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
        this->setRotate1(this->getRotate1()+getRotateSpeed1()*dt);
        this->setRotate2(this->getRotate2()+getRotateSpeed2()*dt);
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
            atmECall(getPositionAbs)
            atmECall(setPositionAbs)
            atmECall(getParent)
            atmECall(setParent)
            atmECall(move)
            atmECall(orient)
            atmECall(computeTransformMatrix)
            atmECall(computeRotationMatrix)
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

    atmJsonizeBlock(
        atmJsonizeSuper(super)
        atmJsonizeMember(m_parent, getParent, setParent)
    )

public:
    TAttr_HaveParent() : m_parent(0)
    {}
    EntityHandle getParent() const { return m_parent; }
    void setParent(EntityHandle v) { m_parent=v; }

    vec3 getPositionAbs() const
    {
        vec3 pos = super::getPosition();
        mat4 pmat;
        if(atmQuery(getParent(), getTransformMatrix, pmat)) {
            vec4 tmp = vec4(pos, 1.0f);
            tmp = pmat * tmp;
            pos = vec3(tmp);
        }
        return pos;
    }

    void setPositionAbs(const vec3 &pos)
    {
        mat4 pmat;
        if(atmQuery(getParent(), getInvTransformMatrix, pmat)) {
            vec4 tmp = vec4(pos, 1.0f);
            tmp = pmat * tmp;
            super::setPosition(vec3(tmp));
        }
        else {
            super::setPosition(pos);
        }
    }

    void move(const vec3 &v)
    {
        vec3 m = v;
        mat4 pmat;
        if(atmQuery(getParent(), getInvTransformMatrix, pmat)) {
            m = vec3(pmat*vec4(v,0.0f));
        }
        super::move(m);
    }

    void orient(const vec3 &v)
    {
        vec3 m = v;
        mat4 pmat;
        if(atmQuery(getParent(), getInvTransformMatrix, pmat)) {
            m = vec3(pmat*vec4(v,0.0f));
        }
        super::call(FID_orient, &m, nullptr);
    }

    mat4 computeTransformMatrix() const
    {
        mat4 mat = super::computeTransformMatrix();
        mat4 pmat;
        if(atmQuery(getParent(), getTransformMatrix, pmat)) {
            mat = pmat * mat;
        }
        return mat;
    }

    mat4 computeRotationMatrix() const
    {
        mat4 mat = super::computeRotationMatrix();
        mat4 pmat;
        if(atmQuery(getParent(), computeRotationMatrix, pmat)) {
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
            atmECall(getTransformMatrix)
            atmECall(setTransformMatrix)
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
    const mat4& getTransformMatrix() const { return m_transform; }
    void setTransformMatrix(const mat4 &v) { m_transform=v; }
    void updateTransformMatrix()     { setTransformMatrix(super::computeTransformMatrix()); }
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
            atmECall(getTransformMatrix)
            atmECall(setTransformMatrix)
            atmECall(getInvTransformMatrix)
            atmECall(updateTransformMatrix)
        )
        atmECallSuper(super)
    )
    wdmScope(void addDebugNodes(const wdmString &path) {
        T::addDebugNodes(path);
    })

public:
    const mat4& getTransformMatrix() const      { return m_transform; }
    const mat4& getInvTransformMatrix() const   { return m_itransform; }

    void setTransformMatrix(const mat4 &v)
    {
        m_transform = v;
        m_itransform = glm::inverse(v);
    }

    void updateTransformMatrix()
    {
        setTransformMatrix(super::computeTransformMatrix());
    }
};

} // namespace atm
#endif // atm_Game_Entity_EntityTransform_h
