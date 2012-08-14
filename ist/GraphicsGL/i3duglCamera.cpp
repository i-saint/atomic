#include "stdafx.h"
#ifdef __ist_with_OpenGL__
#include "ist/Base.h"
#include "i3duglCamera.h"
#include <glm/gtc/matrix_transform.hpp>

namespace ist {
namespace i3dgl {


bool Viewport::bind() const
{
    glViewport(m_pos.x, m_pos.y, m_size.x, m_size.y);
    return true;
}



void Camera::updateMatrix()
{
    m_v_matrix = glm::lookAt(vec3(m_position), vec3(m_target), vec3(m_up));
}


void OrthographicCamera::updateMatrix()
{
    super::updateMatrix();
    m_p_matrix = glm::ortho( m_left, m_right, m_bottom, m_top, m_znear, m_zfar);
    m_vp_matrix = getViewMatrix()*getProjectionMatrix();
}



void PerspectiveCamera::updateMatrix()
{
    super::updateMatrix();
    m_p_matrix = glm::perspective(m_fovy, m_aspect, m_znear, m_zfar);
    m_vp_matrix = getProjectionMatrix()*getViewMatrix();
}


} // namespace i3d
} // namespace ist
#endif // __ist_with_OpenGL__
