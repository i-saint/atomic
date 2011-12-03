#include "stdafx.h"
#include "../Graphics.h"
#include <glm/gtc/matrix_transform.hpp>

namespace ist {
namespace graphics {


bool Viewport::bind() const
{
    glViewport(getX(), getY(), getWidth(), getHeight());
    return true;
}

void Camera::updateMatrix()
{
    m_mv_matrix = glm::lookAt(vec3(m_position), vec3(m_target), vec3(m_up));
}

bool Camera::bind() const
{
    const float *pos = (const float*)&getPosition();
    const float *tar = (const float*)&getTarget();
    const float *dir = (const float*)&getUp();

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    gluLookAt(
        pos[0], pos[1], pos[2],
        tar[0], tar[1], tar[2],
        dir[0], dir[1], dir[2]);
    return true;
}


void OrthographicCamera::updateMatrix()
{
    super::updateMatrix();
    m_p_matrix = glm::ortho( m_left, m_right, m_bottom, m_top, m_znear, m_zfar);
    m_mvp_matrix = getModelViewMatrix()*getProjectionMatrix();
}

bool OrthographicCamera::bind() const
{
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(getLeft(), getRight(), getBottom(), getTop(), getZNear(), getZFar());
    return super::bind();
}


void PerspectiveCamera::updateMatrix()
{
    super::updateMatrix();
    m_p_matrix = glm::perspective(m_fovy, m_aspect, m_znear, m_zfar);
    m_mvp_matrix = getProjectionMatrix()*getModelViewMatrix();
}

bool PerspectiveCamera::bind() const
{
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(getFovy(), getAspect(), getZNear(), getZFar());
    return super::bind();
}


} // namespace graphics
} // namespace ist
