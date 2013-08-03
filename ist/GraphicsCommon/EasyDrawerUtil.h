#ifndef ist_GraphicsCommon_EasyDrawerUtil_h
#define ist_GraphicsCommon_EasyDrawerUtil_h

#include "EasyDrawer.h"

ist_EasyDrawer_NamespaceBegin

istAPI void DrawLine(
    EasyDrawer &drawer,
    const vec2 &pos1, const vec2 &pos2,
    const vec4 &color);
istAPI void DrawLine(
    EasyDrawer &drawer,
    const vec2 &pos1, const vec2 &pos2,
    const vec4 &color1, const vec4 &color2);

istAPI void DrawOutlineRect(
    EasyDrawer &drawer,
    const vec2 &ur, const vec2 &bl,
    const vec4 &color );
istAPI void DrawOutlineRect(
    EasyDrawer &drawer,
    const vec2 &ur, const vec2 &bl,
    const vec4 &cbl, const vec4 &cul, const vec4 &cur, const vec4 &cbr );

istAPI void DrawRect(
    EasyDrawer &drawer,
    const vec2 &ur, const vec2 &bl,
    const vec4 &color );
istAPI void DrawRect(
    EasyDrawer &drawer,
    const vec2 &bl, const vec2 &ur,
    const vec4 &cbl, const vec4 &cul, const vec4 &cur, const vec4 &cbr );
istAPI void DrawRectT(
    EasyDrawer &drawer,
    const vec2 &bl, const vec2 &ur,
    const vec2 &tbl=vec2(0.0f,0.0f), const vec2 &tur=vec2(1.0f,1.0f),
    const vec4 &color=vec4(1.0f) );

ist_EasyDraw_NamespaceEnd
#endif // ist_GraphicsCommon_EasyDrawerUtil_h
