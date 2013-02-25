#include "istPCH.h"
#include "EasyDrawer.h"
#include "EasyDrawerUtil.h"

ist_EasyDrawer_NamespaceBegin



istInterModule void DrawLine( EasyDrawer &drawer, const EasyDrawState &state, const vec2 &pos1, const vec2 &pos2, const vec4 &color )
{
    DrawLine(drawer, state, pos1, pos2, color, color);
}

istInterModule void DrawLine( EasyDrawer &drawer, const EasyDrawState &state, const vec2 &pos1, const vec2 &pos2, const vec4 &color1, const vec4 &color2 )
{
    VertexP2C4 v[2] = {
        VertexP2C4(pos1, color1),
        VertexP2C4(pos2, color2),
    };
    drawer.draw(state, I3D_LINES, v, _countof(v));
}

istInterModule void DrawOutlineRect( EasyDrawer &drawer, const EasyDrawState &state, const vec2 &ur, const vec2 &bl, const vec4 &color )
{
    DrawOutlineRect(drawer, state, ur, bl, color, color, color, color);
}

istInterModule void DrawOutlineRect( EasyDrawer &drawer, const EasyDrawState &state, const vec2 &ur, const vec2 &bl, const vec4 &cur, const vec4 &cul, const vec4 &cbl, const vec4 &cbr )
{
    VertexP2C4 v[4] = {
        VertexP2C4(vec2(ur.x, ur.y), cur),
        VertexP2C4(vec2(bl.x, ur.y), cul),
        VertexP2C4(vec2(bl.x, bl.y), cbl),
        VertexP2C4(vec2(ur.x, bl.y), cbr),
    };
    VertexP2C4 vb[] = {
        v[0],v[1], v[1],v[2], v[2],v[3], v[3],v[0],
    };
    drawer.draw(state, I3D_LINES, vb, _countof(vb));
}

istInterModule void DrawRect( EasyDrawer &drawer, const EasyDrawState &state, const vec2 &ur, const vec2 &bl, const vec4 &color )
{
    DrawRect(drawer, state, ur, bl, color, color, color, color);
}

istInterModule void DrawRect( EasyDrawer &drawer, const EasyDrawState &state, const vec2 &ur, const vec2 &bl, const vec4 &cur, const vec4 &cul, const vec4 &cbl, const vec4 &cbr )
{
    VertexP2C4 v[4] = {
        VertexP2C4(vec2(ur.x, ur.y), cur),
        VertexP2C4(vec2(bl.x, ur.y), cul),
        VertexP2C4(vec2(bl.x, bl.y), cbl),
        VertexP2C4(vec2(ur.x, bl.y), cbr),
    };
    VertexP2C4 vb[] = {
        v[0],v[1],v[2], v[2],v[3],v[0],
    };
    drawer.draw(state, I3D_TRIANGLES, vb, _countof(vb));
}

istInterModule void DrawRectT( EasyDrawer &drawer, const EasyDrawState &state, const vec2 &ur, const vec2 &bl, const vec2 &tur, const vec2 &tbl, const vec4 &color )
{
    VertexP2T2C4 v[4] = {
        VertexP2T2C4(vec2(ur.x, ur.y), vec2(tur.x, tur.y), color),
        VertexP2T2C4(vec2(bl.x, ur.y), vec2(tbl.x, tur.y), color),
        VertexP2T2C4(vec2(bl.x, bl.y), vec2(tbl.x, tbl.y), color),
        VertexP2T2C4(vec2(ur.x, bl.y), vec2(tur.x, tbl.y), color),
    };
    VertexP2T2C4 vb[] = {
        v[0],v[1],v[2], v[2],v[3],v[0],
    };
    drawer.draw(state, I3D_TRIANGLES, vb, _countof(vb));
}

ist_EasyDraw_NamespaceEnd
