
function radians(v)
{
    return v * (Math.PI/180.0);
}

function each(v, f) {
    for (var i = 0; i < v.length; ++i) { f(v[i]); }
}

function clearChildren(e) {
    while(e.lastChild) { e.removeChild(e.lastChild); }
}
function resizeChildren(e, num, creater) {
    while(e.childNodes.length>num) { e.removeChild(e.lastChild); }
    while(e.childNodes.length<num) { e.appendChild(creater()); }
}


function bool_s(v)  { return "bool("+v.toString()+")"; }
function int32_s(v)   { return "int32("+v.toString()+")"; }
function uint32_s(v)  { return "uint32("+v.toString()+")"; }
function float32_s(v) { return "float32("+v.toString()+")"; }
function vec2_s(v)  { return "vec2("+v[0].toString()+","+v[1].toString()+")"; }
function vec3_s(v)  { return "vec3("+v[0].toString()+","+v[1].toString()+","+v[2].toString()+")"; }
function vec4_s(v)  { return "vec4("+v[0].toString()+","+v[1].toString()+","+v[2].toString()+","+v[3].toString()+")"; }
function string_s(v){ return "string(\""+v+"\")"; }
function instruction_s(p,e) { return "instruction("+p[0].toString()+","+p[1].toString()+",0.0,"+e.toString()+")"; }
function curvepoint_s(p) { return "curvepoint("+p[0].toFixed(2)+","+p[1].toFixed(2)+","+p[2].toFixed(2)+","+p[3].toFixed(2)+","+p[4].toString()+")"; }

function createShader(id)
{
    var shaderScript = document.getElementById(id);
    if(!shaderScript) { return null; }

    var str = "";
    var k = shaderScript.firstChild;
    while (k) {
        if (k.nodeType == 3) {
            str += k.textContent;
        }
        k = k.nextSibling;
    }

    var shader;
    if(shaderScript.type == "x-shader/x-fragment") {
        shader = gl.createShader(gl.FRAGMENT_SHADER);
    }
    else if(shaderScript.type == "x-shader/x-vertex") {
        shader = gl.createShader(gl.VERTEX_SHADER);
    }
    else {
        return null;
    }

    gl.shaderSource(shader, str);
    gl.compileShader(shader);
    if(!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        alert(gl.getShaderInfoLog(shader));
        return null;
    }
    return shader;
}

function createShaderProgram(vsid, psid)
{
    var program = gl.createProgram();
    gl.attachShader(program, createShader(vsid));
    gl.attachShader(program, createShader(psid));
    gl.linkProgram(program);
    return program;
}


function interpolate_linear(v1, v2, u)
{
    var d = v2-v1;
    return v1 + d*u;
}
function interpolate_sin90(v1, v2, u)
{
    var d = v2 - v1;
    return v1 + d*Math.sin(radians(90.0*u));
}
function interpolate_cos90inv(v1, v2, u)
{
    var d = v2 - v1;
    return v1 + d*(1.0-Math.cos(radians(90.0*u)));
}
function interpolate_cos180inv(v1, v2, u)
{
    var d = v2 - v1;
    return v1 + d*(1.0-(Math.cos(radians(180.0*u))/2.0+0.5));
}
function interpolate_pow(v1, v2, p, u)
{
    var d = v2 - v1;
    return v1 + d*Math.pow(u,p);
}
function interpolate_bezier(v1, v1out, v2in, v2, u)
{
    var w = [
        (1.0-u) * (1.0-u) * (1.0-u),
        u * (1.0-u) * (1.0-u)*3.0,
        u * u * (1.0-u)*3.0,
        u * u * u,
    ];
    return w[0]*v1 + w[1]*v1out + w[2]*v2in + w[3]*v2;
}

function interpolate2D_linear(v1, v2, u)
{
    var r = vec2.create();
    var d = vec2.create();
    vec2.sub(d, v2, v1);
    vec2.mul(r, d, vec2.fromValues(u,u));
    vec2.add(r, v1, r);
    return r;
}
function interpolate2D_sin90(v1, v2, u)
{
    var r = vec2.create();
    var d = vec2.create();
    vec2.sub(d, v2, v1);
    var s = Math.sin(radians(90.0*u));
    vec2.mul(r, d, vec2.fromValues(s,s));
    vec2.add(r, v1, r);
    return r;
}
function interpolate2D_cos90inv(v1, v2, u)
{
    var r = vec2.create();
    var d = vec2.create();
    vec2.sub(d, v2, v1);
    var s = 1.0-Math.cos(radians(90.0*u));
    vec2.mul(r, d, vec2.fromValues(s,s));
    vec2.add(r, v1, r);
    return r;
}
function interpolate2D_cos180inv(v1, v2, u)
{
    var r = vec2.create();
    var d = vec2.create();
    vec2.sub(d, v2, v1);
    var s = 1.0-(Math.cos(radians(180.0*u))/2.0+0.5);
    vec2.mul(r, d, vec2.fromValues(s,s));
    vec2.add(r, v1, r);
    return r;
}
function interpolate2D_pow(v1, v2, p, u)
{
    var r = vec2.create();
    var d = vec2.create();
    vec2.sub(d, v2, v1);
    var s = Math.pow(u,p);
    vec2.mul(r, d, vec2.fromValues(s,s));
    vec2.add(r, v1, r);
    return r;
}
function interpolate2D_bezier(v1, v1out, v2in, v2, u)
{
    var w = [
        (1.0-u) * (1.0-u) * (1.0-u),
        u * (1.0-u) * (1.0-u)*3.0,
        u * u * (1.0-u)*3.0,
        u * u * u,
    ];
    var r = vec2.create();
    var m = vec2.create();
    vec2.mul(m, v1, vec2.fromValues(w[0],w[0]));
    vec2.add(r, m, r);
    vec2.mul(m, v1out, vec2.fromValues(w[1],w[1]));
    vec2.add(r, m, r);
    vec2.mul(m, v2in, vec2.fromValues(w[2],w[2]));
    vec2.add(r, m, r);
    vec2.mul(m, v2, vec2.fromValues(w[3],w[3]));
    vec2.add(r, m, r);
    return r;
}

var curve = {
    // interplation types
    None:   0,
    Linear: 1,
    Decel:  2,
    Accel:  3,
    Smooth: 4,
    Bezier: 5,
    End:    6,
    TypeStr: ["None", "Linear", "Decel", "Accel", "Smooth", "Bezier"],

    createPoint: function(time, value, i_in, i_out, interpolation) {
        return [time, value, i_in, i_out, interpolation, true];
    },
    pointGetTime:           function(p) { return p[0]; },
    pointGetValue:          function(p) { return p[1]; },
    pointGetIn:             function(p) { return p[2]; },
    pointGetOut:            function(p) { return p[3]; },
    pointGetInterpolation:  function(p) { return p[4]; },
    interpolate: function(v1, v2, time) {
        var u = (time - v1[0]) / (v2[0]-v1[0]);
        var r;
        switch(v1[4]) {
        case curve.None:   r=v1[1]; break;
        case curve.Linear: r=interpolate_linear(v1[1], v2[1], u); break;
        case curve.Decel:  r=interpolate_sin90(v1[1], v2[1], u); break;
        case curve.Accel:  r=interpolate_cos90inv(v1[1], v2[1], u); break;
        case curve.Smooth: r=interpolate_cos180inv(v1[1], v2[1], u); break;
        case curve.Bezier: r=interpolate_bezier(v1[1], v1[1]+v1[3], v2[1]+v2[2], v2[1], u); break;
        }
        return r;
    },

    createPoints: function () {
        var ret = [];
        ret.addPoint = function(p) {
            this.push(p);
            this.sort(function(a,b){ return a[0]-b[0]; });
        };
        ret.setPoint = function(i,p) {
            this[i] = p;
            this.sort(function(a,b){ return a[0]-b[0]; });
        };
        ret.beginTime = function() {
            return this.length>0 ? this.first()[0] : 0.0;
        };
        ret.endTime = function() {
            return this.length>0 ? this.last()[0] : 0.0;
        }
        ret.computeValue = function(t) {
            var r = 0.0;
            if(this.length==0) {}
            else if(t<=this.first()[0]) { r=this.first()[1]; }
            else if(t>=this.last()[0]) { r=this.last()[1]; }
            else {
                var i2=0;
                for(;; i2++) {
                    var p = this[i2];
                    if(p[0]>t) { break; }
                }
                var i1 = i2-1;
                r = curve.interpolate(this[i1], this[i2], t);
            }
            return r;
        };
        ret.sampling = function(tbegin, tend, interval) {
            var ret = [];
            for(var t=tbegin; t<=tend; t+=interval) {
                ret.push(this.computeValue(t));
            }
            return ret;
        };
        return ret;
    },

};
