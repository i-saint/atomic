
function radians(v)
{
    return v * (Math.PI/180.0);
}

function each(v, f) {
    for (var i = 0; i < v.length; ++i) { f(v[i]); }
}

function clearChildren(e) {
    while (e.firstChild) { e.removeChild(e.firstChild); }
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

var curve = {
    // interplation types
    None:   0,
    Linear: 1,
    Decel:  2,
    Accel:  3,
    Smooth: 4,
    Pow:    5,
    Bezier: 6,

    createPoint: function(time, value, i_in, i_out, interpolation) {
        return [time, value, i_in, i_out, interpolation];
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
        case curve.Pow   : r=interpolate_pow(v1[1], v2[1], v1[3], u); break;
        case curve.Bezier: r=interpolate_bezier(v1[1], v1[3], v2[2], v2[1], u); break;
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
                    if(p[0]>=t) { break; }
                }
                var i1 = i2-1;
                r = curve.interpolate(this[i1], this[i2], t);
            }
            return r;
        };
        ret.sampling = function(interval) {
            var tbegin = this.beginTime();
            var tend = this.endTime();
            var ret = [];
            for(var t=tbegin; t<=tend; t+=interval) {
                ret.push(this.computeValue(t));
            }
            return ret;
        };
        return ret;
    },

};
