
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
    createPoint: function(time, value, i_in, i_out, interpolation) {
        return [time, value, i_in, i_out, interpolation];
    },
    pointGetTime:           function(p) { return p[0]; },
    pointGetValue:          function(p) { return p[1]; },
    pointGetIn:             function(p) { return p[2]; },
    pointGetOut:            function(p) { return p[3]; },
    pointGetInterpolation:  function(p) { return p[4]; },

    createPoints: function () {
        var ret = [];
        ret.addPoint = function(p) {
            this.push(p);
            this.sort(function(a,b){ return a[0]<b[0]; });
        };
        ret.setPoint = function(i,p) {
            this[i] = p;
            this.sort(function(a,b){ return a[0]<b[0]; });
        };
        ret.computeValue = function(t) {

        };
        return ret;
    },
};
