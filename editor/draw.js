var sh_entities;
var sh_particles;
var sh_points;
var sh_lasers;
var vb_quad;
var ib_quad;
var ib_quad_line;
var vb_cube;
var ib_cube;
var ib_cube_line;
var vb_bullets;
var vb_lasers;
var vb_fluids;
var vb_curve;
var vb_curve_points;

function initializeGLResources()
{
    sh_entities = createShaderProgram("vs_entities", "ps_entities");
    sh_entities.ia_position = gl.getAttribLocation(sh_entities, "ia_position");
    sh_entities.u_proj = gl.getUniformLocation(sh_entities, "u_proj");
    sh_entities.u_trans = gl.getUniformLocation(sh_entities, "u_trans");
    sh_entities.u_size = gl.getUniformLocation(sh_entities, "u_size");
    sh_entities.u_color = gl.getUniformLocation(sh_entities, "u_color");

    sh_particles = createShaderProgram("vs_particles", "ps_particles");
    sh_particles.ia_position = gl.getAttribLocation(sh_particles, "ia_position");
    sh_particles.u_proj = gl.getUniformLocation(sh_particles, "u_proj");
    sh_particles.u_pointSize = gl.getUniformLocation(sh_particles, "u_pointSize");
    sh_particles.u_color = gl.getUniformLocation(sh_particles, "u_color");

    sh_points = createShaderProgram("vs_points", "ps_points");
    sh_points.ia_position = gl.getAttribLocation(sh_points, "ia_position");
    sh_points.ia_color = gl.getAttribLocation(sh_points, "ia_color");
    sh_points.u_proj = gl.getUniformLocation(sh_points, "u_proj");
    sh_points.u_pointSize = gl.getUniformLocation(sh_points, "u_pointSize");

    sh_lasers = createShaderProgram("vs_lasers", "ps_lasers");
    sh_lasers.ia_position = gl.getAttribLocation(sh_lasers, "ia_position");
    sh_lasers.u_proj = gl.getUniformLocation(sh_lasers, "u_proj");
    sh_lasers.u_color = gl.getUniformLocation(sh_lasers, "u_color");


    var quad_vertices = new Float32Array([
        -1.0, -1.0,
        -1.0, 1.0,
         1.0, 1.0,
         1.0, -1.0
    ]);
    vb_quad = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vb_quad);
    gl.bufferData(gl.ARRAY_BUFFER, quad_vertices, gl.STATIC_DRAW);

    var quad_indices = new Int16Array([0, 1, 2, 2, 3, 0]);
    ib_quad = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ib_quad);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, quad_indices, gl.STATIC_DRAW);

    var quad_line_indices = new Int16Array([0, 1, 1, 2, 2, 3, 3, 0]);
    ib_quad_line = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ib_quad_line);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, quad_line_indices, gl.STATIC_DRAW);


    var cube_vertices = new Float32Array([
         1.0, 1.0, 1.0,
        -1.0, 1.0, 1.0,
        -1.0, -1.0, 1.0,
         1.0, -1.0, 1.0,
         1.0, 1.0, -1.0,
        -1.0, 1.0, -1.0,
        -1.0, -1.0, -1.0,
         1.0, -1.0, -1.0,
    ]);
    vb_cube = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vb_cube);
    gl.bufferData(gl.ARRAY_BUFFER, cube_vertices, gl.STATIC_DRAW);

    var cube_indices = new Int16Array([
        0, 1, 2, 2, 3, 0,
        0, 4, 5, 5, 1, 0,
        2, 1, 5, 5, 6, 2,
        3, 7, 4, 4, 0, 3,
        3, 2, 6, 6, 7, 3,
        7, 6, 5, 5, 4, 7,
    ]);
    ib_cube = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ib_cube);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, cube_indices, gl.STATIC_DRAW);

    var cube_line_indices = new Int16Array([
        0, 1, 1, 2, 2, 3, 3, 0,
        0, 4, 4, 5, 5, 1, 1, 0,
        2, 1, 1, 5, 5, 6, 6, 2,
        3, 7, 7, 4, 4, 0, 0, 3,
        3, 2, 2, 6, 6, 7, 7, 3,
        7, 6, 6, 5, 5, 4, 4, 7,
    ]);
    ib_cube_line = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ib_cube_line);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, cube_line_indices, gl.STATIC_DRAW);

    vb_bullets = gl.createBuffer();
    vb_lasers = gl.createBuffer();
    vb_fluids = gl.createBuffer();
    vb_curve = gl.createBuffer();
    vb_curve_points = gl.createBuffer();
}


function drawGL() {
    camera.updateMatrix();
    var projection = camera.viewProj;

    gl.clearColor(0.0, 0.0, 0.0, 1.0);
    gl.clear(gl.COLOR_BUFFER_BIT);

    gl.enable(gl.CULL_FACE);
    gl.cullFace(gl.BACK);
    gl.enable(gl.BLEND);

    gl.blendFunc(gl.SRC_ALPHA, gl.ONE);
    gl.useProgram(sh_entities);
    gl.bindBuffer(gl.ARRAY_BUFFER, vb_cube);
    gl.enableVertexAttribArray(sh_entities.ia_position);
    gl.vertexAttribPointer(sh_entities.ia_position, 3, gl.FLOAT, gl.FALSE, 12, 0);
    gl.uniformMatrix4fv(sh_entities.u_proj, gl.FALSE, projection);
    for (var i = 0; i < entities.ids.length; ++i) {
        var ipp = i + 1;
        gl.uniformMatrix4fv(sh_entities.u_trans, gl.FALSE, entities.trans.subarray(16 * i, 16 * ipp));
        gl.uniform3fv(sh_entities.u_size, entities.size.subarray(3 * i, 3 * ipp));
        if (editor.entitySelection.include(entities.ids[i])) {
            gl.uniform4fv(sh_entities.u_color, [1, 1, 1, 0.3]);
        }
        else {
            gl.uniform4fv(sh_entities.u_color, entities.color.subarray(4 * i, 4 * ipp));
        }
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ib_cube);
        gl.drawElements(gl.TRIANGLES, 36, gl.UNSIGNED_SHORT, 0);
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ib_cube_line);
        gl.drawElements(gl.LINES, 48, gl.UNSIGNED_SHORT, 0);
    }


    if (entities.fluids.length) {
        gl.useProgram(sh_particles);
        gl.bindBuffer(gl.ARRAY_BUFFER, vb_fluids);
        gl.bufferData(gl.ARRAY_BUFFER, entities.fluids, gl.STREAM_DRAW);
        gl.enableVertexAttribArray(sh_particles.ia_position);
        gl.vertexAttribPointer(sh_particles.ia_position, 2, gl.FLOAT, gl.FALSE, 8, 0);
        gl.uniformMatrix4fv(sh_particles.u_proj, gl.FALSE, projection);
        gl.uniform4fv(sh_particles.u_color, [0.0, 0.5, 1.0, 0.3]);
        gl.uniform1f(sh_particles.u_pointSize, 5.0);
        gl.drawArrays(gl.POINTS, 0, entities.fluids.length / 2);
    }
    if (entities.bullets.length) {
        gl.useProgram(sh_particles);
        gl.bindBuffer(gl.ARRAY_BUFFER, vb_bullets);
        gl.bufferData(gl.ARRAY_BUFFER, entities.bullets, gl.STREAM_DRAW);
        gl.enableVertexAttribArray(sh_particles.ia_position);
        gl.vertexAttribPointer(sh_particles.ia_position, 2, gl.FLOAT, gl.FALSE, 8, 0);
        gl.uniformMatrix4fv(sh_particles.u_proj, gl.FALSE, projection);
        gl.uniform4fv(sh_particles.u_color, [1.0, 1.0, 0.0, 1.0]);
        gl.uniform1f(sh_particles.u_pointSize, 8.0);
        gl.drawArrays(gl.POINTS, 0, entities.bullets.length / 2);
    }
    if (entities.lasers.length) {
        gl.useProgram(sh_lasers);
        gl.bindBuffer(gl.ARRAY_BUFFER, vb_lasers);
        gl.bufferData(gl.ARRAY_BUFFER, entities.lasers, gl.STREAM_DRAW);
        gl.enableVertexAttribArray(sh_lasers.ia_position);
        gl.vertexAttribPointer(sh_lasers.ia_position, 3, gl.FLOAT, gl.FALSE, 12, 0);
        gl.uniformMatrix4fv(sh_lasers.u_proj, gl.FALSE, projection);
        gl.uniform4fv(sh_lasers.u_color, [1.0, 1.0, 0.0, 1.0]);
        gl.lineWidth(10.0);
        gl.drawArrays(gl.LINES, 0, entities.lasers.length / 3);
        gl.lineWidth(1.0);
    }

    var xline = editor.curve.x;
    var yline = editor.curve.y;
    var sx = editor.curveSelection[0];
    var sy = editor.curveSelection[1];
    if (xline.length || yline.length) {
        var tbegin = [xline.beginTime(), yline.beginTime()].min();
        var tend = [xline.endTime(), yline.endTime()].max();
        var xv = xline.sampling(tbegin, tend, 10.0);
        var yv = yline.sampling(tbegin, tend, 10.0);
        var vertices = [];
        for (var i = 0; i < xv.length; ++i) {
            vertices.push(xv[i]);
            vertices.push(yv[i]);
        }
        gl.useProgram(sh_particles);
        gl.bindBuffer(gl.ARRAY_BUFFER, vb_curve_points);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.STREAM_DRAW);
        gl.enableVertexAttribArray(sh_particles.ia_position);
        gl.vertexAttribPointer(sh_particles.ia_position, 2, gl.FLOAT, gl.FALSE, 8, 0);
        gl.uniformMatrix4fv(sh_particles.u_proj, gl.FALSE, projection);
        gl.uniform4fv(sh_particles.u_color, [0.75, 0.75, 1.0, 0.5]);
        gl.uniform1f(sh_particles.u_pointSize, 4.0);
        gl.drawArrays(gl.POINTS, 0, vertices.length / 2);
        gl.drawArrays(gl.LINE_STRIP, 0, vertices.length / 2);


        vertices.clear();
        if (sx > -1 && sy > -1 && xline[sx][4] == curve.Bezier && yline[sy][4] == curve.Bezier) {
            vertices.push(xline[sx][1] + xline[sx][2]);
            vertices.push(yline[sy][1] + yline[sy][2]);
            vertices.push(xline[sx][1]);
            vertices.push(yline[sy][1]);
            vertices.push(xline[sx][1] + xline[sx][3]);
            vertices.push(yline[sy][1] + yline[sy][3]);
        }
        gl.useProgram(sh_particles);
        gl.bindBuffer(gl.ARRAY_BUFFER, vb_curve_points);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.STREAM_DRAW);
        gl.uniform1f(sh_particles.u_pointSize, 8.0);
        gl.drawArrays(gl.POINTS, 0, vertices.length / 2);
        gl.drawArrays(gl.LINE_STRIP, 0, vertices.length / 2);


        vertices.clear();
        var red = [1.0, 0.0, 0.0, 0.5];
        var green = [0.0, 1.0, 0.0, 0.5];
        var white = [1.0, 1.0, 1.0, 0.5];
        for(var i=0; i < xline.length; ++i) {
            vertices.push(xline[i][1]);
            vertices.push(yline.computeValue(xline[i][0]));
            var c = i==sx ? white : red;
            c.each(function(a){ vertices.push(a); });
        }
        for(var i=0; i < yline.length; ++i) {
            vertices.push(xline.computeValue(yline[i][0]));
            vertices.push(yline[i][1]);
            var c = i==sy ? white : green;
            c.each(function (a) { vertices.push(a); });
        }
        gl.useProgram(sh_points);
        gl.bindBuffer(gl.ARRAY_BUFFER, vb_curve_points);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.STREAM_DRAW);
        gl.enableVertexAttribArray(sh_points.ia_position);
        gl.enableVertexAttribArray(sh_points.ia_color);
        gl.vertexAttribPointer(sh_points.ia_position, 2, gl.FLOAT, gl.FALSE, 24, 0);
        gl.vertexAttribPointer(sh_points.ia_color, 4, gl.FLOAT, gl.FALSE, 24, 8);
        gl.uniformMatrix4fv(sh_points.u_proj, gl.FALSE, projection);
        gl.uniform1f(sh_points.u_pointSize, 8.0);
        gl.drawArrays(gl.POINTS, 0, vertices.length / 6);
    }

    gl.flush();
}
