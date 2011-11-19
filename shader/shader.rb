#!/usr/bin/ruby

SH_OUT_HEADER = "glsl_source.h"
SH_OUT_SOURCE = "glsl_source.cpp"
SH_REG = /\.(fsh|vsh)/

CL_OUT_HEADER = "cl_source.h"
CL_OUT_SOURCE = "cl_source.cpp"
CL_REG = /\.(cl)/


def sh_update()
  o_header = open(SH_OUT_HEADER, "wb")
  o_source = open(SH_OUT_SOURCE, "wb")
  o_header << "namespace atomic {\n"
  o_source << "#include \"stdafx.h\"\n"
  o_source << "namespace atomic {\n"

  Dir.foreach(".") do |f|
    next if !f.match(SH_REG)

    sym = "g_"+f.gsub(/[.]/, "_")
    data = ""
    open(f, "rb") do |fin|
      fin.each_line do |l|
        data += l.gsub(/^ +/, "").gsub(/[\r\n]/, "")+"\\n"
      end
    end
    o_header << "extern const char* #{sym};\n"
    o_source << "const char* #{sym} = \""+data+"\";\n"
    puts f
  end
  o_header << "}\n"
  o_source << "}\n"
end

def sh_is_needs_update()
  begin
    t = File.ctime(SH_OUT_HEADER)
    Dir.foreach(".") do |f|
      next if !f.match(SH_REG)
      return true if File.ctime(f) > t
    end
  rescue
    return true
  end
  false
end



def cl_update()
  o_header = open(CL_OUT_HEADER, "wb")
  o_source = open(CL_OUT_SOURCE, "wb")
  o_header << "namespace atomic {\n"
  o_source << "#include \"stdafx.h\"\n"
  o_source << "namespace atomic {\n"

  Dir.foreach(".") do |f|
    next if !f.match(CL_REG)

    sym = "g_"+f.gsub(/[.]/, "_")
    data = ""
    open(f, "rb") do |fin|
      fin.each_line do |l|
        data += l.gsub(/^ +/, "").gsub(/[\r\n]/, "")+"\\n"
      end
    end
    o_header << "extern const char* #{sym};\n"
    o_source << "const char* #{sym} = \""+data+"\";\n"
    puts f
  end
  o_header << "}\n"
  o_source << "}\n"
end

def cl_is_needs_update()
  begin
    t = File.ctime(CL_OUT_HEADER)
    Dir.foreach(".") do |f|
      next if !f.match(CL_REG)
      return true if File.ctime(f) > t
    end
  rescue
    return true
  end
  false
end



sh_update() if sh_is_needs_update()
cl_update() if cl_is_needs_update()
