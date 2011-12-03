#!/usr/bin/ruby

SH_OUT_HEADER = "glsl_source.h"
SH_OUT_SOURCE = "glsl_source.cpp"
SH_REG = /\.(fsh|vsh|glsl|glslh)/




def sh_preprocess(src)
  out = ""

  # process #pragma include("path to header")
  src = src.gsub(/#pragma\s+include\("(.+?)"\)/) do
    puts "including #{$1}"
    included = ""
    open($1, "rb") do |fin|
      included = sh_preprocess(fin.read)
    end
    included
  end

  src.each_line do |l|
    out += l.gsub(/^ +/, "").gsub(/[\r\n]/, "")+"\\n"
  end
  out
end

def sh_update()
  o_header = open(SH_OUT_HEADER, "wb")
  o_source = open(SH_OUT_SOURCE, "wb")
  o_header << "namespace atomic {\n"
  o_source << "#include \"stdafx.h\"\n"
  o_source << "namespace atomic {\n"

  Dir.foreach(".") do |f|
    next if !f.match(SH_REG)

    sym = "g_"+f.gsub(/[.]/, "_")
    source = ""
    open(f, "rb") do |fin|
      source = sh_preprocess(fin.read)
    end
    o_header << "extern const char* #{sym};\n"
    o_source << "const char* #{sym} = \""+source+"\";\n"
    puts f
  end
  o_header << "}\n"
  o_source << "}\n"
end

def sh_is_needs_update()
  begin
    t = File.ctime(SH_OUT_HEADER)
    Dir.foreach(".") do |f|
      next if !f.match(SH_REG) && !f.match(/\.rb$/) && !f.match(/\.h$/)
      return true if File.ctime(f) > t
    end
  rescue
    return true
  end
  false
end


sh_update() if sh_is_needs_update()
