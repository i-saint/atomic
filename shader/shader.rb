#!/usr/bin/ruby

OUT_HEADER = "glsl_source.h"
OUT_SOURCE = "glsl_source.cpp"
SH_REG = /\.(fsh|vsh)/


def update()
  o_header = open(OUT_HEADER, "wb")
  o_source = open(OUT_SOURCE, "wb")
  o_header << "namespace atomic {\n"
  o_source << "#include \"stdafx.h\"\n"
  o_source << "namespace atomic {\n"

  Dir.foreach(".") do |f|
    next if !f.match(SH_REG)

    symbol = "g_"+f.gsub(/[.]/, "_")
    data = ""
    open(f, "rb") do |fin|
      fin.each_line do |l|
        data += l.gsub(/^ +/, "").gsub(/[\r\n]/, "")+"\\n"
      end
    end
    o_header << "extern const char* #{symbol};\n"
    o_source << "const char* #{symbol} = \""+data+"\";\n"
    puts f
  end
  o_header << "}\n"
  o_source << "}\n"
end

def is_needs_update()
  begin
    t = File.ctime(OUT_HEADER)
    Dir.foreach(".") do |f|
      next if !f.match(SH_REG)
      return true if File.ctime(f) > t
    end
  rescue
    return true
  end
  false
end



update() if is_needs_update()
