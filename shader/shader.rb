#!/usr/bin/ruby

require 'fileutils'
include FileUtils

SH_PREPROCESS_VS = "cl /EP /u /DGLSL /DGLSL_VS "
SH_PREPROCESS_PS = "cl /EP /u /DGLSL /DGLSL_PS "


def sh_strip(src)
  src.gsub(/\r\n/, "\n").sub(/^(\n)+/, "").gsub(/\n(\n)+/, "\n\n").gsub("#pragma ", "#")
end

def sh_update()
  mkpath("tmp")
  Dir.foreach(".") do |f|
    next if !f.match(/\.glsl$/)

    vs_path = "tmp/"+f.sub(/\.glsl$/, ".vs")
    ps_path = "tmp/"+f.sub(/\.glsl$/, ".ps")
    next if(
      (File.exists?(vs_path) && File.ctime(vs_path) > File.ctime(f)) &&
      (File.exists?(ps_path) && File.ctime(ps_path) > File.ctime(f)) )

    open(vs_path, "wb") do |fout|
      fout.write sh_strip(`#{SH_PREPROCESS_VS} #{f}`)
    end
    open(ps_path, "wb") do |fout|
      fout.write sh_strip(`#{SH_PREPROCESS_PS} #{f}`)
    end
  end
end

sh_update()
