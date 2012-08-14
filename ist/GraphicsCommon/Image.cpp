#include "stdafx.h"
#include <fstream>
#include "ist/GraphicsCommon/Image.h"

extern "C" {
#ifdef __ist_with_png__
#include <png.h>
#endif // __ist_with_png__
#ifdef __ist_with_jpeg__
#include <jpeglib.h>
#include <jerror.h>
#endif // __ist_with_jpeg__
}

#ifdef istWindows
#ifdef __ist_with_png__
#pragma comment(lib,"zlib.lib")
#pragma comment(lib,"libpng.lib")
#endif // __ist_with_png__

#ifdef __ist_with_jpeg__
#pragma comment(lib,"libjpeg.lib")
#endif // __ist_with_jpeg__
#endif // istWindows

namespace ist {


int8 GetFormatByExtention(const std::string& filename)
{
    uint32 len = filename.size();
    if(len<5) { return Image::FORMAT_UNKNOWN; }

    if(strncmp(&filename[len-3], "bmp", 3)==0) { return Image::FORMAT_BMP; }
    if(strncmp(&filename[len-3], "tga", 3)==0) { return Image::FORMAT_TGA; }
    if(strncmp(&filename[len-3], "png", 3)==0) { return Image::FORMAT_PNG; }
    if(strncmp(&filename[len-3], "jpg", 3)==0) { return Image::FORMAT_JPG; }
    return Image::FORMAT_UNKNOWN;
}


bool Image::load(const std::string& filename)
{
    IOConfig conf;
    conf.setPath(filename);
    conf.setFormat(GetFormatByExtention(filename));
    return load(conf);
}

bool Image::load(const IOConfig& conf)
{
    std::fstream f(conf.getPath().c_str(), std::ios::binary|std::ios::in);
    return load(f, conf);
}

bool Image::load(std::istream& f, const IOConfig& conf)
{
    return load(*f.rdbuf(), conf);
}

bool Image::load(std::streambuf& f, const IOConfig& conf)
{
    clear();

    switch(conf.getFormat())
    {
    case FORMAT_BMP: return loadBMP(f, conf);
    case FORMAT_TGA: return loadTGA(f, conf);
    case FORMAT_PNG: return loadPNG(f, conf);
    case FORMAT_JPG: return loadJPG(f, conf);
    }
    istPrint("認識できないフォーマットが指定されました。\n");
    return false;
}


bool Image::save(const std::string& filename) const
{
    IOConfig conf;
    conf.setPath(filename);
    conf.setFormat(GetFormatByExtention(filename));
    return save(conf);
}

bool Image::save(const IOConfig& conf) const
{
    std::fstream f(conf.getPath().c_str(), std::ios::binary|std::ios::out);
    return save(f, conf);
}

bool Image::save(std::ostream& f, const IOConfig& conf) const
{
    return save(*f.rdbuf(), conf);
}

bool Image::save(std::streambuf& f, const IOConfig& conf) const
{
    switch(conf.getFormat())
    {
    case FORMAT_BMP: return saveBMP(f, conf);
    case FORMAT_TGA: return saveTGA(f, conf);
    case FORMAT_PNG: return savePNG(f, conf);
    case FORMAT_JPG: return saveJPG(f, conf);
    }
    istPrint("認識できないフォーマットが指定されました。\n");
    return false;
}





struct BMPHEAD
{
    char B;
    char M;
    int32 file_size;
    int16 reserve1;
    int16 reserve2;
    int32 offset;

    BMPHEAD()
    {
        memset(this, 0, sizeof(*this));
        B = 'B';
        M = 'M';
        offset = 54;
    }
};

struct BMPINFOHEAD
{
    int32 header_size;
    int32 width;
    int32 height;
    int16 plane;
    int16 bits;
    int32 compression;
    int32 comp_image_size;
    int32 x_resolution;
    int32 y_resolution;
    int32 pallete_num;
    int32 important_pallete_num;

    BMPINFOHEAD()
    {
        memset(this, 0, sizeof(*this));
        header_size=sizeof(*this);
        plane = 1;
        bits = 24;
    }
};

struct TGAHEAD
{
    uint8 No_ID;
    uint8 CMap_Type;
    uint8 image_type;
    uint8 CMap_Spec[5];
    int16 Ox;
    int16 Oy;
    int16 width;
    int16 height;
    uint8 pixel;
    uint8 IDesc_Type;

    TGAHEAD()
    {
        memset(this, 0, sizeof(*this));
        pixel = 32;
        IDesc_Type = 8;
    }
};


static bRGBA Read1Pixel(ist::bistream& bf)
{
    bRGBA t;
    bf >> t.b >> t.g >> t.r >> t.a;
    return t;
}



// BMP

bool Image::loadBMP(std::streambuf& f, const IOConfig& conf)
{
    ist::biostream bf(f);

    BMPHEAD head;
    BMPINFOHEAD infohead;

    bf >> head.B >> head.M;
    if(head.B!='B' || head.M!='M') { return false; }

    bf  >> head.file_size
        >> head.reserve1
        >> head.reserve2
        >> head.offset;

    bf  >> infohead.header_size
        >> infohead.width
        >> infohead.height
        >> infohead.plane
        >> infohead.bits
        >> infohead.compression
        >> infohead.comp_image_size
        >> infohead.x_resolution
        >> infohead.y_resolution
        >> infohead.pallete_num
        >> infohead.important_pallete_num;

    if(infohead.bits!=24) { return false; }

    resize(infohead.width, infohead.height);

    for(uint32 i=height()-1; i>=0; --i) {
        for(uint32 j=0; j<width(); ++j) {
            bRGBA& c = (*this)[i][j];
            bf >> c.b >> c.g >> c.r;
            c.a = 255;
        }
    }

    return true;
}


bool Image::saveBMP(std::streambuf& f, const IOConfig& conf) const
{
    ist::biostream bf(f);

    BMPHEAD head;
    BMPINFOHEAD infohead;

    head.file_size = sizeof(BMPHEAD)+sizeof(BMPINFOHEAD)+width()*height()*3;
    infohead.width = width();
    infohead.height = height();

    bf  << head.B
        << head.M
        << head.file_size
        << head.reserve1
        << head.reserve2
        << head.offset;
    bf  << infohead.header_size
        << infohead.width
        << infohead.height
        << infohead.plane
        << infohead.bits
        << infohead.compression
        << infohead.comp_image_size
        << infohead.x_resolution
        << infohead.y_resolution
        << infohead.pallete_num
        << infohead.important_pallete_num;

    for(uint32 i=height()-1; i>=0; --i) {
        for(uint32 j=0; j<width(); ++j) {
            const bRGBA& c = (*this)[i][j];
            bf << c.b << c.g << c.r;
        }
    }
    return true;
}





// TGA

bool Image::loadTGA(std::streambuf& f, const IOConfig& conf)
{
    ist::biostream bf(f);

    TGAHEAD head;
    bf  >> head.No_ID
        >> head.CMap_Type
        >> head.image_type
        >> head.CMap_Spec
        >> head.Ox
        >> head.Oy
        >> head.width
        >> head.height
        >> head.pixel
        >> head.IDesc_Type;

    if(head.pixel!=32)
    {
        istPrint("32bit データしか対応していません。\n");
        return false;
    }

    resize(head.width, head.height);

    for(uint32 i=height()-1; i>=0; --i) {
        if(head.image_type==2) {
            for(uint32 j=0; j<width(); j++) {
                (*this)[i][j] = Read1Pixel(bf);
            }
        }
        else if(head.image_type==10) {
            uint32 loaded = 0;
            while(loaded<width()) {
                uint8 dist = 0;
                bf >> dist;
                if( dist<0x80) {
                    for(int32 j=0; j<dist+1; ++j, ++loaded) {
                        (*this)[i][loaded] = Read1Pixel(bf);
                    }
                }
                else {
                    bRGBA t = Read1Pixel(bf);
                    for(int32 j=0x80; j<dist+1; ++j, ++loaded) {
                        (*this)[i][loaded] = t;
                    }
                }
            }
        }
    }

    return true;
}


class TGACompress
{
public:
    TGACompress() {}

    const std::vector<uint8>& getCompressedData() const { return m_comp_pixel; }

    void compress(const bRGBA *start, int32 width)
    {
        std::vector<bRGBA> same, diff;

        for(int32 i=0; i!=width; ++i, ++start)
        {
            const bRGBA *ip=start; ++ip;
            bRGBA dist=*start;

            if( i+1!=width && dist==*ip && same.size()<0x79 )
            {
                same.push_back(dist);
                if(diff.size()!=0)
                {
                    writeDifferentData(diff);
                }
            }
            else
            {
                if(same.size()>0x00)
                {
                    writeSameData(same);
                }
                else
                {
                    diff.push_back(dist);
                    if(diff.size()==0x79 )
                    {
                        writeDifferentData(diff);
                    }
                }
            }
        }

        if(same.size()!=0x00)
        {
            writeSameData(same);
        }
        else if(diff.size()!=0)
        {
            writeDifferentData(diff);
        }
    }

private:
    void writeSameData(std::vector<bRGBA> &temp_pixel)
    {
        m_comp_pixel.push_back( temp_pixel.size()+0x80 );

        m_comp_pixel.push_back( temp_pixel[0].b );
        m_comp_pixel.push_back( temp_pixel[0].g );
        m_comp_pixel.push_back( temp_pixel[0].r );
        m_comp_pixel.push_back( temp_pixel[0].a );

        temp_pixel.clear();
    }

    void writeDifferentData(std::vector<bRGBA> &temp_pixel)
    {
        m_comp_pixel.push_back( temp_pixel.size()-1 );

        for(uint32 d=0; d<temp_pixel.size(); d++)
        {
            m_comp_pixel.push_back( temp_pixel[d].b );
            m_comp_pixel.push_back( temp_pixel[d].g );
            m_comp_pixel.push_back( temp_pixel[d].r );
            m_comp_pixel.push_back( temp_pixel[d].a );
        }

        temp_pixel.clear();
    }

private:
    std::vector<uint8> m_comp_pixel;
};

bool Image::saveTGA(std::streambuf &f, const Image::IOConfig &conf) const
{
    ist::biostream bf(f);

    TGAHEAD head;
    head.width = width();
    head.height = height();
    head.image_type = 10;

    bf  << head.No_ID
        << head.CMap_Type
        << head.image_type
        << head.CMap_Spec
        << head.Ox
        << head.Oy
        << head.width
        << head.height
        << head.pixel
        << head.IDesc_Type;

    {
        TGACompress comp;
        for(int32 i=height()-1; i>=0; --i)
        {
            comp.compress((*this)[i], width());
        }
        const std::vector<uint8>& data = comp.getCompressedData();
        bf.write(&data[0], data.size());
    }

    return true;
}



// PNG

#ifdef __ist_with_png__
namespace
{
    void png_streambuf_read(png_structp png_ptr, png_bytep data, png_size_t length)
    {
        std::streambuf* f = reinterpret_cast<std::streambuf*>(png_get_io_ptr(png_ptr));
        f->sgetn((char*)data, length);
    }

    void png_streambuf_write(png_structp png_ptr, png_bytep data, png_size_t length)
    {
        std::streambuf* f = reinterpret_cast<std::streambuf*>(png_get_io_ptr(png_ptr));
        f->sputn((char*)data, length);
    }

    void png_streambuf_flush(png_structp png_ptr)
    {
    }
} // namespace
#endif // __ist_with_png__

bool Image::loadPNG(std::streambuf& f, const IOConfig& conf)
{
#ifdef __ist_with_png__
    png_structp png_ptr = ::png_create_read_struct(PNG_LIBPNG_VER_STRING, 0, 0, 0);
    if(png_ptr==0)
    {
        istPrint("失敗: png_create_read_struct() が null を返しました。");
        return false;
    }

    png_infop info_ptr = ::png_create_info_struct(png_ptr);
    if(info_ptr==0)
    {
        ::png_destroy_read_struct(&png_ptr, png_infopp_NULL, png_infopp_NULL);
        istPrint("失敗: png_create_info_struct() が null を返しました。");
        return false;
    }

    ::png_set_read_fn(png_ptr, &f, png_streambuf_read);

    png_uint_32 w, h;
    int32 bit_depth, color_type, interlace_type;

    ::png_read_info(png_ptr, info_ptr);
    ::png_get_IHDR(png_ptr, info_ptr, &w, &h, &bit_depth, &color_type, &interlace_type, int_p_NULL, int_p_NULL);

    resize(w, h);

    ::png_set_strip_16(png_ptr);
    ::png_set_packing(png_ptr);
    if(color_type==PNG_COLOR_TYPE_PALETTE)
    {
        ::png_set_palette_to_rgb(png_ptr);
    }
    if(color_type == PNG_COLOR_TYPE_GRAY && bit_depth<8)
    {
        ::png_set_gray_1_2_4_to_8(png_ptr);
    }
    if(::png_get_valid(png_ptr, info_ptr, PNG_INFO_tRNS))
    {
        ::png_set_tRNS_to_alpha(png_ptr);
    }
    ::png_read_update_info(png_ptr, info_ptr);


    // 読み込み
    std::vector<png_bytep> row_pointers(height());
    for(uint32 row=0; row<height(); ++row) {
        row_pointers[row] = (png_bytep)png_malloc(png_ptr, png_get_rowbytes(png_ptr, info_ptr));
    }
    png_read_image(png_ptr, &row_pointers[0]);

    for(uint32 i=0; i<height(); ++i) {
        for(uint32 j=0; j<width(); ++j) {
            bRGBA& c = (*this)[i][j];
            if(color_type==PNG_COLOR_TYPE_RGB_ALPHA) {
                c.r = row_pointers[i][j*4+0];
                c.g = row_pointers[i][j*4+1];
                c.b = row_pointers[i][j*4+2];
                c.a = row_pointers[i][j*4+3];
            }
            else if(color_type==PNG_COLOR_TYPE_RGB) {
                c.r = row_pointers[i][j*3+0];
                c.g = row_pointers[i][j*3+1];
                c.b = row_pointers[i][j*3+2];
                c.a = 255;
            }
        }
    }
    for(uint32 row=0; row<height(); ++row) {
        png_free(png_ptr, row_pointers[row]);
    }
    png_read_end(png_ptr, info_ptr);


    png_destroy_read_struct(&png_ptr, &info_ptr, png_infopp_NULL);
    return true;

#else
    istPrint("失敗: png 使用を無効化した設定でビルドされています。");
    return false;
#endif // __ist_with_png__
}

bool Image::savePNG(std::streambuf& f, const Image::IOConfig& conf) const
{
#ifdef __ist_with_png__
    png_structp png_ptr = ::png_create_write_struct(PNG_LIBPNG_VER_STRING, 0, 0, 0);
    if(png_ptr==0)
    {
        istPrint("失敗: png_create_write_struct() が null を返しました。");
        return false;
    }

    png_infop info_ptr = ::png_create_info_struct(png_ptr);
    if(info_ptr==0)
    {
        ::png_destroy_write_struct(&png_ptr,  png_infopp_NULL);
        istPrint("失敗: png_create_info_struct() が null を返しました。");
        return false;
    }

    ::png_set_write_fn(png_ptr, &f, png_streambuf_write, png_streambuf_flush);
    ::png_set_IHDR(png_ptr, info_ptr, width(), height(), 8,
        PNG_COLOR_TYPE_RGB_ALPHA, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);
    ::png_write_info(png_ptr, info_ptr);

    Image tmp(*this);
    std::vector<png_bytep> row_pointers(height());
    for(uint32 i=0; i<height(); ++i)
    {
        row_pointers[i] = tmp[i][0].v;
    }

    ::png_write_image(png_ptr, &row_pointers[0]);

    ::png_write_end(png_ptr, info_ptr);
    ::png_destroy_write_struct(&png_ptr, &info_ptr);
    return true;

#else
    istPrint("失敗: png 使用を無効化した設定でビルドされています。");
    return false;
#endif // __ist_with_png__
}


// JPG

#ifdef __ist_with_jpeg__
namespace
{

    struct my_error_mgr
    {
        struct jpeg_error_mgr pub;
        jmp_buf setjmp_buffer;
    };
    typedef struct my_error_mgr * my_error_ptr;

    void my_error_exit(j_common_ptr cinfo)
    {
        my_error_ptr myerr = (my_error_ptr) cinfo->err;
        (*cinfo->err->output_message)(cinfo);
        longjmp(myerr->setjmp_buffer, 1);
    }


    typedef struct {
        struct jpeg_source_mgr pub;	/* public fields */

        std::streambuf * infile;		/* source stream */
        JOCTET * buffer;		/* start of buffer */
        boolean start_of_file;	/* have we gotten any data yet? */
    } my_source_mgr;

    typedef my_source_mgr * my_src_ptr;

#define INPUT_BUF_SIZE  4096	/* choose an efficiently fread'able size */


    typedef struct {
        struct jpeg_destination_mgr pub; /* public fields */

        std::streambuf * outfile;		/* target stream */
        JOCTET * buffer;		/* start of buffer */
    } my_destination_mgr;

    typedef my_destination_mgr * my_dest_ptr;

#define OUTPUT_BUF_SIZE  4096	/* choose an efficiently fwrite'able size */

#define SIZEOF(object)	((size_t) sizeof(object))


    void init_streambuf_source (j_decompress_ptr cinfo)
    {
        my_src_ptr src = (my_src_ptr) cinfo->src;
        src->start_of_file = TRUE;
    }

    boolean fill_streambuf_input_buffer (j_decompress_ptr cinfo)
    {
        my_src_ptr src = (my_src_ptr) cinfo->src;
        size_t nbytes;

        nbytes = src->infile->sgetn((char*)src->buffer, INPUT_BUF_SIZE);

        if (nbytes <= 0) {
            if (src->start_of_file)	/* Treat empty input file as fatal error */
                ERREXIT(cinfo, JERR_INPUT_EMPTY);
            WARNMS(cinfo, JWRN_JPEG_EOF);
            /* Insert a fake EOI marker */
            src->buffer[0] = (JOCTET) 0xFF;
            src->buffer[1] = (JOCTET) JPEG_EOI;
            nbytes = 2;
        }

        src->pub.next_input_byte = src->buffer;
        src->pub.bytes_in_buffer = nbytes;
        src->start_of_file = FALSE;

        return TRUE;
    }

    void skip_input_data (j_decompress_ptr cinfo, long num_bytes)
    {
        struct jpeg_source_mgr * src = cinfo->src;

        if (num_bytes > 0) {
            while (num_bytes > (long) src->bytes_in_buffer) {
                num_bytes -= (long) src->bytes_in_buffer;
                (void) (*src->fill_input_buffer) (cinfo);
            }
            src->next_input_byte += (size_t) num_bytes;
            src->bytes_in_buffer -= (size_t) num_bytes;
        }
    }

    void term_source (j_decompress_ptr cinfo)
    {
    }

    void jpeg_streambuf_src (j_decompress_ptr cinfo, std::streambuf& streambuf)
    {
        my_src_ptr src;

        if (cinfo->src == NULL) {	/* first time for this JPEG object? */
            cinfo->src = (struct jpeg_source_mgr *)
                (*cinfo->mem->alloc_small) ((j_common_ptr) cinfo, JPOOL_PERMANENT, SIZEOF(my_source_mgr));
            src = (my_src_ptr) cinfo->src;
            src->buffer = (JOCTET *)
                (*cinfo->mem->alloc_small) ((j_common_ptr) cinfo, JPOOL_PERMANENT, INPUT_BUF_SIZE * SIZEOF(JOCTET));
        }

        src = (my_src_ptr) cinfo->src;
        src->pub.init_source = init_streambuf_source;
        src->pub.fill_input_buffer = fill_streambuf_input_buffer;
        src->pub.skip_input_data = skip_input_data;
        src->pub.resync_to_restart = jpeg_resync_to_restart; /* use default method */
        src->pub.term_source = term_source;
        src->infile = &streambuf;
        src->pub.bytes_in_buffer = 0; /* forces fill_input_buffer on first read */
        src->pub.next_input_byte = NULL; /* until buffer loaded */
    }


    void init_streambuf_destination (j_compress_ptr cinfo)
    {
        my_dest_ptr dest = (my_dest_ptr) cinfo->dest;

        /* Allocate the output buffer --- it will be released when done with image */
        dest->buffer = (JOCTET *)
            (*cinfo->mem->alloc_small) ((j_common_ptr) cinfo, JPOOL_IMAGE,
            OUTPUT_BUF_SIZE * SIZEOF(JOCTET));

        dest->pub.next_output_byte = dest->buffer;
        dest->pub.free_in_buffer = OUTPUT_BUF_SIZE;
    }


    boolean empty_streambuf_output_buffer (j_compress_ptr cinfo)
    {
        my_dest_ptr dest = (my_dest_ptr) cinfo->dest;

        if (dest->outfile->sputn((char*)dest->buffer, OUTPUT_BUF_SIZE) != (size_t) OUTPUT_BUF_SIZE)
            ERREXIT(cinfo, JERR_FILE_WRITE);

        dest->pub.next_output_byte = dest->buffer;
        dest->pub.free_in_buffer = OUTPUT_BUF_SIZE;

        return TRUE;
    }

    void term_destination (j_compress_ptr cinfo)
    {
        my_dest_ptr dest = (my_dest_ptr) cinfo->dest;
        size_t datacount = OUTPUT_BUF_SIZE - dest->pub.free_in_buffer;

        /* Write any data remaining in the buffer */
        if (datacount > 0) {
            if (dest->outfile->sputn((char*)dest->buffer, datacount) != datacount)
                ERREXIT(cinfo, JERR_FILE_WRITE);
        }
    }

    void jpeg_streambuf_dest (j_compress_ptr cinfo, std::streambuf& outfile)
    {
        my_dest_ptr dest;

        if (cinfo->dest == NULL) {	/* first time for this JPEG object? */
            cinfo->dest = (struct jpeg_destination_mgr *)
                (*cinfo->mem->alloc_small) ((j_common_ptr) cinfo, JPOOL_PERMANENT,
                SIZEOF(my_destination_mgr));
        }

        dest = (my_dest_ptr) cinfo->dest;
        dest->pub.init_destination = init_streambuf_destination;
        dest->pub.empty_output_buffer = empty_streambuf_output_buffer;
        dest->pub.term_destination = term_destination;
        dest->outfile = &outfile;
    }

} // namespace
#endif // __ist_with_jpeg__


bool Image::loadJPG(std::streambuf& f, const IOConfig& conf)
{
    clear();

#ifdef __ist_with_jpeg__
    jpeg_decompress_struct cinfo;
    my_error_mgr jerr;
    JSAMPARRAY buffer;
    int32 row_stride;

    cinfo.err = jpeg_std_error(&jerr.pub);
    jerr.pub.error_exit = my_error_exit;
    if(setjmp(jerr.setjmp_buffer))
    {
        jpeg_destroy_decompress(&cinfo);
        return false;
    }

    jpeg_create_decompress(&cinfo);
    jpeg_streambuf_src(&cinfo, f);
    jpeg_read_header(&cinfo, TRUE);
    jpeg_start_decompress(&cinfo);
    row_stride = cinfo.output_width * cinfo.output_components;
    buffer = (*cinfo.mem->alloc_sarray)((j_common_ptr) &cinfo, JPOOL_IMAGE, row_stride, 1);

    resize(cinfo.image_width, cinfo.image_height);
    uint32 pix_count = 0;
    while (cinfo.output_scanline < cinfo.output_height)
    {
        jpeg_read_scanlines(&cinfo, buffer, 1);
        for(uint32 i=0; i<row_stride/3; ++i)
        {
            bRGBA col(buffer[0][i*3+0], buffer[0][i*3+1], buffer[0][i*3+2], 255);
            at(pix_count) = col;
            ++pix_count;
        }
    }

    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    return true;

#else
    istPrint("失敗: jpg 使用を無効化した設定でビルドされています。");
    return false;
#endif // __ist_with_jpeg__
}

bool Image::saveJPG(std::streambuf& f, const IOConfig& conf) const
{
#ifdef __ist_with_jpeg__
    jpeg_compress_struct cinfo;
    jpeg_error_mgr jerr;
    JSAMPROW row_pointer[1];
    int32 row_stride;

    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);

    jpeg_streambuf_dest(&cinfo, f);

    cinfo.image_width = width();
    cinfo.image_height = height();
    cinfo.input_components = 3;
    cinfo.in_color_space = JCS_RGB;

    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, conf.getJpgQuality(), TRUE);

    jpeg_start_compress(&cinfo, TRUE);

    row_stride = cinfo.image_width*3;

    uint8 *buf = new uint8[width()*height()*3];
    for(uint32 i=0; i<width()*height(); ++i)
    {
        buf[i*3+0] = at(i).r;
        buf[i*3+1] = at(i).g;
        buf[i*3+2] = at(i).b;
    }
    while (cinfo.next_scanline < cinfo.image_height)
    {
        row_pointer[0] = &buf[cinfo.next_scanline * row_stride];
        jpeg_write_scanlines(&cinfo, row_pointer, 1);
    }
    delete[] buf;

    jpeg_finish_compress(&cinfo);
    jpeg_destroy_compress(&cinfo);
    return true;

#else // __ist_with_jpeg__
    istPrint("失敗: jpg 使用を無効化した設定でビルドされています。");
    return false;
#endif // __ist_with_jpeg__
}



} // namespace ist
