#include "istPCH.h"
#include "ist/GraphicsCommon/Image.h"
#ifdef ist_with_gli
#include "gli/gli.hpp"
#include "gli/gtx/loader.hpp"
#endif // __ist_with_gli__

namespace ist {


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


Image::FileType GetFileTypeByFileHeader(IBinaryStream &f)
{
    char m[4];
    f >> m; f.setReadPos(0);
    if(m[0]=='B' && m[1]=='M') { return Image::FileType_BMP; }
    if(m[1]=='P' && m[2]=='N' && m[3]=='G') { return Image::FileType_PNG; }
    if(m[0]=='D' && m[1]=='D' && m[2]=='S') { return Image::FileType_DDS; }
    if(m[0]==0xff && m[1]==0xd8) { return Image::FileType_JPG; }
    {
        // tga は magic code がないので、それっぽい値が入ってるかで判断
        TGAHEAD tga;
        f.read(&tga, sizeof(tga)); f.setReadPos(0);
        if( (tga.image_type==2 || tga.image_type==10) && tga.Ox==0 && tga.Oy==0 && (tga.pixel==32 || tga.pixel==24)) {
            return Image::FileType_TGA;
        }
    }
    return Image::FileType_Unknown;
}

Image::FileType GetFileTypeByExtention(const char *path)
{
    uint32 len = strlen(path);
    if(len<5) { return Image::FileType_Unknown; }

    if(strncmp(&path[len-3], "bmp", 3)==0) { return Image::FileType_BMP; }
    if(strncmp(&path[len-3], "tga", 3)==0) { return Image::FileType_TGA; }
    if(strncmp(&path[len-3], "png", 3)==0) { return Image::FileType_PNG; }
    if(strncmp(&path[len-3], "jpg", 3)==0) { return Image::FileType_JPG; }
    if(strncmp(&path[len-3], "dds", 3)==0) { return Image::FileType_DDS; }
    return Image::FileType_Unknown;
}


bool Image::load(const char *path, const IOConfig &conf)
{
    FileStream f(path, "rb");
    if(!f.isOpened()) { return false; }

    IOConfig c = conf;
    if(c.getFileType()==FileType_Auto) {
        c.setFileType( GetFileTypeByExtention(path) );
    }
    return load(f, c);
}

bool Image::load(IBinaryStream &f, const IOConfig &conf)
{
    clear();

    FileType ft = conf.getFileType();
    if(ft==FileType_Auto) {
        ft = GetFileTypeByFileHeader(f);
    }
    switch(ft)
    {
    case FileType_BMP: return loadBMP(f, conf);
    case FileType_TGA: return loadTGA(f, conf);
    case FileType_PNG: return loadPNG(f, conf);
    case FileType_JPG: return loadJPG(f, conf);
    case FileType_DDS: return loadDDS(f, conf);
    }
    istPrint("認識できないフォーマットが指定されました。\n");
    return false;
}


bool Image::save(const char *path, const IOConfig &conf) const
{
    FileStream f(path, "wb");
    if(!f.isOpened()) { return false; }
    IOConfig c = conf;
    if(c.getFileType()==FileType_Auto) {
        c.setFileType(GetFileTypeByExtention(path));
    }
    return save(f, c);
}

bool Image::save(IBinaryStream &f, const IOConfig &conf) const
{
    switch(conf.getFileType())
    {
    case FileType_BMP: return saveBMP(f, conf);
    case FileType_TGA: return saveTGA(f, conf);
    case FileType_PNG: return savePNG(f, conf);
    case FileType_JPG: return saveJPG(f, conf);
    case FileType_DDS: return saveDDS(f, conf);
    }
    istPrint(L"認識できないフォーマットが指定されました。\n");
    return false;
}







static RGBA_8U Read1Pixel(IBinaryStream &bf)
{
    RGBA_8U t;
    bf >> t.b >> t.g >> t.r >> t.a;
    return t;
}



// BMP

bool Image::loadBMP(IBinaryStream &bf, const IOConfig &conf)
{
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

    if(infohead.bits!=24 && infohead.bits!=32) {
        istPrint(L"bmp は現在 24bit か 32bit しか対応していません。\n");
        return false;
    }

    resize<RGBA_8U>(infohead.width, infohead.height);

    if(infohead.bits==24) {
        for(int32 yi=(int32)height()-1; yi>=0; --yi) {
            for(int32 xi=0; xi<(int32)width(); ++xi) {
                RGBA_8U& c = get<RGBA_8U>(yi, xi);
                bf >> c.b >> c.g >> c.r;
                c.a = 255;
            }
        }
    }
    else if(infohead.bits==32) {
        for(int32 yi=(int32)height()-1; yi>=0; --yi) {
            for(int32 xi=0; xi<(int32)width(); ++xi) {
                RGBA_8U& c = get<RGBA_8U>(yi, xi);
                bf >> c.b >> c.g >> c.r >> c.a;
            }
        }
    }

    return true;
}


bool Image::saveBMP(IBinaryStream &bf, const IOConfig &conf) const
{
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

    for(int32 yi=(int32)height()-1; yi>=0; --yi) {
        for(int32 xi=0; xi<(int32)width(); ++xi) {
            const RGBA_8U& c = get<RGBA_8U>(yi, xi);
            bf << c.b << c.g << c.r;
        }
    }
    return true;
}





// TGA

bool Image::loadTGA(IBinaryStream &bf, const IOConfig &conf)
{
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

    resize<RGBA_8U>(head.width, head.height);

    for(int32 yi=(int32)height()-1; yi>=0; --yi) {
        if(head.image_type==2) {
            for(int32 xi=0; xi<(int32)width(); xi++) {
                get<RGBA_8U>(yi, xi) = Read1Pixel(bf);
            }
        }
        else if(head.image_type==10) {
            uint32 loaded = 0;
            while(loaded<width()) {
                uint8 dist = 0;
                bf >> dist;
                if( dist<0x80) {
                    for(int32 xi=0; xi<dist+1; ++xi, ++loaded) {
                        get<RGBA_8U>(yi, loaded) = Read1Pixel(bf);
                    }
                }
                else {
                    RGBA_8U t = Read1Pixel(bf);
                    for(int32 xi=0x80; xi<dist+1; ++xi, ++loaded) {
                        get<RGBA_8U>(yi, loaded) = t;
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

    const stl::vector<uint8>& getCompressedData() const { return m_comp_pixel; }

    void compress(const RGBA_8U *start, int32 width)
    {
        stl::vector<RGBA_8U> same, diff;

        for(int32 i=0; i!=width; ++i, ++start)
        {
            const RGBA_8U *ip=start; ++ip;
            RGBA_8U dist=*start;

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
    void writeSameData(stl::vector<RGBA_8U> &temp_pixel)
    {
        m_comp_pixel.push_back( temp_pixel.size()+0x80 );

        m_comp_pixel.push_back( temp_pixel[0].b );
        m_comp_pixel.push_back( temp_pixel[0].g );
        m_comp_pixel.push_back( temp_pixel[0].r );
        m_comp_pixel.push_back( temp_pixel[0].a );

        temp_pixel.clear();
    }

    void writeDifferentData(stl::vector<RGBA_8U> &temp_pixel)
    {
        m_comp_pixel.push_back( temp_pixel.size()-1 );

        for(int32 d=0; d<(int32)temp_pixel.size(); d++)
        {
            m_comp_pixel.push_back( temp_pixel[d].b );
            m_comp_pixel.push_back( temp_pixel[d].g );
            m_comp_pixel.push_back( temp_pixel[d].r );
            m_comp_pixel.push_back( temp_pixel[d].a );
        }

        temp_pixel.clear();
    }

private:
    stl::vector<uint8> m_comp_pixel;
};

bool Image::saveTGA(IBinaryStream &bf, const Image::IOConfig &conf) const
{
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
        for(int32 yi=(int32)height()-1; yi>=0; --yi)
        {
            comp.compress(&get<RGBA_8U>(yi, 0), width());
        }
        const stl::vector<uint8>& data = comp.getCompressedData();
        bf.write(&data[0], data.size());
    }

    return true;
}



// PNG

#ifdef ist_with_png
namespace
{
    void png_streambuf_read(png_structp png_ptr, png_bytep data, png_size_t length)
    {
        IBinaryStream *f = reinterpret_cast<IBinaryStream*>(png_get_io_ptr(png_ptr));
        f->read(data, length);
    }

    void png_streambuf_write(png_structp png_ptr, png_bytep data, png_size_t length)
    {
        IBinaryStream *f = reinterpret_cast<IBinaryStream*>(png_get_io_ptr(png_ptr));
        f->write(data, length);
    }

    void png_streambuf_flush(png_structp png_ptr)
    {
    }
} // namespace
#endif // ist_with_png

bool Image::loadPNG(IBinaryStream &f, const IOConfig &conf)
{
#ifdef ist_with_png
    png_structp png_ptr = ::png_create_read_struct(PNG_LIBPNG_VER_STRING, 0, 0, 0);
    if(png_ptr==0)
    {
        istPrint("失敗: png_create_read_struct() が null を返しました。\n");
        return false;
    }

    png_infop info_ptr = ::png_create_info_struct(png_ptr);
    if(info_ptr==0)
    {
        ::png_destroy_read_struct(&png_ptr, (png_infopp)NULL, (png_infopp)NULL);
        istPrint("失敗: png_create_info_struct() が null を返しました。\n");
        return false;
    }

    ::png_set_read_fn(png_ptr, &f, png_streambuf_read);

    png_uint_32 w, h;
    int32 bit_depth, color_type, interlace_type;

    ::png_read_info(png_ptr, info_ptr);
    ::png_get_IHDR(png_ptr, info_ptr, &w, &h, &bit_depth, &color_type, &interlace_type, NULL, NULL);

    resize<RGBA_8U>(w, h);

    ::png_set_strip_16(png_ptr);
    ::png_set_packing(png_ptr);
    if(color_type==PNG_COLOR_TYPE_PALETTE)
    {
        ::png_set_palette_to_rgb(png_ptr);
    }
    if(color_type == PNG_COLOR_TYPE_GRAY && bit_depth<8)
    {
        ::png_set_expand_gray_1_2_4_to_8(png_ptr);
    }
    if(::png_get_valid(png_ptr, info_ptr, PNG_INFO_tRNS))
    {
        ::png_set_tRNS_to_alpha(png_ptr);
    }
    ::png_read_update_info(png_ptr, info_ptr);


    // 読み込み
    stl::vector<png_bytep> row_pointers(height());
    for(int32 row=0; row<(int32)height(); ++row) {
        row_pointers[row] = (png_bytep)png_malloc(png_ptr, png_get_rowbytes(png_ptr, info_ptr));
    }
    png_read_image(png_ptr, &row_pointers[0]);

    for(int32 yi=0; yi<(int32)height(); ++yi) {
        for(int32 xi=0; xi<(int32)width(); ++xi) {
            RGBA_8U& c = get<RGBA_8U>(yi, xi);
            if(color_type==PNG_COLOR_TYPE_RGB_ALPHA) {
                c.r = row_pointers[yi][xi*4+0];
                c.g = row_pointers[yi][xi*4+1];
                c.b = row_pointers[yi][xi*4+2];
                c.a = row_pointers[yi][xi*4+3];
            }
            else if(color_type==PNG_COLOR_TYPE_RGB) {
                c.r = row_pointers[yi][xi*3+0];
                c.g = row_pointers[yi][xi*3+1];
                c.b = row_pointers[yi][xi*3+2];
                c.a = 255;
            }
        }
    }
    for(int32 row=0; row<(int32)height(); ++row) {
        png_free(png_ptr, row_pointers[row]);
    }
    png_read_end(png_ptr, info_ptr);


    png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
    return true;

#else
    istPrint("失敗: png 使用を無効化した設定でビルドされています。\n");
    return false;
#endif // ist_with_png
}

bool Image::savePNG(IBinaryStream &f, const Image::IOConfig &conf) const
{
#ifdef ist_with_png
    png_structp png_ptr = ::png_create_write_struct(PNG_LIBPNG_VER_STRING, 0, 0, 0);
    if(png_ptr==0)
    {
        istPrint("失敗: png_create_write_struct() が null を返しました。");
        return false;
    }

    png_infop info_ptr = ::png_create_info_struct(png_ptr);
    if(info_ptr==0)
    {
        ::png_destroy_write_struct(&png_ptr,  NULL);
        istPrint("失敗: png_create_info_struct() が null を返しました。");
        return false;
    }

    ::png_set_write_fn(png_ptr, &f, png_streambuf_write, png_streambuf_flush);
    ::png_set_IHDR(png_ptr, info_ptr, width(), height(), 8,
        PNG_COLOR_TYPE_RGB_ALPHA, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);
    ::png_write_info(png_ptr, info_ptr);

    Image tmp(*this);
    stl::vector<png_bytep> row_pointers(height());
    for(int32 yi=0; yi<(int32)height(); ++yi)
    {
        row_pointers[yi] = tmp.get<RGBA_8U>(yi, 0).v;
    }

    ::png_write_image(png_ptr, &row_pointers[0]);

    ::png_write_end(png_ptr, info_ptr);
    ::png_destroy_write_struct(&png_ptr, &info_ptr);
    return true;

#else
    istPrint("失敗: png 使用を無効化した設定でビルドされています。");
    return false;
#endif // ist_with_png
}


// JPG

#ifdef ist_with_jpeg
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

        IBinaryStream *infile;		/* source stream */
        JOCTET * buffer;		/* start of buffer */
        boolean start_of_file;	/* have we gotten any data yet? */
    } my_source_mgr;

    typedef my_source_mgr * my_src_ptr;

#define INPUT_BUF_SIZE  4096	/* choose an efficiently fread'able size */


    typedef struct {
        struct jpeg_destination_mgr pub; /* public fields */

        IBinaryStream * outfile;		/* target stream */
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

        nbytes = src->infile->read(src->buffer, INPUT_BUF_SIZE);

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

    void jpeg_streambuf_src (j_decompress_ptr cinfo, IBinaryStream &streambuf)
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

        if (dest->outfile->write(dest->buffer, OUTPUT_BUF_SIZE) != (size_t) OUTPUT_BUF_SIZE)
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
            if (dest->outfile->write(dest->buffer, datacount) != datacount)
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
#endif // ist_with_jpeg


bool Image::loadJPG(IBinaryStream &f, const IOConfig &conf)
{
    clear();

#ifdef ist_with_jpeg
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
        for(int32 i=0; i<(int32)row_stride/3; ++i)
        {
            RGBA_8U col(buffer[0][i*3+0], buffer[0][i*3+1], buffer[0][i*3+2], 255);
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
#endif // ist_with_jpeg
}

bool Image::saveJPG(IBinaryStream &f, const IOConfig &conf) const
{
#ifdef ist_with_jpeg
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
    for(int32 i=0; i<(int32)width()*(int32)height(); ++i)
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

#else // ist_with_jpeg
    istPrint("失敗: jpg 使用を無効化した設定でビルドされています。");
    return false;
#endif // ist_with_jpeg
}

} // namespace ist


#ifdef ist_with_gli
namespace gli {
namespace gtx {
namespace loader_dds10{
namespace detail {

// gli には std::string を引数にとるやつしかないので、ストリーム版をコピペ改変実装します。参考: loadDDS10()
inline texture2D loadDDS10_ex( ist::IBinaryStream &bin )
{
    loader_dds9::detail::ddsHeader HeaderDesc;
    detail::ddsHeader10 HeaderDesc10;
    char Magic[4]; 

    //* Read magic number and check if valid .dds file 
    bin.read((char*)&Magic, sizeof(Magic));

    assert(strncmp(Magic, "DDS ", 4) == 0);

    // Get the surface descriptor 
    bin.read(&HeaderDesc, sizeof(HeaderDesc));
    if(HeaderDesc.format.flags & loader_dds9::detail::GLI_DDPF_FOURCC && HeaderDesc.format.fourCC == loader_dds9::detail::GLI_FOURCC_DX10)
        bin.read(&HeaderDesc10, sizeof(HeaderDesc10));

    loader_dds9::detail::DDLoader Loader;
    if(HeaderDesc.format.fourCC == loader_dds9::detail::GLI_FOURCC_DX10)
        Loader.Format = detail::format_dds2gli_cast(HeaderDesc10.dxgiFormat);
    else if(HeaderDesc.format.flags & loader_dds9::detail::GLI_DDPF_FOURCC)
        Loader.Format = detail::format_fourcc2gli_cast(HeaderDesc.format.fourCC);
    else
    {
        switch(HeaderDesc.format.bpp)
        {
        case 8:
            Loader.Format = R8U;
            break;
        case 16:
            Loader.Format = RG8U;
            break;
        case 24:
            Loader.Format = RGB8U;
            break;
        case 32:
            Loader.Format = RGBA8U;
            break;
        }
    }
    Loader.BlockSize = size(image(texture2D::dimensions_type(0), Loader.Format), BLOCK_SIZE);
    Loader.BPP = size(image(texture2D::dimensions_type(0), Loader.Format), BIT_PER_PIXEL);

    std::size_t Width = HeaderDesc.width;
    std::size_t Height = HeaderDesc.height;

    gli::format Format = Loader.Format;

    std::streamoff Curr = bin.getReadPos();
    bin.setReadPos(0, ist::IBinaryStream::Seek_End);
    std::streamoff End = bin.getReadPos();
    bin.setReadPos(Curr);

    std::vector<glm::byte> Data(std::size_t(End - Curr), 0);
    std::size_t Offset = 0;

    bin.read(&Data[0], std::streamsize(Data.size()));

    //texture2D Image(glm::min(MipMapCount, Levels));//SurfaceDesc.mipMapLevels);
    std::size_t MipMapCount = (HeaderDesc.flags & loader_dds9::detail::GLI_DDSD_MIPMAPCOUNT) ? HeaderDesc.mipMapLevels : 1;
    //if(Loader.Format == DXT1 || Loader.Format == DXT3 || Loader.Format == DXT5) 
    //	MipMapCount -= 2;
    texture2D Image(MipMapCount);
    for(std::size_t Level = 0; Level < Image.levels() && (Width || Height); ++Level)
    {
        Width = glm::max(std::size_t(Width), std::size_t(1));
        Height = glm::max(std::size_t(Height), std::size_t(1));

        std::size_t MipmapSize = 0;
        if((Loader.BlockSize << 3) > Loader.BPP)
            MipmapSize = ((Width + 3) >> 2) * ((Height + 3) >> 2) * Loader.BlockSize;
        else
            MipmapSize = Width * Height * Loader.BlockSize;
        std::vector<glm::byte> MipmapData(MipmapSize, 0);

        memcpy(&MipmapData[0], &Data[0] + Offset, MipmapSize);

        texture2D::dimensions_type Dimensions(Width, Height);
        Image[Level] = texture2D::image(Dimensions, Format, MipmapData);

        Offset += MipmapSize;
        Width >>= 1;
        Height >>= 1;
    }

    return Image;
}


} // detail
} // loader_dds10
} // namespace gtx
} // namespace gli
#endif // __ist_with_gli__

namespace ist {

bool Image::loadDDS( IBinaryStream &f, const IOConfig &conf )
{
#ifdef ist_with_gli
    gli::texture2D tex = gli::gtx::loader_dds10::detail::loadDDS10_ex(f);
    ivec2 dim = ivec2(tex[0].dimensions().x, tex[0].dimensions().y);
    switch(tex.format()) {
    case gli::R8U:      setup(IF_R8U, dim.x, dim.y, tex[0].capacity()); break;
    case gli::R8I:      setup(IF_R8I, dim.x, dim.y, tex[0].capacity()); break;
    case gli::R32F:     setup(IF_R32F, dim.x, dim.y, tex[0].capacity()); break;
    case gli::RG8U:     setup(IF_RG8U, dim.x, dim.y, tex[0].capacity()); break;
    case gli::RG8I:     setup(IF_RG8I, dim.x, dim.y, tex[0].capacity()); break;
    case gli::RG32F:    setup(IF_RG32F, dim.x, dim.y, tex[0].capacity()); break;
    case gli::RGB8U:    setup(IF_RGB8U, dim.x, dim.y, tex[0].capacity()); break;
    case gli::RGB8I:    setup(IF_RGB8I, dim.x, dim.y, tex[0].capacity()); break;
    case gli::RGB32F:   setup(IF_RGB32F, dim.x, dim.y, tex[0].capacity()); break;
    case gli::RGBA8U:   setup(IF_RGBA8U, dim.x, dim.y, tex[0].capacity()); break;
    case gli::RGBA8I:   setup(IF_RGBA8I, dim.x, dim.y, tex[0].capacity()); break;
    case gli::RGBA32F:  setup(IF_RGBA32F, dim.x, dim.y, tex[0].capacity()); break;

    case gli::DXT1:     setup(IF_RGBA_DXT1, dim.x, dim.y, tex[0].capacity()); break;
    case gli::DXT3:     setup(IF_RGBA_DXT3, dim.x, dim.y, tex[0].capacity()); break;
    case gli::DXT5:     setup(IF_RGBA_DXT5, dim.x, dim.y, tex[0].capacity()); break;

    default: return false;
    }
    memcpy(data(), tex[0].data(), tex[0].capacity());
    return true;
#endif // __ist_with_gli__
    return false;
}

bool Image::saveDDS( IBinaryStream &f, const IOConfig &conf ) const
{
    istPrint("未実装");
    return false;
}


} // namespace ist
