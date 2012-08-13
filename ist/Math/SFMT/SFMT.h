/** 
 * @file SFMT.h 
 *
 * @brief SIMD oriented Fast Mersenne Twister(SFMT) pseudorandom
 * number generator
 *
 * @author Mutsuo Saito (Hiroshima University)
 * @author Makoto Matsumoto (Hiroshima University)
 *
 * Copyright (C) 2006, 2007 Mutsuo Saito, Makoto Matsumoto and Hiroshima
 * University. All rights reserved.
 *
 * The new BSD License is applied to this software.
 * see LICENSE.txt
 *
 * @note We assume that your system has inttypes.h.  If your system
 * doesn't have inttypes.h, you have to typedef uint32_t and uint64_t,
 * and you have to define PRIu64 and PRIx64 in this file as follows:
 * @verbatim
 typedef unsigned int uint32_t
 typedef unsigned long long uint64_t  
 #define PRIu64 "llu"
 #define PRIx64 "llx"
@endverbatim
 * uint32_t must be exactly 32-bit unsigned integer type (no more, no
 * less), and uint64_t must be exactly 64-bit unsigned integer type.
 * PRIu64 and PRIx64 are used for printf function to print 64-bit
 * unsigned int and 64-bit unsigned int in hexadecimal format.
 */

#define HAVE_SSE2

#ifndef SFMT_H
#define SFMT_H

#include <stdio.h>

#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L)
  #include <inttypes.h>
#elif defined(_MSC_VER) || defined(__BORLANDC__)
  typedef unsigned int uint32_t;
  typedef unsigned __int64 uint64_t;
  #define inline __inline
#else
  #include <inttypes.h>
  #if defined(__GNUC__)
    #define inline __inline__
  #endif
#endif

#ifndef PRIu64
  #if defined(_MSC_VER) || defined(__BORLANDC__)
    #define PRIu64 "I64u"
    #define PRIx64 "I64x"
  #else
    #define PRIu64 "llu"
    #define PRIx64 "llx"
  #endif
#endif

#if defined(__GNUC__)
#define ALWAYSINLINE __attribute__((always_inline))
#else
#define ALWAYSINLINE
#endif

#if defined(_MSC_VER)
  #if _MSC_VER >= 1200
    #define PRE_ALWAYS __forceinline
  #else
    #define PRE_ALWAYS inline
  #endif
#else
  #define PRE_ALWAYS inline
#endif




#if defined(__BIG_ENDIAN__) && !defined(__amd64) && !defined(BIG_ENDIAN64)
#define BIG_ENDIAN64 1
#endif
#if defined(HAVE_ALTIVEC) && !defined(BIG_ENDIAN64)
#define BIG_ENDIAN64 1
#endif
#if defined(ONLY64) && !defined(BIG_ENDIAN64)
  #if defined(__GNUC__)
    #error "-DONLY64 must be specified with -DBIG_ENDIAN64"
  #endif
#undef ONLY64
#endif
/*------------------------------------------------------
  128-bit SIMD data type for Altivec, SSE2 or standard C
  ------------------------------------------------------*/
#if defined(HAVE_ALTIVEC)
  #if !defined(__APPLE__)
    #include <altivec.h>
  #endif
/** 128-bit data structure */
union W128_T {
    vector unsigned int s;
    uint32_t u[4];
};
/** 128-bit data type */
typedef union W128_T w128_t;

#elif defined(HAVE_SSE2)
  #include <emmintrin.h>

/** 128-bit data structure */
union W128_T {
    __m128i si;
    uint32_t u[4];
};
/** 128-bit data type */
typedef union W128_T w128_t;

#else

/** 128-bit data structure */
struct W128_T {
    uint32_t u[4];
};
/** 128-bit data type */
typedef struct W128_T w128_t;

#endif


#include "SFMT-params.h"

namespace ist {

class __declspec(align(16)) SFMT
{
private:
    /*--------------------------------------
      FILE GLOBAL VARIABLES
      internal state, index counter and flag 
      --------------------------------------*/
    /** the 128-bit internal state array */
    w128_t m_sfmt[N];
    /** the 32bit integer pointer to the 128-bit internal state array */
    uint32_t *m_psfmt32;
    #if !defined(BIG_ENDIAN64) || defined(ONLY64)
    /** the 64bit integer pointer to the 128-bit internal state array */
    uint64_t *m_psfmt64;
    #endif
    /** index counter to the 32-bit internal state array */
    int m_idx;
    /** a flag: it is 0 if and only if the internal state is not yet
     * initialized. */
    int m_initialized;
    /** a parity check vector which certificate the period of 2^{MEXP} */
    uint32_t m_parity[4];

    uint32_t m_seed;


private:
    void gen_rand_all(void);
    void gen_rand_array(w128_t *array, int size);
    void period_certification(void);

    uint32_t gen_rand32(void);
    uint64_t gen_rand64(void);
    void fill_array32(uint32_t *array, int size);
    void fill_array64(uint64_t *array, int size);
    void init_gen_rand(uint32_t seed);
    void init_by_array(uint32_t *init_key, int key_length);
    const char *get_idstring(void);
    int get_min_array_size32(void);
    int get_min_array_size64(void);

    /* These real versions are due to Isaku Wada */
    /** generates a random number on [0,1]-real-interval */
    inline double to_real1(uint32_t v)
    {
        return v * (1.0/4294967295.0); 
        /* divided by 2^32-1 */ 
    }

    /** generates a random number on [0,1]-real-interval */
    inline double genrand_real1(void)
    {
        return to_real1(gen_rand32());
    }

    /** generates a random number on [0,1)-real-interval */
    inline double to_real2(uint32_t v)
    {
        return v * (1.0/4294967296.0); 
        /* divided by 2^32 */
    }

    /** generates a random number on [0,1)-real-interval */
    inline double genrand_real2(void)
    {
        return to_real2(gen_rand32());
    }

    /** generates a random number on (0,1)-real-interval */
    inline double to_real3(uint32_t v)
    {
        return (((double)v) + 0.5)*(1.0/4294967296.0); 
        /* divided by 2^32 */
    }

    /** generates a random number on (0,1)-real-interval */
    inline double genrand_real3(void)
    {
        return to_real3(gen_rand32());
    }
    /** These real versions are due to Isaku Wada */

    /** generates a random number on [0,1) with 53-bit resolution*/
    inline double to_res53(uint64_t v) 
    { 
        return v * (1.0/18446744073709551616.0L);
    }

    /** generates a random number on [0,1) with 53-bit resolution from two
     * 32 bit integers */
    inline double to_res53_mix(uint32_t x, uint32_t y) 
    { 
        return to_res53(x | ((uint64_t)y << 32));
    }

    /** generates a random number on [0,1) with 53-bit resolution
     */
    inline double genrand_res53(void) 
    { 
        return to_res53(gen_rand64());
    } 

    /** generates a random number on [0,1) with 53-bit resolution
        using 32bit integer.
     */
    inline double genrand_res53_mix(void) 
    { 
        uint32_t x, y;

        x = gen_rand32();
        y = gen_rand32();
        return to_res53_mix(x, y);
    }

public:
    SFMT()
    {
        m_psfmt32 =  &m_sfmt[0].u[0];
#if !defined(BIG_ENDIAN64) || defined(ONLY64)
        m_psfmt64 = (uint64_t *)&m_sfmt[0].u[0];
#endif
        m_initialized = 0;
        m_parity[0] = PARITY1;
        m_parity[1] = PARITY2;
        m_parity[2] = PARITY3;
        m_parity[3] = PARITY4;
    }

    SFMT(const SFMT& v)
    {
        *this = v;
    }

    SFMT& operator=(const SFMT& v)
    {
        memcpy(m_sfmt, v.m_sfmt, sizeof(m_sfmt));
        memcpy(m_parity, v.m_parity, sizeof(m_parity));
        m_idx = v.m_idx;
        m_initialized = v.m_initialized;

        m_psfmt32 =  &m_sfmt[0].u[0];
#if !defined(BIG_ENDIAN64) || defined(ONLY64)
        m_psfmt64 = (uint64_t *)&m_sfmt[0].u[0];
#endif
        return *this;
    }

    void initialize(uint32_t seed) { m_seed=seed; init_gen_rand(seed); }
    bool isInitialized() const { return m_initialized!=0; }
    uint32_t getSeed() const { return m_seed; }

    double genFloat64() { return genrand_real1(); }
    float genFloat32()  { return (float)genrand_real1(); }
    uint32_t genInt32() { return gen_rand32(); }
    uint64_t genInt64() { return gen_rand64(); }
    __m128 genVector2() { return _mm_set_ps(0.0f, 0.0f, genFloat32(), genFloat32()); }
    __m128 genVector3() { return _mm_set_ps(0.0f, genFloat32(), genFloat32(), genFloat32()); }
    __m128 genVector4() { return _mm_set_ps(genFloat32(), genFloat32(), genFloat32(), genFloat32()); }

};

} // namespace ist
#endif
