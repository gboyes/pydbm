#define PY_SSIZE_T_CLEAN
#include "Python.h"
#ifndef Py_PYTHON_H
    #error Python headers needed to compile C extensions, please install development version of Python.
#else

#include <stddef.h> /* For offsetof */
#ifndef offsetof
#define offsetof(type, member) ( (size_t) & ((type*)0) -> member )
#endif

#if !defined(WIN32) && !defined(MS_WINDOWS)
  #ifndef __stdcall
    #define __stdcall
  #endif
  #ifndef __cdecl
    #define __cdecl
  #endif
  #ifndef __fastcall
    #define __fastcall
  #endif
#endif

#ifndef DL_IMPORT
  #define DL_IMPORT(t) t
#endif
#ifndef DL_EXPORT
  #define DL_EXPORT(t) t
#endif

#ifndef PY_LONG_LONG
  #define PY_LONG_LONG LONG_LONG
#endif

#if PY_VERSION_HEX < 0x02040000
  #define METH_COEXIST 0
  #define PyDict_CheckExact(op) (Py_TYPE(op) == &PyDict_Type)
  #define PyDict_Contains(d,o)   PySequence_Contains(d,o)
#endif

#if PY_VERSION_HEX < 0x02050000
  typedef int Py_ssize_t;
  #define PY_SSIZE_T_MAX INT_MAX
  #define PY_SSIZE_T_MIN INT_MIN
  #define PY_FORMAT_SIZE_T ""
  #define PyInt_FromSsize_t(z) PyInt_FromLong(z)
  #define PyInt_AsSsize_t(o)   __Pyx_PyInt_AsInt(o)
  #define PyNumber_Index(o)    PyNumber_Int(o)
  #define PyIndex_Check(o)     PyNumber_Check(o)
  #define PyErr_WarnEx(category, message, stacklevel) PyErr_Warn(category, message)
#endif

#if PY_VERSION_HEX < 0x02060000
  #define Py_REFCNT(ob) (((PyObject*)(ob))->ob_refcnt)
  #define Py_TYPE(ob)   (((PyObject*)(ob))->ob_type)
  #define Py_SIZE(ob)   (((PyVarObject*)(ob))->ob_size)
  #define PyVarObject_HEAD_INIT(type, size) \
          PyObject_HEAD_INIT(type) size,
  #define PyType_Modified(t)

  typedef struct {
     void *buf;
     PyObject *obj;
     Py_ssize_t len;
     Py_ssize_t itemsize;
     int readonly;
     int ndim;
     char *format;
     Py_ssize_t *shape;
     Py_ssize_t *strides;
     Py_ssize_t *suboffsets;
     void *internal;
  } Py_buffer;

  #define PyBUF_SIMPLE 0
  #define PyBUF_WRITABLE 0x0001
  #define PyBUF_FORMAT 0x0004
  #define PyBUF_ND 0x0008
  #define PyBUF_STRIDES (0x0010 | PyBUF_ND)
  #define PyBUF_C_CONTIGUOUS (0x0020 | PyBUF_STRIDES)
  #define PyBUF_F_CONTIGUOUS (0x0040 | PyBUF_STRIDES)
  #define PyBUF_ANY_CONTIGUOUS (0x0080 | PyBUF_STRIDES)
  #define PyBUF_INDIRECT (0x0100 | PyBUF_STRIDES)

#endif

#if PY_MAJOR_VERSION < 3
  #define __Pyx_BUILTIN_MODULE_NAME "__builtin__"
#else
  #define __Pyx_BUILTIN_MODULE_NAME "builtins"
#endif

#if PY_MAJOR_VERSION >= 3
  #define Py_TPFLAGS_CHECKTYPES 0
  #define Py_TPFLAGS_HAVE_INDEX 0
#endif

#if (PY_VERSION_HEX < 0x02060000) || (PY_MAJOR_VERSION >= 3)
  #define Py_TPFLAGS_HAVE_NEWBUFFER 0
#endif

#if PY_MAJOR_VERSION >= 3
  #define PyBaseString_Type            PyUnicode_Type
  #define PyStringObject               PyUnicodeObject
  #define PyString_Type                PyUnicode_Type
  #define PyString_Check               PyUnicode_Check
  #define PyString_CheckExact          PyUnicode_CheckExact
#endif

#if PY_VERSION_HEX < 0x02060000
  #define PyBytesObject                PyStringObject
  #define PyBytes_Type                 PyString_Type
  #define PyBytes_Check                PyString_Check
  #define PyBytes_CheckExact           PyString_CheckExact
  #define PyBytes_FromString           PyString_FromString
  #define PyBytes_FromStringAndSize    PyString_FromStringAndSize
  #define PyBytes_FromFormat           PyString_FromFormat
  #define PyBytes_DecodeEscape         PyString_DecodeEscape
  #define PyBytes_AsString             PyString_AsString
  #define PyBytes_AsStringAndSize      PyString_AsStringAndSize
  #define PyBytes_Size                 PyString_Size
  #define PyBytes_AS_STRING            PyString_AS_STRING
  #define PyBytes_GET_SIZE             PyString_GET_SIZE
  #define PyBytes_Repr                 PyString_Repr
  #define PyBytes_Concat               PyString_Concat
  #define PyBytes_ConcatAndDel         PyString_ConcatAndDel
#endif

#if PY_VERSION_HEX < 0x02060000
  #define PySet_Check(obj)             PyObject_TypeCheck(obj, &PySet_Type)
  #define PyFrozenSet_Check(obj)       PyObject_TypeCheck(obj, &PyFrozenSet_Type)
#endif
#ifndef PySet_CheckExact
  #define PySet_CheckExact(obj)        (Py_TYPE(obj) == &PySet_Type)
#endif

#define __Pyx_TypeCheck(obj, type) PyObject_TypeCheck(obj, (PyTypeObject *)type)

#if PY_MAJOR_VERSION >= 3
  #define PyIntObject                  PyLongObject
  #define PyInt_Type                   PyLong_Type
  #define PyInt_Check(op)              PyLong_Check(op)
  #define PyInt_CheckExact(op)         PyLong_CheckExact(op)
  #define PyInt_FromString             PyLong_FromString
  #define PyInt_FromUnicode            PyLong_FromUnicode
  #define PyInt_FromLong               PyLong_FromLong
  #define PyInt_FromSize_t             PyLong_FromSize_t
  #define PyInt_FromSsize_t            PyLong_FromSsize_t
  #define PyInt_AsLong                 PyLong_AsLong
  #define PyInt_AS_LONG                PyLong_AS_LONG
  #define PyInt_AsSsize_t              PyLong_AsSsize_t
  #define PyInt_AsUnsignedLongMask     PyLong_AsUnsignedLongMask
  #define PyInt_AsUnsignedLongLongMask PyLong_AsUnsignedLongLongMask
#endif

#if PY_MAJOR_VERSION >= 3
  #define PyBoolObject                 PyLongObject
#endif

#if PY_VERSION_HEX < 0x03020000
  typedef long Py_hash_t;
  #define __Pyx_PyInt_FromHash_t PyInt_FromLong
  #define __Pyx_PyInt_AsHash_t   PyInt_AsLong
#else
  #define __Pyx_PyInt_FromHash_t PyInt_FromSsize_t
  #define __Pyx_PyInt_AsHash_t   PyInt_AsSsize_t
#endif


#if PY_MAJOR_VERSION >= 3
  #define __Pyx_PyNumber_Divide(x,y)         PyNumber_TrueDivide(x,y)
  #define __Pyx_PyNumber_InPlaceDivide(x,y)  PyNumber_InPlaceTrueDivide(x,y)
#else
  #define __Pyx_PyNumber_Divide(x,y)         PyNumber_Divide(x,y)
  #define __Pyx_PyNumber_InPlaceDivide(x,y)  PyNumber_InPlaceDivide(x,y)
#endif

#if (PY_MAJOR_VERSION < 3) || (PY_VERSION_HEX >= 0x03010300)
  #define __Pyx_PySequence_GetSlice(obj, a, b) PySequence_GetSlice(obj, a, b)
  #define __Pyx_PySequence_SetSlice(obj, a, b, value) PySequence_SetSlice(obj, a, b, value)
  #define __Pyx_PySequence_DelSlice(obj, a, b) PySequence_DelSlice(obj, a, b)
#else
  #define __Pyx_PySequence_GetSlice(obj, a, b) (unlikely(!(obj)) ? \
        (PyErr_SetString(PyExc_SystemError, "null argument to internal routine"), (PyObject*)0) : \
        (likely((obj)->ob_type->tp_as_mapping) ? (PySequence_GetSlice(obj, a, b)) : \
            (PyErr_Format(PyExc_TypeError, "'%.200s' object is unsliceable", (obj)->ob_type->tp_name), (PyObject*)0)))
  #define __Pyx_PySequence_SetSlice(obj, a, b, value) (unlikely(!(obj)) ? \
        (PyErr_SetString(PyExc_SystemError, "null argument to internal routine"), -1) : \
        (likely((obj)->ob_type->tp_as_mapping) ? (PySequence_SetSlice(obj, a, b, value)) : \
            (PyErr_Format(PyExc_TypeError, "'%.200s' object doesn't support slice assignment", (obj)->ob_type->tp_name), -1)))
  #define __Pyx_PySequence_DelSlice(obj, a, b) (unlikely(!(obj)) ? \
        (PyErr_SetString(PyExc_SystemError, "null argument to internal routine"), -1) : \
        (likely((obj)->ob_type->tp_as_mapping) ? (PySequence_DelSlice(obj, a, b)) : \
            (PyErr_Format(PyExc_TypeError, "'%.200s' object doesn't support slice deletion", (obj)->ob_type->tp_name), -1)))
#endif

#if PY_MAJOR_VERSION >= 3
  #define PyMethod_New(func, self, klass) ((self) ? PyMethod_New(func, self) : PyInstanceMethod_New(func))
#endif

#if PY_VERSION_HEX < 0x02050000
  #define __Pyx_GetAttrString(o,n)   PyObject_GetAttrString((o),((char *)(n)))
  #define __Pyx_SetAttrString(o,n,a) PyObject_SetAttrString((o),((char *)(n)),(a))
  #define __Pyx_DelAttrString(o,n)   PyObject_DelAttrString((o),((char *)(n)))
#else
  #define __Pyx_GetAttrString(o,n)   PyObject_GetAttrString((o),(n))
  #define __Pyx_SetAttrString(o,n,a) PyObject_SetAttrString((o),(n),(a))
  #define __Pyx_DelAttrString(o,n)   PyObject_DelAttrString((o),(n))
#endif

#if PY_VERSION_HEX < 0x02050000
  #define __Pyx_NAMESTR(n) ((char *)(n))
  #define __Pyx_DOCSTR(n)  ((char *)(n))
#else
  #define __Pyx_NAMESTR(n) (n)
  #define __Pyx_DOCSTR(n)  (n)
#endif

#ifndef __PYX_EXTERN_C
  #ifdef __cplusplus
    #define __PYX_EXTERN_C extern "C"
  #else
    #define __PYX_EXTERN_C extern
  #endif
#endif

#if defined(WIN32) || defined(MS_WINDOWS)
#define _USE_MATH_DEFINES
#endif
#include <math.h>
#define __PYX_HAVE__atom_
#define __PYX_HAVE_API__atom_
#include "stdio.h"
#include "stdlib.h"
#include "numpy/arrayobject.h"
#include "numpy/ufuncobject.h"
#include "math.h"
#ifdef _OPENMP
#include <omp.h>
#endif /* _OPENMP */

#ifdef PYREX_WITHOUT_ASSERTIONS
#define CYTHON_WITHOUT_ASSERTIONS
#endif


/* inline attribute */
#ifndef CYTHON_INLINE
  #if defined(__GNUC__)
    #define CYTHON_INLINE __inline__
  #elif defined(_MSC_VER)
    #define CYTHON_INLINE __inline
  #elif defined (__STDC_VERSION__) && __STDC_VERSION__ >= 199901L
    #define CYTHON_INLINE inline
  #else
    #define CYTHON_INLINE
  #endif
#endif

/* unused attribute */
#ifndef CYTHON_UNUSED
# if defined(__GNUC__)
#   if !(defined(__cplusplus)) || (__GNUC__ > 3 || (__GNUC__ == 3 && __GNUC_MINOR__ >= 4))
#     define CYTHON_UNUSED __attribute__ ((__unused__))
#   else
#     define CYTHON_UNUSED
#   endif
# elif defined(__ICC) || defined(__INTEL_COMPILER)
#   define CYTHON_UNUSED __attribute__ ((__unused__))
# else
#   define CYTHON_UNUSED
# endif
#endif

typedef struct {PyObject **p; char *s; const long n; const char* encoding; const char is_unicode; const char is_str; const char intern; } __Pyx_StringTabEntry; /*proto*/


/* Type Conversion Predeclarations */

#define __Pyx_PyBytes_FromUString(s) PyBytes_FromString((char*)s)
#define __Pyx_PyBytes_AsUString(s)   ((unsigned char*) PyBytes_AsString(s))

#define __Pyx_Owned_Py_None(b) (Py_INCREF(Py_None), Py_None)
#define __Pyx_PyBool_FromLong(b) ((b) ? (Py_INCREF(Py_True), Py_True) : (Py_INCREF(Py_False), Py_False))
static CYTHON_INLINE int __Pyx_PyObject_IsTrue(PyObject*);
static CYTHON_INLINE PyObject* __Pyx_PyNumber_Int(PyObject* x);

static CYTHON_INLINE Py_ssize_t __Pyx_PyIndex_AsSsize_t(PyObject*);
static CYTHON_INLINE PyObject * __Pyx_PyInt_FromSize_t(size_t);
static CYTHON_INLINE size_t __Pyx_PyInt_AsSize_t(PyObject*);

#define __pyx_PyFloat_AsDouble(x) (PyFloat_CheckExact(x) ? PyFloat_AS_DOUBLE(x) : PyFloat_AsDouble(x))


#ifdef __GNUC__
  /* Test for GCC > 2.95 */
  #if __GNUC__ > 2 || (__GNUC__ == 2 && (__GNUC_MINOR__ > 95))
    #define likely(x)   __builtin_expect(!!(x), 1)
    #define unlikely(x) __builtin_expect(!!(x), 0)
  #else /* __GNUC__ > 2 ... */
    #define likely(x)   (x)
    #define unlikely(x) (x)
  #endif /* __GNUC__ > 2 ... */
#else /* __GNUC__ */
  #define likely(x)   (x)
  #define unlikely(x) (x)
#endif /* __GNUC__ */
    
static PyObject *__pyx_m;
static PyObject *__pyx_b;
static PyObject *__pyx_empty_tuple;
static PyObject *__pyx_empty_bytes;
static int __pyx_lineno;
static int __pyx_clineno = 0;
static const char * __pyx_cfilenm= __FILE__;
static const char *__pyx_filename;


#if !defined(CYTHON_CCOMPLEX)
  #if defined(__cplusplus)
    #define CYTHON_CCOMPLEX 1
  #elif defined(_Complex_I)
    #define CYTHON_CCOMPLEX 1
  #else
    #define CYTHON_CCOMPLEX 0
  #endif
#endif

#if CYTHON_CCOMPLEX
  #ifdef __cplusplus
    #include <complex>
  #else
    #include <complex.h>
  #endif
#endif

#if CYTHON_CCOMPLEX && !defined(__cplusplus) && defined(__sun__) && defined(__GNUC__)
  #undef _Complex_I
  #define _Complex_I 1.0fj
#endif

static const char *__pyx_f[] = {
  "atom_.pyx",
  "numpy.pxd",
};

/* "numpy.pxd":719
 * # in Cython to enable them only on the right systems.
 * 
 * ctypedef npy_int8       int8_t             # <<<<<<<<<<<<<<
 * ctypedef npy_int16      int16_t
 * ctypedef npy_int32      int32_t
 */
typedef npy_int8 __pyx_t_5numpy_int8_t;

/* "numpy.pxd":720
 * 
 * ctypedef npy_int8       int8_t
 * ctypedef npy_int16      int16_t             # <<<<<<<<<<<<<<
 * ctypedef npy_int32      int32_t
 * ctypedef npy_int64      int64_t
 */
typedef npy_int16 __pyx_t_5numpy_int16_t;

/* "numpy.pxd":721
 * ctypedef npy_int8       int8_t
 * ctypedef npy_int16      int16_t
 * ctypedef npy_int32      int32_t             # <<<<<<<<<<<<<<
 * ctypedef npy_int64      int64_t
 * #ctypedef npy_int96      int96_t
 */
typedef npy_int32 __pyx_t_5numpy_int32_t;

/* "numpy.pxd":722
 * ctypedef npy_int16      int16_t
 * ctypedef npy_int32      int32_t
 * ctypedef npy_int64      int64_t             # <<<<<<<<<<<<<<
 * #ctypedef npy_int96      int96_t
 * #ctypedef npy_int128     int128_t
 */
typedef npy_int64 __pyx_t_5numpy_int64_t;

/* "numpy.pxd":726
 * #ctypedef npy_int128     int128_t
 * 
 * ctypedef npy_uint8      uint8_t             # <<<<<<<<<<<<<<
 * ctypedef npy_uint16     uint16_t
 * ctypedef npy_uint32     uint32_t
 */
typedef npy_uint8 __pyx_t_5numpy_uint8_t;

/* "numpy.pxd":727
 * 
 * ctypedef npy_uint8      uint8_t
 * ctypedef npy_uint16     uint16_t             # <<<<<<<<<<<<<<
 * ctypedef npy_uint32     uint32_t
 * ctypedef npy_uint64     uint64_t
 */
typedef npy_uint16 __pyx_t_5numpy_uint16_t;

/* "numpy.pxd":728
 * ctypedef npy_uint8      uint8_t
 * ctypedef npy_uint16     uint16_t
 * ctypedef npy_uint32     uint32_t             # <<<<<<<<<<<<<<
 * ctypedef npy_uint64     uint64_t
 * #ctypedef npy_uint96     uint96_t
 */
typedef npy_uint32 __pyx_t_5numpy_uint32_t;

/* "numpy.pxd":729
 * ctypedef npy_uint16     uint16_t
 * ctypedef npy_uint32     uint32_t
 * ctypedef npy_uint64     uint64_t             # <<<<<<<<<<<<<<
 * #ctypedef npy_uint96     uint96_t
 * #ctypedef npy_uint128    uint128_t
 */
typedef npy_uint64 __pyx_t_5numpy_uint64_t;

/* "numpy.pxd":733
 * #ctypedef npy_uint128    uint128_t
 * 
 * ctypedef npy_float32    float32_t             # <<<<<<<<<<<<<<
 * ctypedef npy_float64    float64_t
 * #ctypedef npy_float80    float80_t
 */
typedef npy_float32 __pyx_t_5numpy_float32_t;

/* "numpy.pxd":734
 * 
 * ctypedef npy_float32    float32_t
 * ctypedef npy_float64    float64_t             # <<<<<<<<<<<<<<
 * #ctypedef npy_float80    float80_t
 * #ctypedef npy_float128   float128_t
 */
typedef npy_float64 __pyx_t_5numpy_float64_t;

/* "numpy.pxd":743
 * # The int types are mapped a bit surprising --
 * # numpy.int corresponds to 'l' and numpy.long to 'q'
 * ctypedef npy_long       int_t             # <<<<<<<<<<<<<<
 * ctypedef npy_longlong   long_t
 * ctypedef npy_longlong   longlong_t
 */
typedef npy_long __pyx_t_5numpy_int_t;

/* "numpy.pxd":744
 * # numpy.int corresponds to 'l' and numpy.long to 'q'
 * ctypedef npy_long       int_t
 * ctypedef npy_longlong   long_t             # <<<<<<<<<<<<<<
 * ctypedef npy_longlong   longlong_t
 * 
 */
typedef npy_longlong __pyx_t_5numpy_long_t;

/* "numpy.pxd":745
 * ctypedef npy_long       int_t
 * ctypedef npy_longlong   long_t
 * ctypedef npy_longlong   longlong_t             # <<<<<<<<<<<<<<
 * 
 * ctypedef npy_ulong      uint_t
 */
typedef npy_longlong __pyx_t_5numpy_longlong_t;

/* "numpy.pxd":747
 * ctypedef npy_longlong   longlong_t
 * 
 * ctypedef npy_ulong      uint_t             # <<<<<<<<<<<<<<
 * ctypedef npy_ulonglong  ulong_t
 * ctypedef npy_ulonglong  ulonglong_t
 */
typedef npy_ulong __pyx_t_5numpy_uint_t;

/* "numpy.pxd":748
 * 
 * ctypedef npy_ulong      uint_t
 * ctypedef npy_ulonglong  ulong_t             # <<<<<<<<<<<<<<
 * ctypedef npy_ulonglong  ulonglong_t
 * 
 */
typedef npy_ulonglong __pyx_t_5numpy_ulong_t;

/* "numpy.pxd":749
 * ctypedef npy_ulong      uint_t
 * ctypedef npy_ulonglong  ulong_t
 * ctypedef npy_ulonglong  ulonglong_t             # <<<<<<<<<<<<<<
 * 
 * ctypedef npy_intp       intp_t
 */
typedef npy_ulonglong __pyx_t_5numpy_ulonglong_t;

/* "numpy.pxd":751
 * ctypedef npy_ulonglong  ulonglong_t
 * 
 * ctypedef npy_intp       intp_t             # <<<<<<<<<<<<<<
 * ctypedef npy_uintp      uintp_t
 * 
 */
typedef npy_intp __pyx_t_5numpy_intp_t;

/* "numpy.pxd":752
 * 
 * ctypedef npy_intp       intp_t
 * ctypedef npy_uintp      uintp_t             # <<<<<<<<<<<<<<
 * 
 * ctypedef npy_double     float_t
 */
typedef npy_uintp __pyx_t_5numpy_uintp_t;

/* "numpy.pxd":754
 * ctypedef npy_uintp      uintp_t
 * 
 * ctypedef npy_double     float_t             # <<<<<<<<<<<<<<
 * ctypedef npy_double     double_t
 * ctypedef npy_longdouble longdouble_t
 */
typedef npy_double __pyx_t_5numpy_float_t;

/* "numpy.pxd":755
 * 
 * ctypedef npy_double     float_t
 * ctypedef npy_double     double_t             # <<<<<<<<<<<<<<
 * ctypedef npy_longdouble longdouble_t
 * 
 */
typedef npy_double __pyx_t_5numpy_double_t;

/* "numpy.pxd":756
 * ctypedef npy_double     float_t
 * ctypedef npy_double     double_t
 * ctypedef npy_longdouble longdouble_t             # <<<<<<<<<<<<<<
 * 
 * ctypedef npy_cfloat      cfloat_t
 */
typedef npy_longdouble __pyx_t_5numpy_longdouble_t;

#if CYTHON_CCOMPLEX
  #ifdef __cplusplus
    typedef ::std::complex< float > __pyx_t_float_complex;
  #else
    typedef float _Complex __pyx_t_float_complex;
  #endif
#else
    typedef struct { float real, imag; } __pyx_t_float_complex;
#endif

#if CYTHON_CCOMPLEX
  #ifdef __cplusplus
    typedef ::std::complex< double > __pyx_t_double_complex;
  #else
    typedef double _Complex __pyx_t_double_complex;
  #endif
#else
    typedef struct { double real, imag; } __pyx_t_double_complex;
#endif

/*--- Type declarations ---*/

/* "numpy.pxd":758
 * ctypedef npy_longdouble longdouble_t
 * 
 * ctypedef npy_cfloat      cfloat_t             # <<<<<<<<<<<<<<
 * ctypedef npy_cdouble     cdouble_t
 * ctypedef npy_clongdouble clongdouble_t
 */
typedef npy_cfloat __pyx_t_5numpy_cfloat_t;

/* "numpy.pxd":759
 * 
 * ctypedef npy_cfloat      cfloat_t
 * ctypedef npy_cdouble     cdouble_t             # <<<<<<<<<<<<<<
 * ctypedef npy_clongdouble clongdouble_t
 * 
 */
typedef npy_cdouble __pyx_t_5numpy_cdouble_t;

/* "numpy.pxd":760
 * ctypedef npy_cfloat      cfloat_t
 * ctypedef npy_cdouble     cdouble_t
 * ctypedef npy_clongdouble clongdouble_t             # <<<<<<<<<<<<<<
 * 
 * ctypedef npy_cdouble     complex_t
 */
typedef npy_clongdouble __pyx_t_5numpy_clongdouble_t;

/* "numpy.pxd":762
 * ctypedef npy_clongdouble clongdouble_t
 * 
 * ctypedef npy_cdouble     complex_t             # <<<<<<<<<<<<<<
 * 
 * cdef inline object PyArray_MultiIterNew1(a):
 */
typedef npy_cdouble __pyx_t_5numpy_complex_t;
struct __pyx_opt_args_5atom__gaborFM_;
struct __pyx_opt_args_5atom__hannFM_;
struct __pyx_opt_args_5atom__blackmanFM_;
struct __pyx_opt_args_5atom__gammaFM_;
struct __pyx_opt_args_5atom__dampedFM_;
struct __pyx_opt_args_5atom__fofFM_;

/* "atom_.pyx":82
 * @cython.profile(False)
 * @cython.boundscheck(False)
 * cpdef np.ndarray[np.float_t, ndim=1, mode="c"] gaborFM_(float phi, int N, float omega, float chirp, float omega_m=0., float phi_m=0., float depth=0.):             # <<<<<<<<<<<<<<
 *     '''A gabor atom where
 *        phi := initial phase
 */
struct __pyx_opt_args_5atom__gaborFM_ {
  int __pyx_n;
  float omega_m;
  float phi_m;
  float depth;
};

/* "atom_.pyx":118
 * @cython.profile(False)
 * @cython.boundscheck(False)
 * cpdef np.ndarray[np.float_t, ndim=1, mode="c"] hannFM_(float phi, int N, float omega, float chirp, float omega_m=0., float phi_m=0., float depth=0.):             # <<<<<<<<<<<<<<
 *     '''A hann atom where
 *        phi := initial phase
 */
struct __pyx_opt_args_5atom__hannFM_ {
  int __pyx_n;
  float omega_m;
  float phi_m;
  float depth;
};

/* "atom_.pyx":154
 * @cython.profile(False)
 * @cython.boundscheck(False)
 * cpdef np.ndarray[np.float_t, ndim=1, mode="c"] blackmanFM_(float phi, int N, float omega, float chirp, float omega_m=0., float phi_m=0., float depth=0.):             # <<<<<<<<<<<<<<
 *     '''A blackman FM atom where
 *        phi := initial phase
 */
struct __pyx_opt_args_5atom__blackmanFM_ {
  int __pyx_n;
  float omega_m;
  float phi_m;
  float depth;
};

/* "atom_.pyx":187
 * @cython.profile(False)
 * @cython.boundscheck(False)
 * cpdef np.ndarray[np.float_t, ndim=1, mode="c"] gammaFM_(float phi, int N, float omega, float chirp, float order, float bandwidth, float omega_m=0., float phi_m=0., float depth=0.):             # <<<<<<<<<<<<<<
 * 
 *     cdef np.ndarray[np.float_t, ndim=1, mode="c"] out = np.zeros(N, dtype=float)
 */
struct __pyx_opt_args_5atom__gammaFM_ {
  int __pyx_n;
  float omega_m;
  float phi_m;
  float depth;
};

/* "atom_.pyx":216
 * @cython.profile(False)
 * @cython.boundscheck(False)
 * cpdef np.ndarray[np.float_t, ndim=1, mode="c"] dampedFM_(float phi, int N, float omega, float chirp, float damp, float omega_m=0., float phi_m=0., float depth=0.):             # <<<<<<<<<<<<<<
 * 
 *     cdef np.ndarray[np.float_t, ndim=1, mode="c"] out = np.zeros(N, dtype=float)
 */
struct __pyx_opt_args_5atom__dampedFM_ {
  int __pyx_n;
  float omega_m;
  float phi_m;
  float depth;
};

/* "atom_.pyx":260
 * @cython.profile(False)
 * @cython.boundscheck(False)
 * cpdef np.ndarray[np.float_t, ndim=1, mode="c"] fofFM_(float phi, int N, float omega, float chirp, int rise_n, int decay_n, float omega_m=0., float phi_m=0., float depth=0.):             # <<<<<<<<<<<<<<
 * 
 *     cdef np.ndarray[np.float_t, ndim=1, mode='c'] out = np.zeros(N, dtype=float)
 */
struct __pyx_opt_args_5atom__fofFM_ {
  int __pyx_n;
  float omega_m;
  float phi_m;
  float depth;
};


#ifndef CYTHON_REFNANNY
  #define CYTHON_REFNANNY 0
#endif

#if CYTHON_REFNANNY
  typedef struct {
    void (*INCREF)(void*, PyObject*, int);
    void (*DECREF)(void*, PyObject*, int);
    void (*GOTREF)(void*, PyObject*, int);
    void (*GIVEREF)(void*, PyObject*, int);
    void* (*SetupContext)(const char*, int, const char*);
    void (*FinishContext)(void**);
  } __Pyx_RefNannyAPIStruct;
  static __Pyx_RefNannyAPIStruct *__Pyx_RefNanny = NULL;
  static __Pyx_RefNannyAPIStruct *__Pyx_RefNannyImportAPI(const char *modname); /*proto*/
  #define __Pyx_RefNannyDeclarations void *__pyx_refnanny = NULL;
  #define __Pyx_RefNannySetupContext(name)           __pyx_refnanny = __Pyx_RefNanny->SetupContext((name), __LINE__, __FILE__)
  #define __Pyx_RefNannyFinishContext()           __Pyx_RefNanny->FinishContext(&__pyx_refnanny)
  #define __Pyx_INCREF(r)  __Pyx_RefNanny->INCREF(__pyx_refnanny, (PyObject *)(r), __LINE__)
  #define __Pyx_DECREF(r)  __Pyx_RefNanny->DECREF(__pyx_refnanny, (PyObject *)(r), __LINE__)
  #define __Pyx_GOTREF(r)  __Pyx_RefNanny->GOTREF(__pyx_refnanny, (PyObject *)(r), __LINE__)
  #define __Pyx_GIVEREF(r) __Pyx_RefNanny->GIVEREF(__pyx_refnanny, (PyObject *)(r), __LINE__)
  #define __Pyx_XINCREF(r)  do { if((r) != NULL) {__Pyx_INCREF(r); }} while(0)
  #define __Pyx_XDECREF(r)  do { if((r) != NULL) {__Pyx_DECREF(r); }} while(0)
  #define __Pyx_XGOTREF(r)  do { if((r) != NULL) {__Pyx_GOTREF(r); }} while(0)
  #define __Pyx_XGIVEREF(r) do { if((r) != NULL) {__Pyx_GIVEREF(r);}} while(0)
#else
  #define __Pyx_RefNannyDeclarations
  #define __Pyx_RefNannySetupContext(name)
  #define __Pyx_RefNannyFinishContext()
  #define __Pyx_INCREF(r) Py_INCREF(r)
  #define __Pyx_DECREF(r) Py_DECREF(r)
  #define __Pyx_GOTREF(r)
  #define __Pyx_GIVEREF(r)
  #define __Pyx_XINCREF(r) Py_XINCREF(r)
  #define __Pyx_XDECREF(r) Py_XDECREF(r)
  #define __Pyx_XGOTREF(r)
  #define __Pyx_XGIVEREF(r)
#endif /* CYTHON_REFNANNY */

static PyObject *__Pyx_GetName(PyObject *dict, PyObject *name); /*proto*/

static CYTHON_INLINE int __Pyx_TypeTest(PyObject *obj, PyTypeObject *type); /*proto*/

/* Run-time type information about structs used with buffers */
struct __Pyx_StructField_;

typedef struct {
  const char* name; /* for error messages only */
  struct __Pyx_StructField_* fields;
  size_t size;     /* sizeof(type) */
  char typegroup; /* _R_eal, _C_omplex, Signed _I_nt, _U_nsigned int, _S_truct, _P_ointer, _O_bject */
} __Pyx_TypeInfo;

typedef struct __Pyx_StructField_ {
  __Pyx_TypeInfo* type;
  const char* name;
  size_t offset;
} __Pyx_StructField;

typedef struct {
  __Pyx_StructField* field;
  size_t parent_offset;
} __Pyx_BufFmt_StackElem;


static CYTHON_INLINE int  __Pyx_GetBufferAndValidate(Py_buffer* buf, PyObject* obj, __Pyx_TypeInfo* dtype, int flags, int nd, int cast, __Pyx_BufFmt_StackElem* stack);
static CYTHON_INLINE void __Pyx_SafeReleaseBuffer(Py_buffer* info);

static CYTHON_INLINE void __Pyx_ErrRestore(PyObject *type, PyObject *value, PyObject *tb); /*proto*/
static CYTHON_INLINE void __Pyx_ErrFetch(PyObject **type, PyObject **value, PyObject **tb); /*proto*/

static void __Pyx_RaiseArgtupleInvalid(const char* func_name, int exact,
    Py_ssize_t num_min, Py_ssize_t num_max, Py_ssize_t num_found); /*proto*/

static void __Pyx_RaiseDoubleKeywordsError(
    const char* func_name, PyObject* kw_name); /*proto*/

static int __Pyx_ParseOptionalKeywords(PyObject *kwds, PyObject **argnames[],     PyObject *kwds2, PyObject *values[], Py_ssize_t num_pos_args,     const char* function_name); /*proto*/
#define __Pyx_BufPtrCContig1d(type, buf, i0, s0) ((type)buf + i0)

static void __Pyx_Raise(PyObject *type, PyObject *value, PyObject *tb, PyObject *cause); /*proto*/

static CYTHON_INLINE void __Pyx_RaiseNeedMoreValuesError(Py_ssize_t index);

static CYTHON_INLINE void __Pyx_RaiseTooManyValuesError(Py_ssize_t expected);

static CYTHON_INLINE void __Pyx_RaiseNoneNotIterableError(void);

static void __Pyx_UnpackTupleError(PyObject *, Py_ssize_t index); /*proto*/
#if PY_MAJOR_VERSION < 3
static int __Pyx_GetBuffer(PyObject *obj, Py_buffer *view, int flags);
static void __Pyx_ReleaseBuffer(Py_buffer *view);
#else
#define __Pyx_GetBuffer PyObject_GetBuffer
#define __Pyx_ReleaseBuffer PyBuffer_Release
#endif

Py_ssize_t __Pyx_zeros[] = {0};
Py_ssize_t __Pyx_minusones[] = {-1};

static PyObject *__Pyx_Import(PyObject *name, PyObject *from_list, long level); /*proto*/

#if CYTHON_CCOMPLEX
  #ifdef __cplusplus
    #define __Pyx_CREAL(z) ((z).real())
    #define __Pyx_CIMAG(z) ((z).imag())
  #else
    #define __Pyx_CREAL(z) (__real__(z))
    #define __Pyx_CIMAG(z) (__imag__(z))
  #endif
#else
    #define __Pyx_CREAL(z) ((z).real)
    #define __Pyx_CIMAG(z) ((z).imag)
#endif

#if defined(_WIN32) && defined(__cplusplus) && CYTHON_CCOMPLEX
    #define __Pyx_SET_CREAL(z,x) ((z).real(x))
    #define __Pyx_SET_CIMAG(z,y) ((z).imag(y))
#else
    #define __Pyx_SET_CREAL(z,x) __Pyx_CREAL(z) = (x)
    #define __Pyx_SET_CIMAG(z,y) __Pyx_CIMAG(z) = (y)
#endif

static CYTHON_INLINE __pyx_t_float_complex __pyx_t_float_complex_from_parts(float, float);

#if CYTHON_CCOMPLEX
    #define __Pyx_c_eqf(a, b)   ((a)==(b))
    #define __Pyx_c_sumf(a, b)  ((a)+(b))
    #define __Pyx_c_difff(a, b) ((a)-(b))
    #define __Pyx_c_prodf(a, b) ((a)*(b))
    #define __Pyx_c_quotf(a, b) ((a)/(b))
    #define __Pyx_c_negf(a)     (-(a))
  #ifdef __cplusplus
    #define __Pyx_c_is_zerof(z) ((z)==(float)0)
    #define __Pyx_c_conjf(z)    (::std::conj(z))
    #if 1
        #define __Pyx_c_absf(z)     (::std::abs(z))
        #define __Pyx_c_powf(a, b)  (::std::pow(a, b))
    #endif
  #else
    #define __Pyx_c_is_zerof(z) ((z)==0)
    #define __Pyx_c_conjf(z)    (conjf(z))
    #if 1
        #define __Pyx_c_absf(z)     (cabsf(z))
        #define __Pyx_c_powf(a, b)  (cpowf(a, b))
    #endif
 #endif
#else
    static CYTHON_INLINE int __Pyx_c_eqf(__pyx_t_float_complex, __pyx_t_float_complex);
    static CYTHON_INLINE __pyx_t_float_complex __Pyx_c_sumf(__pyx_t_float_complex, __pyx_t_float_complex);
    static CYTHON_INLINE __pyx_t_float_complex __Pyx_c_difff(__pyx_t_float_complex, __pyx_t_float_complex);
    static CYTHON_INLINE __pyx_t_float_complex __Pyx_c_prodf(__pyx_t_float_complex, __pyx_t_float_complex);
    static CYTHON_INLINE __pyx_t_float_complex __Pyx_c_quotf(__pyx_t_float_complex, __pyx_t_float_complex);
    static CYTHON_INLINE __pyx_t_float_complex __Pyx_c_negf(__pyx_t_float_complex);
    static CYTHON_INLINE int __Pyx_c_is_zerof(__pyx_t_float_complex);
    static CYTHON_INLINE __pyx_t_float_complex __Pyx_c_conjf(__pyx_t_float_complex);
    #if 1
        static CYTHON_INLINE float __Pyx_c_absf(__pyx_t_float_complex);
        static CYTHON_INLINE __pyx_t_float_complex __Pyx_c_powf(__pyx_t_float_complex, __pyx_t_float_complex);
    #endif
#endif

static CYTHON_INLINE __pyx_t_double_complex __pyx_t_double_complex_from_parts(double, double);

#if CYTHON_CCOMPLEX
    #define __Pyx_c_eq(a, b)   ((a)==(b))
    #define __Pyx_c_sum(a, b)  ((a)+(b))
    #define __Pyx_c_diff(a, b) ((a)-(b))
    #define __Pyx_c_prod(a, b) ((a)*(b))
    #define __Pyx_c_quot(a, b) ((a)/(b))
    #define __Pyx_c_neg(a)     (-(a))
  #ifdef __cplusplus
    #define __Pyx_c_is_zero(z) ((z)==(double)0)
    #define __Pyx_c_conj(z)    (::std::conj(z))
    #if 1
        #define __Pyx_c_abs(z)     (::std::abs(z))
        #define __Pyx_c_pow(a, b)  (::std::pow(a, b))
    #endif
  #else
    #define __Pyx_c_is_zero(z) ((z)==0)
    #define __Pyx_c_conj(z)    (conj(z))
    #if 1
        #define __Pyx_c_abs(z)     (cabs(z))
        #define __Pyx_c_pow(a, b)  (cpow(a, b))
    #endif
 #endif
#else
    static CYTHON_INLINE int __Pyx_c_eq(__pyx_t_double_complex, __pyx_t_double_complex);
    static CYTHON_INLINE __pyx_t_double_complex __Pyx_c_sum(__pyx_t_double_complex, __pyx_t_double_complex);
    static CYTHON_INLINE __pyx_t_double_complex __Pyx_c_diff(__pyx_t_double_complex, __pyx_t_double_complex);
    static CYTHON_INLINE __pyx_t_double_complex __Pyx_c_prod(__pyx_t_double_complex, __pyx_t_double_complex);
    static CYTHON_INLINE __pyx_t_double_complex __Pyx_c_quot(__pyx_t_double_complex, __pyx_t_double_complex);
    static CYTHON_INLINE __pyx_t_double_complex __Pyx_c_neg(__pyx_t_double_complex);
    static CYTHON_INLINE int __Pyx_c_is_zero(__pyx_t_double_complex);
    static CYTHON_INLINE __pyx_t_double_complex __Pyx_c_conj(__pyx_t_double_complex);
    #if 1
        static CYTHON_INLINE double __Pyx_c_abs(__pyx_t_double_complex);
        static CYTHON_INLINE __pyx_t_double_complex __Pyx_c_pow(__pyx_t_double_complex, __pyx_t_double_complex);
    #endif
#endif

static CYTHON_INLINE unsigned char __Pyx_PyInt_AsUnsignedChar(PyObject *);

static CYTHON_INLINE unsigned short __Pyx_PyInt_AsUnsignedShort(PyObject *);

static CYTHON_INLINE unsigned int __Pyx_PyInt_AsUnsignedInt(PyObject *);

static CYTHON_INLINE char __Pyx_PyInt_AsChar(PyObject *);

static CYTHON_INLINE short __Pyx_PyInt_AsShort(PyObject *);

static CYTHON_INLINE int __Pyx_PyInt_AsInt(PyObject *);

static CYTHON_INLINE signed char __Pyx_PyInt_AsSignedChar(PyObject *);

static CYTHON_INLINE signed short __Pyx_PyInt_AsSignedShort(PyObject *);

static CYTHON_INLINE signed int __Pyx_PyInt_AsSignedInt(PyObject *);

static CYTHON_INLINE int __Pyx_PyInt_AsLongDouble(PyObject *);

static CYTHON_INLINE unsigned long __Pyx_PyInt_AsUnsignedLong(PyObject *);

static CYTHON_INLINE unsigned PY_LONG_LONG __Pyx_PyInt_AsUnsignedLongLong(PyObject *);

static CYTHON_INLINE long __Pyx_PyInt_AsLong(PyObject *);

static CYTHON_INLINE PY_LONG_LONG __Pyx_PyInt_AsLongLong(PyObject *);

static CYTHON_INLINE signed long __Pyx_PyInt_AsSignedLong(PyObject *);

static CYTHON_INLINE signed PY_LONG_LONG __Pyx_PyInt_AsSignedLongLong(PyObject *);

static void __Pyx_WriteUnraisable(const char *name, int clineno,
                                  int lineno, const char *filename); /*proto*/

static int __Pyx_check_binary_version(void);

static PyTypeObject *__Pyx_ImportType(const char *module_name, const char *class_name, size_t size, int strict);  /*proto*/

static PyObject *__Pyx_ImportModule(const char *name); /*proto*/

static void __Pyx_AddTraceback(const char *funcname, int __pyx_clineno,
                               int __pyx_lineno, const char *__pyx_filename); /*proto*/

static int __Pyx_InitStrings(__Pyx_StringTabEntry *t); /*proto*/

/* Module declarations from 'cython.cython.view' */

/* Module declarations from 'cython' */

/* Module declarations from 'cpython.buffer' */

/* Module declarations from 'cpython.ref' */

/* Module declarations from 'libc.stdio' */

/* Module declarations from 'cpython.object' */

/* Module declarations from 'libc.stdlib' */

/* Module declarations from 'numpy' */

/* Module declarations from 'numpy' */
static PyTypeObject *__pyx_ptype_5numpy_dtype = 0;
static PyTypeObject *__pyx_ptype_5numpy_flatiter = 0;
static PyTypeObject *__pyx_ptype_5numpy_broadcast = 0;
static PyTypeObject *__pyx_ptype_5numpy_ndarray = 0;
static PyTypeObject *__pyx_ptype_5numpy_ufunc = 0;
static CYTHON_INLINE PyObject *__pyx_f_5numpy_PyArray_MultiIterNew1(PyObject *); /*proto*/
static CYTHON_INLINE PyObject *__pyx_f_5numpy_PyArray_MultiIterNew2(PyObject *, PyObject *); /*proto*/
static CYTHON_INLINE PyObject *__pyx_f_5numpy_PyArray_MultiIterNew3(PyObject *, PyObject *, PyObject *); /*proto*/
static CYTHON_INLINE PyObject *__pyx_f_5numpy_PyArray_MultiIterNew4(PyObject *, PyObject *, PyObject *, PyObject *); /*proto*/
static CYTHON_INLINE PyObject *__pyx_f_5numpy_PyArray_MultiIterNew5(PyObject *, PyObject *, PyObject *, PyObject *, PyObject *); /*proto*/
static CYTHON_INLINE char *__pyx_f_5numpy__util_dtypestring(PyArray_Descr *, char *, char *, int *); /*proto*/
static CYTHON_INLINE void __pyx_f_5numpy_set_array_base(PyArrayObject *, PyObject *); /*proto*/
static CYTHON_INLINE PyObject *__pyx_f_5numpy_get_array_base(PyArrayObject *); /*proto*/

/* Module declarations from 'atom_' */
static double __pyx_v_5atom__pi;
static CYTHON_INLINE double __pyx_f_5atom__gauss(int, int); /*proto*/
static CYTHON_INLINE double __pyx_f_5atom__hann(int, int); /*proto*/
static CYTHON_INLINE double __pyx_f_5atom__blackman(int, int); /*proto*/
static CYTHON_INLINE double __pyx_f_5atom__realSinusoid(int, float, float, float); /*proto*/
static CYTHON_INLINE double __pyx_f_5atom__realSinusoidFM(int, float, float, float, float, float, PyObject *); /*proto*/
static PyArrayObject *__pyx_f_5atom__realSinusoid_(int, float, float, float, int __pyx_skip_dispatch); /*proto*/
static PyArrayObject *__pyx_f_5atom__gabor_(float, int, float, float, int __pyx_skip_dispatch); /*proto*/
static PyArrayObject *__pyx_f_5atom__gaborFM_(float, int, float, float, int __pyx_skip_dispatch, struct __pyx_opt_args_5atom__gaborFM_ *__pyx_optional_args); /*proto*/
static PyArrayObject *__pyx_f_5atom__hann_(float, int, float, float, int __pyx_skip_dispatch); /*proto*/
static PyArrayObject *__pyx_f_5atom__hannFM_(float, int, float, float, int __pyx_skip_dispatch, struct __pyx_opt_args_5atom__hannFM_ *__pyx_optional_args); /*proto*/
static PyArrayObject *__pyx_f_5atom__blackman_(float, int, float, float, int __pyx_skip_dispatch); /*proto*/
static PyArrayObject *__pyx_f_5atom__blackmanFM_(float, int, float, float, int __pyx_skip_dispatch, struct __pyx_opt_args_5atom__blackmanFM_ *__pyx_optional_args); /*proto*/
static PyArrayObject *__pyx_f_5atom__gamma_(float, int, float, float, float, float, int __pyx_skip_dispatch); /*proto*/
static PyArrayObject *__pyx_f_5atom__gammaFM_(float, int, float, float, float, float, int __pyx_skip_dispatch, struct __pyx_opt_args_5atom__gammaFM_ *__pyx_optional_args); /*proto*/
static PyArrayObject *__pyx_f_5atom__damped_(float, int, float, float, float, int __pyx_skip_dispatch); /*proto*/
static PyArrayObject *__pyx_f_5atom__dampedFM_(float, int, float, float, float, int __pyx_skip_dispatch, struct __pyx_opt_args_5atom__dampedFM_ *__pyx_optional_args); /*proto*/
static PyArrayObject *__pyx_f_5atom__fof_(float, int, float, float, int, int, int __pyx_skip_dispatch); /*proto*/
static PyArrayObject *__pyx_f_5atom__fofFM_(float, int, float, float, int, int, int __pyx_skip_dispatch, struct __pyx_opt_args_5atom__fofFM_ *__pyx_optional_args); /*proto*/
static __Pyx_TypeInfo __Pyx_TypeInfo_nn___pyx_t_5numpy_float_t = { "float_t", NULL, sizeof(__pyx_t_5numpy_float_t), 'R' };
#define __Pyx_MODULE_NAME "atom_"
int __pyx_module_is_main_atom_ = 0;

/* Implementation of 'atom_' */
static PyObject *__pyx_builtin_range;
static PyObject *__pyx_builtin_max;
static PyObject *__pyx_builtin_ValueError;
static PyObject *__pyx_builtin_RuntimeError;
static char __pyx_k_1[] = "ndarray is not C contiguous";
static char __pyx_k_3[] = "ndarray is not Fortran contiguous";
static char __pyx_k_5[] = "Non-native byte order not supported";
static char __pyx_k_7[] = "unknown dtype code in numpy.pxd (%d)";
static char __pyx_k_8[] = "Format string allocated too short, see comment in numpy.pxd";
static char __pyx_k_11[] = "Format string allocated too short.";
static char __pyx_k__B[] = "B";
static char __pyx_k__H[] = "H";
static char __pyx_k__I[] = "I";
static char __pyx_k__L[] = "L";
static char __pyx_k__N[] = "N";
static char __pyx_k__O[] = "O";
static char __pyx_k__Q[] = "Q";
static char __pyx_k__b[] = "b";
static char __pyx_k__d[] = "d";
static char __pyx_k__f[] = "f";
static char __pyx_k__g[] = "g";
static char __pyx_k__h[] = "h";
static char __pyx_k__i[] = "i";
static char __pyx_k__l[] = "l";
static char __pyx_k__q[] = "q";
static char __pyx_k__Zd[] = "Zd";
static char __pyx_k__Zf[] = "Zf";
static char __pyx_k__Zg[] = "Zg";
static char __pyx_k__np[] = "np";
static char __pyx_k__cos[] = "cos";
static char __pyx_k__max[] = "max";
static char __pyx_k__phi[] = "phi";
static char __pyx_k__damp[] = "damp";
static char __pyx_k__chirp[] = "chirp";
static char __pyx_k__depth[] = "depth";
static char __pyx_k__dtype[] = "dtype";
static char __pyx_k__numpy[] = "numpy";
static char __pyx_k__omega[] = "omega";
static char __pyx_k__order[] = "order";
static char __pyx_k__phi_m[] = "phi_m";
static char __pyx_k__range[] = "range";
static char __pyx_k__zeros[] = "zeros";
static char __pyx_k__arange[] = "arange";
static char __pyx_k__rise_n[] = "rise_n";
static char __pyx_k__decay_n[] = "decay_n";
static char __pyx_k__omega_m[] = "omega_m";
static char __pyx_k____main__[] = "__main__";
static char __pyx_k____test__[] = "__test__";
static char __pyx_k__bandwidth[] = "bandwidth";
static char __pyx_k__ValueError[] = "ValueError";
static char __pyx_k__RuntimeError[] = "RuntimeError";
static PyObject *__pyx_kp_u_1;
static PyObject *__pyx_kp_u_11;
static PyObject *__pyx_kp_u_3;
static PyObject *__pyx_kp_u_5;
static PyObject *__pyx_kp_u_7;
static PyObject *__pyx_kp_u_8;
static PyObject *__pyx_n_s__N;
static PyObject *__pyx_n_s__RuntimeError;
static PyObject *__pyx_n_s__ValueError;
static PyObject *__pyx_n_s____main__;
static PyObject *__pyx_n_s____test__;
static PyObject *__pyx_n_s__arange;
static PyObject *__pyx_n_s__bandwidth;
static PyObject *__pyx_n_s__chirp;
static PyObject *__pyx_n_s__cos;
static PyObject *__pyx_n_s__damp;
static PyObject *__pyx_n_s__decay_n;
static PyObject *__pyx_n_s__depth;
static PyObject *__pyx_n_s__dtype;
static PyObject *__pyx_n_s__max;
static PyObject *__pyx_n_s__np;
static PyObject *__pyx_n_s__numpy;
static PyObject *__pyx_n_s__omega;
static PyObject *__pyx_n_s__omega_m;
static PyObject *__pyx_n_s__order;
static PyObject *__pyx_n_s__phi;
static PyObject *__pyx_n_s__phi_m;
static PyObject *__pyx_n_s__range;
static PyObject *__pyx_n_s__rise_n;
static PyObject *__pyx_n_s__zeros;
static PyObject *__pyx_int_15;
static PyObject *__pyx_k_tuple_2;
static PyObject *__pyx_k_tuple_4;
static PyObject *__pyx_k_tuple_6;
static PyObject *__pyx_k_tuple_9;
static PyObject *__pyx_k_tuple_10;
static PyObject *__pyx_k_tuple_12;

/* "atom_.pyx":21
 * @cython.profile(False)
 * @cython.boundscheck(False)
 * cdef inline double gauss(int i, int N):             # <<<<<<<<<<<<<<
 *     return exp(-( (i - N/2. )**2) / (2. * ( (0.125 * N)**2)))
 * 
 */

static CYTHON_INLINE double __pyx_f_5atom__gauss(int __pyx_v_i, int __pyx_v_N) {
  double __pyx_r;
  __Pyx_RefNannyDeclarations
  double __pyx_t_1;
  double __pyx_t_2;
  int __pyx_lineno = 0;
  const char *__pyx_filename = NULL;
  int __pyx_clineno = 0;
  __Pyx_RefNannySetupContext("gauss");

  /* "atom_.pyx":22
 * @cython.boundscheck(False)
 * cdef inline double gauss(int i, int N):
 *     return exp(-( (i - N/2. )**2) / (2. * ( (0.125 * N)**2)))             # <<<<<<<<<<<<<<
 * 
 * #Hann for C#
 */
  __pyx_t_1 = (-pow((__pyx_v_i - (__pyx_v_N / 2.)), 2.0));
  __pyx_t_2 = (2. * pow((0.125 * __pyx_v_N), 2.0));
  if (unlikely(__pyx_t_2 == 0)) {
    PyErr_Format(PyExc_ZeroDivisionError, "float division");
    {__pyx_filename = __pyx_f[0]; __pyx_lineno = 22; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  }
  __pyx_r = exp((__pyx_t_1 / __pyx_t_2));
  goto __pyx_L0;

  __pyx_r = 0;
  goto __pyx_L0;
  __pyx_L1_error:;
  __Pyx_WriteUnraisable("atom_.gauss", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __pyx_r = 0;
  __pyx_L0:;
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

/* "atom_.pyx":28
 * @cython.profile(False)
 * @cython.boundscheck(False)
 * cdef inline double hann(int i, int N):             # <<<<<<<<<<<<<<
 *     return 0.5 * (1 - cos( 2*pi*i / (N-1)))
 * 
 */

static CYTHON_INLINE double __pyx_f_5atom__hann(int __pyx_v_i, int __pyx_v_N) {
  double __pyx_r;
  __Pyx_RefNannyDeclarations
  double __pyx_t_1;
  long __pyx_t_2;
  int __pyx_lineno = 0;
  const char *__pyx_filename = NULL;
  int __pyx_clineno = 0;
  __Pyx_RefNannySetupContext("hann");

  /* "atom_.pyx":29
 * @cython.boundscheck(False)
 * cdef inline double hann(int i, int N):
 *     return 0.5 * (1 - cos( 2*pi*i / (N-1)))             # <<<<<<<<<<<<<<
 * 
 * #Blackman for C#
 */
  __pyx_t_1 = ((2.0 * __pyx_v_5atom__pi) * __pyx_v_i);
  __pyx_t_2 = (__pyx_v_N - 1);
  if (unlikely(__pyx_t_2 == 0)) {
    PyErr_Format(PyExc_ZeroDivisionError, "float division");
    {__pyx_filename = __pyx_f[0]; __pyx_lineno = 29; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  }
  __pyx_r = (0.5 * (1.0 - cos((__pyx_t_1 / __pyx_t_2))));
  goto __pyx_L0;

  __pyx_r = 0;
  goto __pyx_L0;
  __pyx_L1_error:;
  __Pyx_WriteUnraisable("atom_.hann", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __pyx_r = 0;
  __pyx_L0:;
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

/* "atom_.pyx":35
 * @cython.profile(False)
 * @cython.boundscheck(False)
 * cdef inline double blackman(int i, int N):             # <<<<<<<<<<<<<<
 *     return 0.42 - 0.5 * cos(2*pi*i/(N-1)) + 0.08 * cos(4*pi*i/(N-1))
 * 
 */

static CYTHON_INLINE double __pyx_f_5atom__blackman(int __pyx_v_i, int __pyx_v_N) {
  double __pyx_r;
  __Pyx_RefNannyDeclarations
  double __pyx_t_1;
  long __pyx_t_2;
  double __pyx_t_3;
  long __pyx_t_4;
  int __pyx_lineno = 0;
  const char *__pyx_filename = NULL;
  int __pyx_clineno = 0;
  __Pyx_RefNannySetupContext("blackman");

  /* "atom_.pyx":36
 * @cython.boundscheck(False)
 * cdef inline double blackman(int i, int N):
 *     return 0.42 - 0.5 * cos(2*pi*i/(N-1)) + 0.08 * cos(4*pi*i/(N-1))             # <<<<<<<<<<<<<<
 * 
 * #Real Sinusoid for C
 */
  __pyx_t_1 = ((2.0 * __pyx_v_5atom__pi) * __pyx_v_i);
  __pyx_t_2 = (__pyx_v_N - 1);
  if (unlikely(__pyx_t_2 == 0)) {
    PyErr_Format(PyExc_ZeroDivisionError, "float division");
    {__pyx_filename = __pyx_f[0]; __pyx_lineno = 36; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  }
  __pyx_t_3 = ((4.0 * __pyx_v_5atom__pi) * __pyx_v_i);
  __pyx_t_4 = (__pyx_v_N - 1);
  if (unlikely(__pyx_t_4 == 0)) {
    PyErr_Format(PyExc_ZeroDivisionError, "float division");
    {__pyx_filename = __pyx_f[0]; __pyx_lineno = 36; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  }
  __pyx_r = ((0.42 - (0.5 * cos((__pyx_t_1 / __pyx_t_2)))) + (0.08 * cos((__pyx_t_3 / __pyx_t_4))));
  goto __pyx_L0;

  __pyx_r = 0;
  goto __pyx_L0;
  __pyx_L1_error:;
  __Pyx_WriteUnraisable("atom_.blackman", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __pyx_r = 0;
  __pyx_L0:;
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

/* "atom_.pyx":42
 * @cython.profile(False)
 * @cython.boundscheck(False)
 * cdef inline double realSinusoid(int i, float omega, float chirp, float phi):             # <<<<<<<<<<<<<<
 *     return cos(2 * pi * (omega + 0.5 * chirp * i) * i + phi)
 * 
 */

static CYTHON_INLINE double __pyx_f_5atom__realSinusoid(int __pyx_v_i, float __pyx_v_omega, float __pyx_v_chirp, float __pyx_v_phi) {
  double __pyx_r;
  __Pyx_RefNannyDeclarations
  __Pyx_RefNannySetupContext("realSinusoid");

  /* "atom_.pyx":43
 * @cython.boundscheck(False)
 * cdef inline double realSinusoid(int i, float omega, float chirp, float phi):
 *     return cos(2 * pi * (omega + 0.5 * chirp * i) * i + phi)             # <<<<<<<<<<<<<<
 * 
 * #Real Vibrating Sinusoid for C
 */
  __pyx_r = cos(((((2.0 * __pyx_v_5atom__pi) * (__pyx_v_omega + ((0.5 * __pyx_v_chirp) * __pyx_v_i))) * __pyx_v_i) + __pyx_v_phi));
  goto __pyx_L0;

  __pyx_r = 0;
  __pyx_L0:;
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

/* "atom_.pyx":49
 * @cython.profile(False)
 * @cython.boundscheck(False)
 * cdef inline double realSinusoidFM(int i, float omega, float chirp, float phi, float omega_m, float phi_m, depth):             # <<<<<<<<<<<<<<
 *     return cos(2 * pi * (omega + 0.5 * chirp * i) * i + (depth/omega_m * sin(2 * pi * omega_m * i + phi_m)) + phi)
 * 
 */

static CYTHON_INLINE double __pyx_f_5atom__realSinusoidFM(int __pyx_v_i, float __pyx_v_omega, float __pyx_v_chirp, float __pyx_v_phi, float __pyx_v_omega_m, float __pyx_v_phi_m, PyObject *__pyx_v_depth) {
  double __pyx_r;
  __Pyx_RefNannyDeclarations
  PyObject *__pyx_t_1 = NULL;
  PyObject *__pyx_t_2 = NULL;
  PyObject *__pyx_t_3 = NULL;
  PyObject *__pyx_t_4 = NULL;
  double __pyx_t_5;
  int __pyx_lineno = 0;
  const char *__pyx_filename = NULL;
  int __pyx_clineno = 0;
  __Pyx_RefNannySetupContext("realSinusoidFM");

  /* "atom_.pyx":50
 * @cython.boundscheck(False)
 * cdef inline double realSinusoidFM(int i, float omega, float chirp, float phi, float omega_m, float phi_m, depth):
 *     return cos(2 * pi * (omega + 0.5 * chirp * i) * i + (depth/omega_m * sin(2 * pi * omega_m * i + phi_m)) + phi)             # <<<<<<<<<<<<<<
 * 
 * #Real Sinusoid for python
 */
  __pyx_t_1 = PyFloat_FromDouble((((2.0 * __pyx_v_5atom__pi) * (__pyx_v_omega + ((0.5 * __pyx_v_chirp) * __pyx_v_i))) * __pyx_v_i)); if (unlikely(!__pyx_t_1)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 50; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(__pyx_t_1);
  __pyx_t_2 = PyFloat_FromDouble(__pyx_v_omega_m); if (unlikely(!__pyx_t_2)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 50; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(__pyx_t_2);
  __pyx_t_3 = __Pyx_PyNumber_Divide(__pyx_v_depth, __pyx_t_2); if (unlikely(!__pyx_t_3)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 50; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(__pyx_t_3);
  __Pyx_DECREF(__pyx_t_2); __pyx_t_2 = 0;
  __pyx_t_2 = PyFloat_FromDouble(sin(((((2.0 * __pyx_v_5atom__pi) * __pyx_v_omega_m) * __pyx_v_i) + __pyx_v_phi_m))); if (unlikely(!__pyx_t_2)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 50; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(__pyx_t_2);
  __pyx_t_4 = PyNumber_Multiply(__pyx_t_3, __pyx_t_2); if (unlikely(!__pyx_t_4)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 50; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(__pyx_t_4);
  __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
  __Pyx_DECREF(__pyx_t_2); __pyx_t_2 = 0;
  __pyx_t_2 = PyNumber_Add(__pyx_t_1, __pyx_t_4); if (unlikely(!__pyx_t_2)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 50; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(__pyx_t_2);
  __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;
  __Pyx_DECREF(__pyx_t_4); __pyx_t_4 = 0;
  __pyx_t_4 = PyFloat_FromDouble(__pyx_v_phi); if (unlikely(!__pyx_t_4)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 50; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(__pyx_t_4);
  __pyx_t_1 = PyNumber_Add(__pyx_t_2, __pyx_t_4); if (unlikely(!__pyx_t_1)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 50; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(__pyx_t_1);
  __Pyx_DECREF(__pyx_t_2); __pyx_t_2 = 0;
  __Pyx_DECREF(__pyx_t_4); __pyx_t_4 = 0;
  __pyx_t_5 = __pyx_PyFloat_AsDouble(__pyx_t_1); if (unlikely((__pyx_t_5 == (double)-1) && PyErr_Occurred())) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 50; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;
  __pyx_r = cos(__pyx_t_5);
  goto __pyx_L0;

  __pyx_r = 0;
  goto __pyx_L0;
  __pyx_L1_error:;
  __Pyx_XDECREF(__pyx_t_1);
  __Pyx_XDECREF(__pyx_t_2);
  __Pyx_XDECREF(__pyx_t_3);
  __Pyx_XDECREF(__pyx_t_4);
  __Pyx_WriteUnraisable("atom_.realSinusoidFM", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __pyx_r = 0;
  __pyx_L0:;
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

/* "atom_.pyx":56
 * @cython.profile(False)
 * @cython.boundscheck(False)
 * cpdef np.ndarray[np.float_t, ndim=1, mode="c"] realSinusoid_(int N, float omega, float chirp, float phi):             # <<<<<<<<<<<<<<
 *     cdef np.ndarray[np.float_t, ndim=1, mode="c"] out = np.cos(2 * pi * (omega + chirp * 0.5 * np.arange(N)) * np.arange(N) + phi)
 *     return out
 */

static PyObject *__pyx_pf_5atom__realSinusoid_(PyObject *__pyx_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
static PyArrayObject *__pyx_f_5atom__realSinusoid_(int __pyx_v_N, float __pyx_v_omega, float __pyx_v_chirp, float __pyx_v_phi, int __pyx_skip_dispatch) {
  PyArrayObject *__pyx_v_out = 0;
  Py_buffer __pyx_bstruct_out;
  Py_ssize_t __pyx_bstride_0_out = 0;
  Py_ssize_t __pyx_bshape_0_out = 0;
  PyArrayObject *__pyx_r = NULL;
  __Pyx_RefNannyDeclarations
  PyObject *__pyx_t_1 = NULL;
  PyObject *__pyx_t_2 = NULL;
  PyObject *__pyx_t_3 = NULL;
  PyObject *__pyx_t_4 = NULL;
  PyObject *__pyx_t_5 = NULL;
  PyObject *__pyx_t_6 = NULL;
  PyObject *__pyx_t_7 = NULL;
  PyArrayObject *__pyx_t_8 = NULL;
  int __pyx_lineno = 0;
  const char *__pyx_filename = NULL;
  int __pyx_clineno = 0;
  __Pyx_RefNannySetupContext("realSinusoid_");
  __pyx_bstruct_out.buf = NULL;

  /* "atom_.pyx":57
 * @cython.boundscheck(False)
 * cpdef np.ndarray[np.float_t, ndim=1, mode="c"] realSinusoid_(int N, float omega, float chirp, float phi):
 *     cdef np.ndarray[np.float_t, ndim=1, mode="c"] out = np.cos(2 * pi * (omega + chirp * 0.5 * np.arange(N)) * np.arange(N) + phi)             # <<<<<<<<<<<<<<
 *     return out
 * 
 */
  __pyx_t_1 = __Pyx_GetName(__pyx_m, __pyx_n_s__np); if (unlikely(!__pyx_t_1)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 57; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(__pyx_t_1);
  __pyx_t_2 = PyObject_GetAttr(__pyx_t_1, __pyx_n_s__cos); if (unlikely(!__pyx_t_2)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 57; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(__pyx_t_2);
  __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;
  __pyx_t_1 = PyFloat_FromDouble((2.0 * __pyx_v_5atom__pi)); if (unlikely(!__pyx_t_1)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 57; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(__pyx_t_1);
  __pyx_t_3 = PyFloat_FromDouble(__pyx_v_omega); if (unlikely(!__pyx_t_3)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 57; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(__pyx_t_3);
  __pyx_t_4 = PyFloat_FromDouble((__pyx_v_chirp * 0.5)); if (unlikely(!__pyx_t_4)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 57; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(__pyx_t_4);
  __pyx_t_5 = __Pyx_GetName(__pyx_m, __pyx_n_s__np); if (unlikely(!__pyx_t_5)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 57; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(__pyx_t_5);
  __pyx_t_6 = PyObject_GetAttr(__pyx_t_5, __pyx_n_s__arange); if (unlikely(!__pyx_t_6)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 57; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(__pyx_t_6);
  __Pyx_DECREF(__pyx_t_5); __pyx_t_5 = 0;
  __pyx_t_5 = PyInt_FromLong(__pyx_v_N); if (unlikely(!__pyx_t_5)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 57; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(__pyx_t_5);
  __pyx_t_7 = PyTuple_New(1); if (unlikely(!__pyx_t_7)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 57; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(((PyObject *)__pyx_t_7));
  PyTuple_SET_ITEM(__pyx_t_7, 0, __pyx_t_5);
  __Pyx_GIVEREF(__pyx_t_5);
  __pyx_t_5 = 0;
  __pyx_t_5 = PyObject_Call(__pyx_t_6, ((PyObject *)__pyx_t_7), NULL); if (unlikely(!__pyx_t_5)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 57; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(__pyx_t_5);
  __Pyx_DECREF(__pyx_t_6); __pyx_t_6 = 0;
  __Pyx_DECREF(((PyObject *)__pyx_t_7)); __pyx_t_7 = 0;
  __pyx_t_7 = PyNumber_Multiply(__pyx_t_4, __pyx_t_5); if (unlikely(!__pyx_t_7)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 57; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(__pyx_t_7);
  __Pyx_DECREF(__pyx_t_4); __pyx_t_4 = 0;
  __Pyx_DECREF(__pyx_t_5); __pyx_t_5 = 0;
  __pyx_t_5 = PyNumber_Add(__pyx_t_3, __pyx_t_7); if (unlikely(!__pyx_t_5)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 57; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(__pyx_t_5);
  __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
  __Pyx_DECREF(__pyx_t_7); __pyx_t_7 = 0;
  __pyx_t_7 = PyNumber_Multiply(__pyx_t_1, __pyx_t_5); if (unlikely(!__pyx_t_7)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 57; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(__pyx_t_7);
  __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;
  __Pyx_DECREF(__pyx_t_5); __pyx_t_5 = 0;
  __pyx_t_5 = __Pyx_GetName(__pyx_m, __pyx_n_s__np); if (unlikely(!__pyx_t_5)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 57; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(__pyx_t_5);
  __pyx_t_1 = PyObject_GetAttr(__pyx_t_5, __pyx_n_s__arange); if (unlikely(!__pyx_t_1)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 57; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(__pyx_t_1);
  __Pyx_DECREF(__pyx_t_5); __pyx_t_5 = 0;
  __pyx_t_5 = PyInt_FromLong(__pyx_v_N); if (unlikely(!__pyx_t_5)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 57; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(__pyx_t_5);
  __pyx_t_3 = PyTuple_New(1); if (unlikely(!__pyx_t_3)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 57; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(((PyObject *)__pyx_t_3));
  PyTuple_SET_ITEM(__pyx_t_3, 0, __pyx_t_5);
  __Pyx_GIVEREF(__pyx_t_5);
  __pyx_t_5 = 0;
  __pyx_t_5 = PyObject_Call(__pyx_t_1, ((PyObject *)__pyx_t_3), NULL); if (unlikely(!__pyx_t_5)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 57; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(__pyx_t_5);
  __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;
  __Pyx_DECREF(((PyObject *)__pyx_t_3)); __pyx_t_3 = 0;
  __pyx_t_3 = PyNumber_Multiply(__pyx_t_7, __pyx_t_5); if (unlikely(!__pyx_t_3)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 57; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(__pyx_t_3);
  __Pyx_DECREF(__pyx_t_7); __pyx_t_7 = 0;
  __Pyx_DECREF(__pyx_t_5); __pyx_t_5 = 0;
  __pyx_t_5 = PyFloat_FromDouble(__pyx_v_phi); if (unlikely(!__pyx_t_5)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 57; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(__pyx_t_5);
  __pyx_t_7 = PyNumber_Add(__pyx_t_3, __pyx_t_5); if (unlikely(!__pyx_t_7)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 57; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(__pyx_t_7);
  __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
  __Pyx_DECREF(__pyx_t_5); __pyx_t_5 = 0;
  __pyx_t_5 = PyTuple_New(1); if (unlikely(!__pyx_t_5)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 57; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(((PyObject *)__pyx_t_5));
  PyTuple_SET_ITEM(__pyx_t_5, 0, __pyx_t_7);
  __Pyx_GIVEREF(__pyx_t_7);
  __pyx_t_7 = 0;
  __pyx_t_7 = PyObject_Call(__pyx_t_2, ((PyObject *)__pyx_t_5), NULL); if (unlikely(!__pyx_t_7)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 57; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(__pyx_t_7);
  __Pyx_DECREF(__pyx_t_2); __pyx_t_2 = 0;
  __Pyx_DECREF(((PyObject *)__pyx_t_5)); __pyx_t_5 = 0;
  if (!(likely(((__pyx_t_7) == Py_None) || likely(__Pyx_TypeTest(__pyx_t_7, __pyx_ptype_5numpy_ndarray))))) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 57; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __pyx_t_8 = ((PyArrayObject *)__pyx_t_7);
  {
    __Pyx_BufFmt_StackElem __pyx_stack[1];
    if (unlikely(__Pyx_GetBufferAndValidate(&__pyx_bstruct_out, (PyObject*)__pyx_t_8, &__Pyx_TypeInfo_nn___pyx_t_5numpy_float_t, PyBUF_FORMAT| PyBUF_C_CONTIGUOUS, 1, 0, __pyx_stack) == -1)) {
      __pyx_v_out = ((PyArrayObject *)Py_None); __Pyx_INCREF(Py_None); __pyx_bstruct_out.buf = NULL;
      {__pyx_filename = __pyx_f[0]; __pyx_lineno = 57; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
    } else {__pyx_bstride_0_out = __pyx_bstruct_out.strides[0];
      __pyx_bshape_0_out = __pyx_bstruct_out.shape[0];
    }
  }
  __pyx_t_8 = 0;
  __pyx_v_out = ((PyArrayObject *)__pyx_t_7);
  __pyx_t_7 = 0;

  /* "atom_.pyx":58
 * cpdef np.ndarray[np.float_t, ndim=1, mode="c"] realSinusoid_(int N, float omega, float chirp, float phi):
 *     cdef np.ndarray[np.float_t, ndim=1, mode="c"] out = np.cos(2 * pi * (omega + chirp * 0.5 * np.arange(N)) * np.arange(N) + phi)
 *     return out             # <<<<<<<<<<<<<<
 * 
 * #Gabor for python
 */
  __Pyx_XDECREF(((PyObject *)__pyx_r));
  __Pyx_INCREF(((PyObject *)__pyx_v_out));
  __pyx_r = ((PyArrayObject *)__pyx_v_out);
  goto __pyx_L0;

  __pyx_r = ((PyArrayObject *)Py_None); __Pyx_INCREF(Py_None);
  goto __pyx_L0;
  __pyx_L1_error:;
  __Pyx_XDECREF(__pyx_t_1);
  __Pyx_XDECREF(__pyx_t_2);
  __Pyx_XDECREF(__pyx_t_3);
  __Pyx_XDECREF(__pyx_t_4);
  __Pyx_XDECREF(__pyx_t_5);
  __Pyx_XDECREF(__pyx_t_6);
  __Pyx_XDECREF(__pyx_t_7);
  { PyObject *__pyx_type, *__pyx_value, *__pyx_tb;
    __Pyx_ErrFetch(&__pyx_type, &__pyx_value, &__pyx_tb);
    __Pyx_SafeReleaseBuffer(&__pyx_bstruct_out);
  __Pyx_ErrRestore(__pyx_type, __pyx_value, __pyx_tb);}
  __Pyx_AddTraceback("atom_.realSinusoid_", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __pyx_r = 0;
  goto __pyx_L2;
  __pyx_L0:;
  __Pyx_SafeReleaseBuffer(&__pyx_bstruct_out);
  __pyx_L2:;
  __Pyx_XDECREF((PyObject *)__pyx_v_out);
  __Pyx_XGIVEREF((PyObject *)__pyx_r);
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

/* "atom_.pyx":56
 * @cython.profile(False)
 * @cython.boundscheck(False)
 * cpdef np.ndarray[np.float_t, ndim=1, mode="c"] realSinusoid_(int N, float omega, float chirp, float phi):             # <<<<<<<<<<<<<<
 *     cdef np.ndarray[np.float_t, ndim=1, mode="c"] out = np.cos(2 * pi * (omega + chirp * 0.5 * np.arange(N)) * np.arange(N) + phi)
 *     return out
 */

static PyObject *__pyx_pf_5atom__realSinusoid_(PyObject *__pyx_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
static PyObject *__pyx_pf_5atom__realSinusoid_(PyObject *__pyx_self, PyObject *__pyx_args, PyObject *__pyx_kwds) {
  int __pyx_v_N;
  float __pyx_v_omega;
  float __pyx_v_chirp;
  float __pyx_v_phi;
  PyObject *__pyx_r = NULL;
  __Pyx_RefNannyDeclarations
  PyObject *__pyx_t_1 = NULL;
  int __pyx_lineno = 0;
  const char *__pyx_filename = NULL;
  int __pyx_clineno = 0;
  static PyObject **__pyx_pyargnames[] = {&__pyx_n_s__N,&__pyx_n_s__omega,&__pyx_n_s__chirp,&__pyx_n_s__phi,0};
  __Pyx_RefNannySetupContext("realSinusoid_");
  __pyx_self = __pyx_self;
  {
    PyObject* values[4] = {0,0,0,0};
    if (unlikely(__pyx_kwds)) {
      Py_ssize_t kw_args;
      switch (PyTuple_GET_SIZE(__pyx_args)) {
        case  4: values[3] = PyTuple_GET_ITEM(__pyx_args, 3);
        case  3: values[2] = PyTuple_GET_ITEM(__pyx_args, 2);
        case  2: values[1] = PyTuple_GET_ITEM(__pyx_args, 1);
        case  1: values[0] = PyTuple_GET_ITEM(__pyx_args, 0);
        case  0: break;
        default: goto __pyx_L5_argtuple_error;
      }
      kw_args = PyDict_Size(__pyx_kwds);
      switch (PyTuple_GET_SIZE(__pyx_args)) {
        case  0:
        values[0] = PyDict_GetItem(__pyx_kwds, __pyx_n_s__N);
        if (likely(values[0])) kw_args--;
        else goto __pyx_L5_argtuple_error;
        case  1:
        values[1] = PyDict_GetItem(__pyx_kwds, __pyx_n_s__omega);
        if (likely(values[1])) kw_args--;
        else {
          __Pyx_RaiseArgtupleInvalid("realSinusoid_", 1, 4, 4, 1); {__pyx_filename = __pyx_f[0]; __pyx_lineno = 56; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
        }
        case  2:
        values[2] = PyDict_GetItem(__pyx_kwds, __pyx_n_s__chirp);
        if (likely(values[2])) kw_args--;
        else {
          __Pyx_RaiseArgtupleInvalid("realSinusoid_", 1, 4, 4, 2); {__pyx_filename = __pyx_f[0]; __pyx_lineno = 56; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
        }
        case  3:
        values[3] = PyDict_GetItem(__pyx_kwds, __pyx_n_s__phi);
        if (likely(values[3])) kw_args--;
        else {
          __Pyx_RaiseArgtupleInvalid("realSinusoid_", 1, 4, 4, 3); {__pyx_filename = __pyx_f[0]; __pyx_lineno = 56; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
        }
      }
      if (unlikely(kw_args > 0)) {
        if (unlikely(__Pyx_ParseOptionalKeywords(__pyx_kwds, __pyx_pyargnames, 0, values, PyTuple_GET_SIZE(__pyx_args), "realSinusoid_") < 0)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 56; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
      }
    } else if (PyTuple_GET_SIZE(__pyx_args) != 4) {
      goto __pyx_L5_argtuple_error;
    } else {
      values[0] = PyTuple_GET_ITEM(__pyx_args, 0);
      values[1] = PyTuple_GET_ITEM(__pyx_args, 1);
      values[2] = PyTuple_GET_ITEM(__pyx_args, 2);
      values[3] = PyTuple_GET_ITEM(__pyx_args, 3);
    }
    __pyx_v_N = __Pyx_PyInt_AsInt(values[0]); if (unlikely((__pyx_v_N == (int)-1) && PyErr_Occurred())) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 56; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
    __pyx_v_omega = __pyx_PyFloat_AsDouble(values[1]); if (unlikely((__pyx_v_omega == (float)-1) && PyErr_Occurred())) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 56; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
    __pyx_v_chirp = __pyx_PyFloat_AsDouble(values[2]); if (unlikely((__pyx_v_chirp == (float)-1) && PyErr_Occurred())) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 56; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
    __pyx_v_phi = __pyx_PyFloat_AsDouble(values[3]); if (unlikely((__pyx_v_phi == (float)-1) && PyErr_Occurred())) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 56; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
  }
  goto __pyx_L4_argument_unpacking_done;
  __pyx_L5_argtuple_error:;
  __Pyx_RaiseArgtupleInvalid("realSinusoid_", 1, 4, 4, PyTuple_GET_SIZE(__pyx_args)); {__pyx_filename = __pyx_f[0]; __pyx_lineno = 56; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
  __pyx_L3_error:;
  __Pyx_AddTraceback("atom_.realSinusoid_", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __Pyx_RefNannyFinishContext();
  return NULL;
  __pyx_L4_argument_unpacking_done:;
  __Pyx_XDECREF(__pyx_r);
  __pyx_t_1 = ((PyObject *)__pyx_f_5atom__realSinusoid_(__pyx_v_N, __pyx_v_omega, __pyx_v_chirp, __pyx_v_phi, 0)); if (unlikely(!__pyx_t_1)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 56; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(__pyx_t_1);
  __pyx_r = __pyx_t_1;
  __pyx_t_1 = 0;
  goto __pyx_L0;

  __pyx_r = Py_None; __Pyx_INCREF(Py_None);
  goto __pyx_L0;
  __pyx_L1_error:;
  __Pyx_XDECREF(__pyx_t_1);
  __Pyx_AddTraceback("atom_.realSinusoid_", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __pyx_r = NULL;
  __pyx_L0:;
  __Pyx_XGIVEREF(__pyx_r);
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

/* "atom_.pyx":64
 * @cython.profile(False)
 * @cython.boundscheck(False)
 * cpdef np.ndarray[np.float_t, ndim=1, mode="c"] gabor_(float phi, int N, float omega, float chirp):             # <<<<<<<<<<<<<<
 *     '''A gabor atom where
 *        phi := initial phase
 */

static PyObject *__pyx_pf_5atom__1gabor_(PyObject *__pyx_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
static PyArrayObject *__pyx_f_5atom__gabor_(float __pyx_v_phi, int __pyx_v_N, float __pyx_v_omega, float __pyx_v_chirp, int __pyx_skip_dispatch) {
  PyArrayObject *__pyx_v_out = 0;
  int __pyx_v_i;
  Py_buffer __pyx_bstruct_out;
  Py_ssize_t __pyx_bstride_0_out = 0;
  Py_ssize_t __pyx_bshape_0_out = 0;
  PyArrayObject *__pyx_r = NULL;
  __Pyx_RefNannyDeclarations
  PyObject *__pyx_t_1 = NULL;
  PyObject *__pyx_t_2 = NULL;
  PyObject *__pyx_t_3 = NULL;
  PyObject *__pyx_t_4 = NULL;
  PyArrayObject *__pyx_t_5 = NULL;
  int __pyx_t_6;
  int __pyx_t_7;
  int __pyx_t_8;
  int __pyx_lineno = 0;
  const char *__pyx_filename = NULL;
  int __pyx_clineno = 0;
  __Pyx_RefNannySetupContext("gabor_");
  __pyx_bstruct_out.buf = NULL;

  /* "atom_.pyx":70
 *        omega := normalized frequency'''
 * 
 *     cdef np.ndarray[np.float_t, ndim=1, mode="c"] out = np.zeros(N, dtype=float)             # <<<<<<<<<<<<<<
 *     cdef int i
 * 
 */
  __pyx_t_1 = __Pyx_GetName(__pyx_m, __pyx_n_s__np); if (unlikely(!__pyx_t_1)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 70; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(__pyx_t_1);
  __pyx_t_2 = PyObject_GetAttr(__pyx_t_1, __pyx_n_s__zeros); if (unlikely(!__pyx_t_2)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 70; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(__pyx_t_2);
  __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;
  __pyx_t_1 = PyInt_FromLong(__pyx_v_N); if (unlikely(!__pyx_t_1)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 70; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(__pyx_t_1);
  __pyx_t_3 = PyTuple_New(1); if (unlikely(!__pyx_t_3)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 70; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(((PyObject *)__pyx_t_3));
  PyTuple_SET_ITEM(__pyx_t_3, 0, __pyx_t_1);
  __Pyx_GIVEREF(__pyx_t_1);
  __pyx_t_1 = 0;
  __pyx_t_1 = PyDict_New(); if (unlikely(!__pyx_t_1)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 70; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(((PyObject *)__pyx_t_1));
  if (PyDict_SetItem(__pyx_t_1, ((PyObject *)__pyx_n_s__dtype), ((PyObject *)((PyObject*)(&PyFloat_Type)))) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 70; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __pyx_t_4 = PyEval_CallObjectWithKeywords(__pyx_t_2, ((PyObject *)__pyx_t_3), ((PyObject *)__pyx_t_1)); if (unlikely(!__pyx_t_4)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 70; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(__pyx_t_4);
  __Pyx_DECREF(__pyx_t_2); __pyx_t_2 = 0;
  __Pyx_DECREF(((PyObject *)__pyx_t_3)); __pyx_t_3 = 0;
  __Pyx_DECREF(((PyObject *)__pyx_t_1)); __pyx_t_1 = 0;
  if (!(likely(((__pyx_t_4) == Py_None) || likely(__Pyx_TypeTest(__pyx_t_4, __pyx_ptype_5numpy_ndarray))))) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 70; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __pyx_t_5 = ((PyArrayObject *)__pyx_t_4);
  {
    __Pyx_BufFmt_StackElem __pyx_stack[1];
    if (unlikely(__Pyx_GetBufferAndValidate(&__pyx_bstruct_out, (PyObject*)__pyx_t_5, &__Pyx_TypeInfo_nn___pyx_t_5numpy_float_t, PyBUF_FORMAT| PyBUF_C_CONTIGUOUS| PyBUF_WRITABLE, 1, 0, __pyx_stack) == -1)) {
      __pyx_v_out = ((PyArrayObject *)Py_None); __Pyx_INCREF(Py_None); __pyx_bstruct_out.buf = NULL;
      {__pyx_filename = __pyx_f[0]; __pyx_lineno = 70; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
    } else {__pyx_bstride_0_out = __pyx_bstruct_out.strides[0];
      __pyx_bshape_0_out = __pyx_bstruct_out.shape[0];
    }
  }
  __pyx_t_5 = 0;
  __pyx_v_out = ((PyArrayObject *)__pyx_t_4);
  __pyx_t_4 = 0;

  /* "atom_.pyx":73
 *     cdef int i
 * 
 *     for i in range(N):             # <<<<<<<<<<<<<<
 *         out[i] = gauss(i, N) * realSinusoid(i, omega, chirp/N, phi)
 * 
 */
  __pyx_t_6 = __pyx_v_N;
  for (__pyx_t_7 = 0; __pyx_t_7 < __pyx_t_6; __pyx_t_7+=1) {
    __pyx_v_i = __pyx_t_7;

    /* "atom_.pyx":74
 * 
 *     for i in range(N):
 *         out[i] = gauss(i, N) * realSinusoid(i, omega, chirp/N, phi)             # <<<<<<<<<<<<<<
 * 
 *     return out
 */
    if (unlikely(__pyx_v_N == 0)) {
      PyErr_Format(PyExc_ZeroDivisionError, "float division");
      {__pyx_filename = __pyx_f[0]; __pyx_lineno = 74; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
    }
    __pyx_t_8 = __pyx_v_i;
    if (__pyx_t_8 < 0) __pyx_t_8 += __pyx_bshape_0_out;
    *__Pyx_BufPtrCContig1d(__pyx_t_5numpy_float_t *, __pyx_bstruct_out.buf, __pyx_t_8, __pyx_bstride_0_out) = (__pyx_f_5atom__gauss(__pyx_v_i, __pyx_v_N) * __pyx_f_5atom__realSinusoid(__pyx_v_i, __pyx_v_omega, (__pyx_v_chirp / __pyx_v_N), __pyx_v_phi));
  }

  /* "atom_.pyx":76
 *         out[i] = gauss(i, N) * realSinusoid(i, omega, chirp/N, phi)
 * 
 *     return out             # <<<<<<<<<<<<<<
 * 
 * #GaborFM for python
 */
  __Pyx_XDECREF(((PyObject *)__pyx_r));
  __Pyx_INCREF(((PyObject *)__pyx_v_out));
  __pyx_r = ((PyArrayObject *)__pyx_v_out);
  goto __pyx_L0;

  __pyx_r = ((PyArrayObject *)Py_None); __Pyx_INCREF(Py_None);
  goto __pyx_L0;
  __pyx_L1_error:;
  __Pyx_XDECREF(__pyx_t_1);
  __Pyx_XDECREF(__pyx_t_2);
  __Pyx_XDECREF(__pyx_t_3);
  __Pyx_XDECREF(__pyx_t_4);
  { PyObject *__pyx_type, *__pyx_value, *__pyx_tb;
    __Pyx_ErrFetch(&__pyx_type, &__pyx_value, &__pyx_tb);
    __Pyx_SafeReleaseBuffer(&__pyx_bstruct_out);
  __Pyx_ErrRestore(__pyx_type, __pyx_value, __pyx_tb);}
  __Pyx_AddTraceback("atom_.gabor_", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __pyx_r = 0;
  goto __pyx_L2;
  __pyx_L0:;
  __Pyx_SafeReleaseBuffer(&__pyx_bstruct_out);
  __pyx_L2:;
  __Pyx_XDECREF((PyObject *)__pyx_v_out);
  __Pyx_XGIVEREF((PyObject *)__pyx_r);
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

/* "atom_.pyx":64
 * @cython.profile(False)
 * @cython.boundscheck(False)
 * cpdef np.ndarray[np.float_t, ndim=1, mode="c"] gabor_(float phi, int N, float omega, float chirp):             # <<<<<<<<<<<<<<
 *     '''A gabor atom where
 *        phi := initial phase
 */

static PyObject *__pyx_pf_5atom__1gabor_(PyObject *__pyx_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
static char __pyx_doc_5atom__1gabor_[] = "A gabor atom where\n       phi := initial phase\n       N := scale, i.e. length\n       omega := normalized frequency";
static PyObject *__pyx_pf_5atom__1gabor_(PyObject *__pyx_self, PyObject *__pyx_args, PyObject *__pyx_kwds) {
  float __pyx_v_phi;
  int __pyx_v_N;
  float __pyx_v_omega;
  float __pyx_v_chirp;
  PyObject *__pyx_r = NULL;
  __Pyx_RefNannyDeclarations
  PyObject *__pyx_t_1 = NULL;
  int __pyx_lineno = 0;
  const char *__pyx_filename = NULL;
  int __pyx_clineno = 0;
  static PyObject **__pyx_pyargnames[] = {&__pyx_n_s__phi,&__pyx_n_s__N,&__pyx_n_s__omega,&__pyx_n_s__chirp,0};
  __Pyx_RefNannySetupContext("gabor_");
  __pyx_self = __pyx_self;
  {
    PyObject* values[4] = {0,0,0,0};
    if (unlikely(__pyx_kwds)) {
      Py_ssize_t kw_args;
      switch (PyTuple_GET_SIZE(__pyx_args)) {
        case  4: values[3] = PyTuple_GET_ITEM(__pyx_args, 3);
        case  3: values[2] = PyTuple_GET_ITEM(__pyx_args, 2);
        case  2: values[1] = PyTuple_GET_ITEM(__pyx_args, 1);
        case  1: values[0] = PyTuple_GET_ITEM(__pyx_args, 0);
        case  0: break;
        default: goto __pyx_L5_argtuple_error;
      }
      kw_args = PyDict_Size(__pyx_kwds);
      switch (PyTuple_GET_SIZE(__pyx_args)) {
        case  0:
        values[0] = PyDict_GetItem(__pyx_kwds, __pyx_n_s__phi);
        if (likely(values[0])) kw_args--;
        else goto __pyx_L5_argtuple_error;
        case  1:
        values[1] = PyDict_GetItem(__pyx_kwds, __pyx_n_s__N);
        if (likely(values[1])) kw_args--;
        else {
          __Pyx_RaiseArgtupleInvalid("gabor_", 1, 4, 4, 1); {__pyx_filename = __pyx_f[0]; __pyx_lineno = 64; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
        }
        case  2:
        values[2] = PyDict_GetItem(__pyx_kwds, __pyx_n_s__omega);
        if (likely(values[2])) kw_args--;
        else {
          __Pyx_RaiseArgtupleInvalid("gabor_", 1, 4, 4, 2); {__pyx_filename = __pyx_f[0]; __pyx_lineno = 64; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
        }
        case  3:
        values[3] = PyDict_GetItem(__pyx_kwds, __pyx_n_s__chirp);
        if (likely(values[3])) kw_args--;
        else {
          __Pyx_RaiseArgtupleInvalid("gabor_", 1, 4, 4, 3); {__pyx_filename = __pyx_f[0]; __pyx_lineno = 64; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
        }
      }
      if (unlikely(kw_args > 0)) {
        if (unlikely(__Pyx_ParseOptionalKeywords(__pyx_kwds, __pyx_pyargnames, 0, values, PyTuple_GET_SIZE(__pyx_args), "gabor_") < 0)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 64; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
      }
    } else if (PyTuple_GET_SIZE(__pyx_args) != 4) {
      goto __pyx_L5_argtuple_error;
    } else {
      values[0] = PyTuple_GET_ITEM(__pyx_args, 0);
      values[1] = PyTuple_GET_ITEM(__pyx_args, 1);
      values[2] = PyTuple_GET_ITEM(__pyx_args, 2);
      values[3] = PyTuple_GET_ITEM(__pyx_args, 3);
    }
    __pyx_v_phi = __pyx_PyFloat_AsDouble(values[0]); if (unlikely((__pyx_v_phi == (float)-1) && PyErr_Occurred())) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 64; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
    __pyx_v_N = __Pyx_PyInt_AsInt(values[1]); if (unlikely((__pyx_v_N == (int)-1) && PyErr_Occurred())) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 64; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
    __pyx_v_omega = __pyx_PyFloat_AsDouble(values[2]); if (unlikely((__pyx_v_omega == (float)-1) && PyErr_Occurred())) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 64; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
    __pyx_v_chirp = __pyx_PyFloat_AsDouble(values[3]); if (unlikely((__pyx_v_chirp == (float)-1) && PyErr_Occurred())) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 64; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
  }
  goto __pyx_L4_argument_unpacking_done;
  __pyx_L5_argtuple_error:;
  __Pyx_RaiseArgtupleInvalid("gabor_", 1, 4, 4, PyTuple_GET_SIZE(__pyx_args)); {__pyx_filename = __pyx_f[0]; __pyx_lineno = 64; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
  __pyx_L3_error:;
  __Pyx_AddTraceback("atom_.gabor_", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __Pyx_RefNannyFinishContext();
  return NULL;
  __pyx_L4_argument_unpacking_done:;
  __Pyx_XDECREF(__pyx_r);
  __pyx_t_1 = ((PyObject *)__pyx_f_5atom__gabor_(__pyx_v_phi, __pyx_v_N, __pyx_v_omega, __pyx_v_chirp, 0)); if (unlikely(!__pyx_t_1)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 64; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(__pyx_t_1);
  __pyx_r = __pyx_t_1;
  __pyx_t_1 = 0;
  goto __pyx_L0;

  __pyx_r = Py_None; __Pyx_INCREF(Py_None);
  goto __pyx_L0;
  __pyx_L1_error:;
  __Pyx_XDECREF(__pyx_t_1);
  __Pyx_AddTraceback("atom_.gabor_", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __pyx_r = NULL;
  __pyx_L0:;
  __Pyx_XGIVEREF(__pyx_r);
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

/* "atom_.pyx":82
 * @cython.profile(False)
 * @cython.boundscheck(False)
 * cpdef np.ndarray[np.float_t, ndim=1, mode="c"] gaborFM_(float phi, int N, float omega, float chirp, float omega_m=0., float phi_m=0., float depth=0.):             # <<<<<<<<<<<<<<
 *     '''A gabor atom where
 *        phi := initial phase
 */

static PyObject *__pyx_pf_5atom__2gaborFM_(PyObject *__pyx_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
static PyArrayObject *__pyx_f_5atom__gaborFM_(float __pyx_v_phi, int __pyx_v_N, float __pyx_v_omega, float __pyx_v_chirp, int __pyx_skip_dispatch, struct __pyx_opt_args_5atom__gaborFM_ *__pyx_optional_args) {
  float __pyx_v_omega_m = ((float)0.);
  float __pyx_v_phi_m = ((float)0.);
  float __pyx_v_depth = ((float)0.);
  PyArrayObject *__pyx_v_out = 0;
  int __pyx_v_i;
  Py_buffer __pyx_bstruct_out;
  Py_ssize_t __pyx_bstride_0_out = 0;
  Py_ssize_t __pyx_bshape_0_out = 0;
  PyArrayObject *__pyx_r = NULL;
  __Pyx_RefNannyDeclarations
  PyObject *__pyx_t_1 = NULL;
  PyObject *__pyx_t_2 = NULL;
  PyObject *__pyx_t_3 = NULL;
  PyObject *__pyx_t_4 = NULL;
  PyArrayObject *__pyx_t_5 = NULL;
  int __pyx_t_6;
  int __pyx_t_7;
  int __pyx_t_8;
  int __pyx_lineno = 0;
  const char *__pyx_filename = NULL;
  int __pyx_clineno = 0;
  __Pyx_RefNannySetupContext("gaborFM_");
  if (__pyx_optional_args) {
    if (__pyx_optional_args->__pyx_n > 0) {
      __pyx_v_omega_m = __pyx_optional_args->omega_m;
      if (__pyx_optional_args->__pyx_n > 1) {
        __pyx_v_phi_m = __pyx_optional_args->phi_m;
        if (__pyx_optional_args->__pyx_n > 2) {
          __pyx_v_depth = __pyx_optional_args->depth;
        }
      }
    }
  }
  __pyx_bstruct_out.buf = NULL;

  /* "atom_.pyx":88
 *        omega := normalized frequency'''
 * 
 *     cdef np.ndarray[np.float_t, ndim=1, mode="c"] out = np.zeros(N, dtype=float)             # <<<<<<<<<<<<<<
 *     cdef int i
 * 
 */
  __pyx_t_1 = __Pyx_GetName(__pyx_m, __pyx_n_s__np); if (unlikely(!__pyx_t_1)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 88; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(__pyx_t_1);
  __pyx_t_2 = PyObject_GetAttr(__pyx_t_1, __pyx_n_s__zeros); if (unlikely(!__pyx_t_2)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 88; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(__pyx_t_2);
  __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;
  __pyx_t_1 = PyInt_FromLong(__pyx_v_N); if (unlikely(!__pyx_t_1)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 88; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(__pyx_t_1);
  __pyx_t_3 = PyTuple_New(1); if (unlikely(!__pyx_t_3)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 88; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(((PyObject *)__pyx_t_3));
  PyTuple_SET_ITEM(__pyx_t_3, 0, __pyx_t_1);
  __Pyx_GIVEREF(__pyx_t_1);
  __pyx_t_1 = 0;
  __pyx_t_1 = PyDict_New(); if (unlikely(!__pyx_t_1)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 88; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(((PyObject *)__pyx_t_1));
  if (PyDict_SetItem(__pyx_t_1, ((PyObject *)__pyx_n_s__dtype), ((PyObject *)((PyObject*)(&PyFloat_Type)))) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 88; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __pyx_t_4 = PyEval_CallObjectWithKeywords(__pyx_t_2, ((PyObject *)__pyx_t_3), ((PyObject *)__pyx_t_1)); if (unlikely(!__pyx_t_4)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 88; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(__pyx_t_4);
  __Pyx_DECREF(__pyx_t_2); __pyx_t_2 = 0;
  __Pyx_DECREF(((PyObject *)__pyx_t_3)); __pyx_t_3 = 0;
  __Pyx_DECREF(((PyObject *)__pyx_t_1)); __pyx_t_1 = 0;
  if (!(likely(((__pyx_t_4) == Py_None) || likely(__Pyx_TypeTest(__pyx_t_4, __pyx_ptype_5numpy_ndarray))))) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 88; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __pyx_t_5 = ((PyArrayObject *)__pyx_t_4);
  {
    __Pyx_BufFmt_StackElem __pyx_stack[1];
    if (unlikely(__Pyx_GetBufferAndValidate(&__pyx_bstruct_out, (PyObject*)__pyx_t_5, &__Pyx_TypeInfo_nn___pyx_t_5numpy_float_t, PyBUF_FORMAT| PyBUF_C_CONTIGUOUS| PyBUF_WRITABLE, 1, 0, __pyx_stack) == -1)) {
      __pyx_v_out = ((PyArrayObject *)Py_None); __Pyx_INCREF(Py_None); __pyx_bstruct_out.buf = NULL;
      {__pyx_filename = __pyx_f[0]; __pyx_lineno = 88; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
    } else {__pyx_bstride_0_out = __pyx_bstruct_out.strides[0];
      __pyx_bshape_0_out = __pyx_bstruct_out.shape[0];
    }
  }
  __pyx_t_5 = 0;
  __pyx_v_out = ((PyArrayObject *)__pyx_t_4);
  __pyx_t_4 = 0;

  /* "atom_.pyx":91
 *     cdef int i
 * 
 *     for i in range(N):             # <<<<<<<<<<<<<<
 *         out[i] = gauss(i, N) * realSinusoidFM(i, omega, chirp, phi, omega_m, phi_m, depth)
 * 
 */
  __pyx_t_6 = __pyx_v_N;
  for (__pyx_t_7 = 0; __pyx_t_7 < __pyx_t_6; __pyx_t_7+=1) {
    __pyx_v_i = __pyx_t_7;

    /* "atom_.pyx":92
 * 
 *     for i in range(N):
 *         out[i] = gauss(i, N) * realSinusoidFM(i, omega, chirp, phi, omega_m, phi_m, depth)             # <<<<<<<<<<<<<<
 * 
 *     return out
 */
    __pyx_t_4 = PyFloat_FromDouble(__pyx_v_depth); if (unlikely(!__pyx_t_4)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 92; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
    __Pyx_GOTREF(__pyx_t_4);
    __pyx_t_8 = __pyx_v_i;
    if (__pyx_t_8 < 0) __pyx_t_8 += __pyx_bshape_0_out;
    *__Pyx_BufPtrCContig1d(__pyx_t_5numpy_float_t *, __pyx_bstruct_out.buf, __pyx_t_8, __pyx_bstride_0_out) = (__pyx_f_5atom__gauss(__pyx_v_i, __pyx_v_N) * __pyx_f_5atom__realSinusoidFM(__pyx_v_i, __pyx_v_omega, __pyx_v_chirp, __pyx_v_phi, __pyx_v_omega_m, __pyx_v_phi_m, __pyx_t_4));
    __Pyx_DECREF(__pyx_t_4); __pyx_t_4 = 0;
  }

  /* "atom_.pyx":94
 *         out[i] = gauss(i, N) * realSinusoidFM(i, omega, chirp, phi, omega_m, phi_m, depth)
 * 
 *     return out             # <<<<<<<<<<<<<<
 * 
 * #Hann for python
 */
  __Pyx_XDECREF(((PyObject *)__pyx_r));
  __Pyx_INCREF(((PyObject *)__pyx_v_out));
  __pyx_r = ((PyArrayObject *)__pyx_v_out);
  goto __pyx_L0;

  __pyx_r = ((PyArrayObject *)Py_None); __Pyx_INCREF(Py_None);
  goto __pyx_L0;
  __pyx_L1_error:;
  __Pyx_XDECREF(__pyx_t_1);
  __Pyx_XDECREF(__pyx_t_2);
  __Pyx_XDECREF(__pyx_t_3);
  __Pyx_XDECREF(__pyx_t_4);
  { PyObject *__pyx_type, *__pyx_value, *__pyx_tb;
    __Pyx_ErrFetch(&__pyx_type, &__pyx_value, &__pyx_tb);
    __Pyx_SafeReleaseBuffer(&__pyx_bstruct_out);
  __Pyx_ErrRestore(__pyx_type, __pyx_value, __pyx_tb);}
  __Pyx_AddTraceback("atom_.gaborFM_", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __pyx_r = 0;
  goto __pyx_L2;
  __pyx_L0:;
  __Pyx_SafeReleaseBuffer(&__pyx_bstruct_out);
  __pyx_L2:;
  __Pyx_XDECREF((PyObject *)__pyx_v_out);
  __Pyx_XGIVEREF((PyObject *)__pyx_r);
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

/* "atom_.pyx":82
 * @cython.profile(False)
 * @cython.boundscheck(False)
 * cpdef np.ndarray[np.float_t, ndim=1, mode="c"] gaborFM_(float phi, int N, float omega, float chirp, float omega_m=0., float phi_m=0., float depth=0.):             # <<<<<<<<<<<<<<
 *     '''A gabor atom where
 *        phi := initial phase
 */

static PyObject *__pyx_pf_5atom__2gaborFM_(PyObject *__pyx_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
static char __pyx_doc_5atom__2gaborFM_[] = "A gabor atom where\n       phi := initial phase\n       N := scale, i.e. length\n       omega := normalized frequency";
static PyObject *__pyx_pf_5atom__2gaborFM_(PyObject *__pyx_self, PyObject *__pyx_args, PyObject *__pyx_kwds) {
  float __pyx_v_phi;
  int __pyx_v_N;
  float __pyx_v_omega;
  float __pyx_v_chirp;
  float __pyx_v_omega_m;
  float __pyx_v_phi_m;
  float __pyx_v_depth;
  PyObject *__pyx_r = NULL;
  __Pyx_RefNannyDeclarations
  PyObject *__pyx_t_1 = NULL;
  struct __pyx_opt_args_5atom__gaborFM_ __pyx_t_2;
  int __pyx_lineno = 0;
  const char *__pyx_filename = NULL;
  int __pyx_clineno = 0;
  static PyObject **__pyx_pyargnames[] = {&__pyx_n_s__phi,&__pyx_n_s__N,&__pyx_n_s__omega,&__pyx_n_s__chirp,&__pyx_n_s__omega_m,&__pyx_n_s__phi_m,&__pyx_n_s__depth,0};
  __Pyx_RefNannySetupContext("gaborFM_");
  __pyx_self = __pyx_self;
  {
    PyObject* values[7] = {0,0,0,0,0,0,0};
    if (unlikely(__pyx_kwds)) {
      Py_ssize_t kw_args;
      switch (PyTuple_GET_SIZE(__pyx_args)) {
        case  7: values[6] = PyTuple_GET_ITEM(__pyx_args, 6);
        case  6: values[5] = PyTuple_GET_ITEM(__pyx_args, 5);
        case  5: values[4] = PyTuple_GET_ITEM(__pyx_args, 4);
        case  4: values[3] = PyTuple_GET_ITEM(__pyx_args, 3);
        case  3: values[2] = PyTuple_GET_ITEM(__pyx_args, 2);
        case  2: values[1] = PyTuple_GET_ITEM(__pyx_args, 1);
        case  1: values[0] = PyTuple_GET_ITEM(__pyx_args, 0);
        case  0: break;
        default: goto __pyx_L5_argtuple_error;
      }
      kw_args = PyDict_Size(__pyx_kwds);
      switch (PyTuple_GET_SIZE(__pyx_args)) {
        case  0:
        values[0] = PyDict_GetItem(__pyx_kwds, __pyx_n_s__phi);
        if (likely(values[0])) kw_args--;
        else goto __pyx_L5_argtuple_error;
        case  1:
        values[1] = PyDict_GetItem(__pyx_kwds, __pyx_n_s__N);
        if (likely(values[1])) kw_args--;
        else {
          __Pyx_RaiseArgtupleInvalid("gaborFM_", 0, 4, 7, 1); {__pyx_filename = __pyx_f[0]; __pyx_lineno = 82; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
        }
        case  2:
        values[2] = PyDict_GetItem(__pyx_kwds, __pyx_n_s__omega);
        if (likely(values[2])) kw_args--;
        else {
          __Pyx_RaiseArgtupleInvalid("gaborFM_", 0, 4, 7, 2); {__pyx_filename = __pyx_f[0]; __pyx_lineno = 82; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
        }
        case  3:
        values[3] = PyDict_GetItem(__pyx_kwds, __pyx_n_s__chirp);
        if (likely(values[3])) kw_args--;
        else {
          __Pyx_RaiseArgtupleInvalid("gaborFM_", 0, 4, 7, 3); {__pyx_filename = __pyx_f[0]; __pyx_lineno = 82; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
        }
        case  4:
        if (kw_args > 0) {
          PyObject* value = PyDict_GetItem(__pyx_kwds, __pyx_n_s__omega_m);
          if (value) { values[4] = value; kw_args--; }
        }
        case  5:
        if (kw_args > 0) {
          PyObject* value = PyDict_GetItem(__pyx_kwds, __pyx_n_s__phi_m);
          if (value) { values[5] = value; kw_args--; }
        }
        case  6:
        if (kw_args > 0) {
          PyObject* value = PyDict_GetItem(__pyx_kwds, __pyx_n_s__depth);
          if (value) { values[6] = value; kw_args--; }
        }
      }
      if (unlikely(kw_args > 0)) {
        if (unlikely(__Pyx_ParseOptionalKeywords(__pyx_kwds, __pyx_pyargnames, 0, values, PyTuple_GET_SIZE(__pyx_args), "gaborFM_") < 0)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 82; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
      }
    } else {
      switch (PyTuple_GET_SIZE(__pyx_args)) {
        case  7: values[6] = PyTuple_GET_ITEM(__pyx_args, 6);
        case  6: values[5] = PyTuple_GET_ITEM(__pyx_args, 5);
        case  5: values[4] = PyTuple_GET_ITEM(__pyx_args, 4);
        case  4: values[3] = PyTuple_GET_ITEM(__pyx_args, 3);
        values[2] = PyTuple_GET_ITEM(__pyx_args, 2);
        values[1] = PyTuple_GET_ITEM(__pyx_args, 1);
        values[0] = PyTuple_GET_ITEM(__pyx_args, 0);
        break;
        default: goto __pyx_L5_argtuple_error;
      }
    }
    __pyx_v_phi = __pyx_PyFloat_AsDouble(values[0]); if (unlikely((__pyx_v_phi == (float)-1) && PyErr_Occurred())) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 82; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
    __pyx_v_N = __Pyx_PyInt_AsInt(values[1]); if (unlikely((__pyx_v_N == (int)-1) && PyErr_Occurred())) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 82; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
    __pyx_v_omega = __pyx_PyFloat_AsDouble(values[2]); if (unlikely((__pyx_v_omega == (float)-1) && PyErr_Occurred())) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 82; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
    __pyx_v_chirp = __pyx_PyFloat_AsDouble(values[3]); if (unlikely((__pyx_v_chirp == (float)-1) && PyErr_Occurred())) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 82; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
    if (values[4]) {
      __pyx_v_omega_m = __pyx_PyFloat_AsDouble(values[4]); if (unlikely((__pyx_v_omega_m == (float)-1) && PyErr_Occurred())) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 82; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
    } else {
      __pyx_v_omega_m = ((float)0.);
    }
    if (values[5]) {
      __pyx_v_phi_m = __pyx_PyFloat_AsDouble(values[5]); if (unlikely((__pyx_v_phi_m == (float)-1) && PyErr_Occurred())) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 82; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
    } else {
      __pyx_v_phi_m = ((float)0.);
    }
    if (values[6]) {
      __pyx_v_depth = __pyx_PyFloat_AsDouble(values[6]); if (unlikely((__pyx_v_depth == (float)-1) && PyErr_Occurred())) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 82; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
    } else {
      __pyx_v_depth = ((float)0.);
    }
  }
  goto __pyx_L4_argument_unpacking_done;
  __pyx_L5_argtuple_error:;
  __Pyx_RaiseArgtupleInvalid("gaborFM_", 0, 4, 7, PyTuple_GET_SIZE(__pyx_args)); {__pyx_filename = __pyx_f[0]; __pyx_lineno = 82; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
  __pyx_L3_error:;
  __Pyx_AddTraceback("atom_.gaborFM_", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __Pyx_RefNannyFinishContext();
  return NULL;
  __pyx_L4_argument_unpacking_done:;
  __Pyx_XDECREF(__pyx_r);
  __pyx_t_2.__pyx_n = 3;
  __pyx_t_2.omega_m = __pyx_v_omega_m;
  __pyx_t_2.phi_m = __pyx_v_phi_m;
  __pyx_t_2.depth = __pyx_v_depth;
  __pyx_t_1 = ((PyObject *)__pyx_f_5atom__gaborFM_(__pyx_v_phi, __pyx_v_N, __pyx_v_omega, __pyx_v_chirp, 0, &__pyx_t_2)); if (unlikely(!__pyx_t_1)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 82; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(__pyx_t_1);
  __pyx_r = __pyx_t_1;
  __pyx_t_1 = 0;
  goto __pyx_L0;

  __pyx_r = Py_None; __Pyx_INCREF(Py_None);
  goto __pyx_L0;
  __pyx_L1_error:;
  __Pyx_XDECREF(__pyx_t_1);
  __Pyx_AddTraceback("atom_.gaborFM_", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __pyx_r = NULL;
  __pyx_L0:;
  __Pyx_XGIVEREF(__pyx_r);
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

/* "atom_.pyx":100
 * @cython.profile(False)
 * @cython.boundscheck(False)
 * cpdef np.ndarray[np.float_t, ndim=1, mode="c"] hann_(float phi, int N, float omega, float chirp):             # <<<<<<<<<<<<<<
 *     '''A hann atom where
 *        phi := initial phase
 */

static PyObject *__pyx_pf_5atom__3hann_(PyObject *__pyx_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
static PyArrayObject *__pyx_f_5atom__hann_(float __pyx_v_phi, int __pyx_v_N, float __pyx_v_omega, float __pyx_v_chirp, int __pyx_skip_dispatch) {
  PyArrayObject *__pyx_v_out = 0;
  int __pyx_v_i;
  Py_buffer __pyx_bstruct_out;
  Py_ssize_t __pyx_bstride_0_out = 0;
  Py_ssize_t __pyx_bshape_0_out = 0;
  PyArrayObject *__pyx_r = NULL;
  __Pyx_RefNannyDeclarations
  PyObject *__pyx_t_1 = NULL;
  PyObject *__pyx_t_2 = NULL;
  PyObject *__pyx_t_3 = NULL;
  PyObject *__pyx_t_4 = NULL;
  PyArrayObject *__pyx_t_5 = NULL;
  int __pyx_t_6;
  int __pyx_t_7;
  int __pyx_t_8;
  int __pyx_lineno = 0;
  const char *__pyx_filename = NULL;
  int __pyx_clineno = 0;
  __Pyx_RefNannySetupContext("hann_");
  __pyx_bstruct_out.buf = NULL;

  /* "atom_.pyx":106
 *        omega := normalized frequency'''
 * 
 *     cdef np.ndarray[np.float_t, ndim=1, mode="c"] out = np.zeros(N, dtype=float)             # <<<<<<<<<<<<<<
 *     cdef int i
 * 
 */
  __pyx_t_1 = __Pyx_GetName(__pyx_m, __pyx_n_s__np); if (unlikely(!__pyx_t_1)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 106; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(__pyx_t_1);
  __pyx_t_2 = PyObject_GetAttr(__pyx_t_1, __pyx_n_s__zeros); if (unlikely(!__pyx_t_2)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 106; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(__pyx_t_2);
  __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;
  __pyx_t_1 = PyInt_FromLong(__pyx_v_N); if (unlikely(!__pyx_t_1)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 106; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(__pyx_t_1);
  __pyx_t_3 = PyTuple_New(1); if (unlikely(!__pyx_t_3)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 106; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(((PyObject *)__pyx_t_3));
  PyTuple_SET_ITEM(__pyx_t_3, 0, __pyx_t_1);
  __Pyx_GIVEREF(__pyx_t_1);
  __pyx_t_1 = 0;
  __pyx_t_1 = PyDict_New(); if (unlikely(!__pyx_t_1)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 106; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(((PyObject *)__pyx_t_1));
  if (PyDict_SetItem(__pyx_t_1, ((PyObject *)__pyx_n_s__dtype), ((PyObject *)((PyObject*)(&PyFloat_Type)))) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 106; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __pyx_t_4 = PyEval_CallObjectWithKeywords(__pyx_t_2, ((PyObject *)__pyx_t_3), ((PyObject *)__pyx_t_1)); if (unlikely(!__pyx_t_4)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 106; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(__pyx_t_4);
  __Pyx_DECREF(__pyx_t_2); __pyx_t_2 = 0;
  __Pyx_DECREF(((PyObject *)__pyx_t_3)); __pyx_t_3 = 0;
  __Pyx_DECREF(((PyObject *)__pyx_t_1)); __pyx_t_1 = 0;
  if (!(likely(((__pyx_t_4) == Py_None) || likely(__Pyx_TypeTest(__pyx_t_4, __pyx_ptype_5numpy_ndarray))))) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 106; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __pyx_t_5 = ((PyArrayObject *)__pyx_t_4);
  {
    __Pyx_BufFmt_StackElem __pyx_stack[1];
    if (unlikely(__Pyx_GetBufferAndValidate(&__pyx_bstruct_out, (PyObject*)__pyx_t_5, &__Pyx_TypeInfo_nn___pyx_t_5numpy_float_t, PyBUF_FORMAT| PyBUF_C_CONTIGUOUS| PyBUF_WRITABLE, 1, 0, __pyx_stack) == -1)) {
      __pyx_v_out = ((PyArrayObject *)Py_None); __Pyx_INCREF(Py_None); __pyx_bstruct_out.buf = NULL;
      {__pyx_filename = __pyx_f[0]; __pyx_lineno = 106; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
    } else {__pyx_bstride_0_out = __pyx_bstruct_out.strides[0];
      __pyx_bshape_0_out = __pyx_bstruct_out.shape[0];
    }
  }
  __pyx_t_5 = 0;
  __pyx_v_out = ((PyArrayObject *)__pyx_t_4);
  __pyx_t_4 = 0;

  /* "atom_.pyx":109
 *     cdef int i
 * 
 *     for i in range(N):             # <<<<<<<<<<<<<<
 *         out[i] = hann(i, N) * realSinusoid(i, omega, chirp/N, phi)
 * 
 */
  __pyx_t_6 = __pyx_v_N;
  for (__pyx_t_7 = 0; __pyx_t_7 < __pyx_t_6; __pyx_t_7+=1) {
    __pyx_v_i = __pyx_t_7;

    /* "atom_.pyx":110
 * 
 *     for i in range(N):
 *         out[i] = hann(i, N) * realSinusoid(i, omega, chirp/N, phi)             # <<<<<<<<<<<<<<
 * 
 *     return out
 */
    if (unlikely(__pyx_v_N == 0)) {
      PyErr_Format(PyExc_ZeroDivisionError, "float division");
      {__pyx_filename = __pyx_f[0]; __pyx_lineno = 110; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
    }
    __pyx_t_8 = __pyx_v_i;
    if (__pyx_t_8 < 0) __pyx_t_8 += __pyx_bshape_0_out;
    *__Pyx_BufPtrCContig1d(__pyx_t_5numpy_float_t *, __pyx_bstruct_out.buf, __pyx_t_8, __pyx_bstride_0_out) = (__pyx_f_5atom__hann(__pyx_v_i, __pyx_v_N) * __pyx_f_5atom__realSinusoid(__pyx_v_i, __pyx_v_omega, (__pyx_v_chirp / __pyx_v_N), __pyx_v_phi));
  }

  /* "atom_.pyx":112
 *         out[i] = hann(i, N) * realSinusoid(i, omega, chirp/N, phi)
 * 
 *     return out             # <<<<<<<<<<<<<<
 * 
 * #HannFM for python
 */
  __Pyx_XDECREF(((PyObject *)__pyx_r));
  __Pyx_INCREF(((PyObject *)__pyx_v_out));
  __pyx_r = ((PyArrayObject *)__pyx_v_out);
  goto __pyx_L0;

  __pyx_r = ((PyArrayObject *)Py_None); __Pyx_INCREF(Py_None);
  goto __pyx_L0;
  __pyx_L1_error:;
  __Pyx_XDECREF(__pyx_t_1);
  __Pyx_XDECREF(__pyx_t_2);
  __Pyx_XDECREF(__pyx_t_3);
  __Pyx_XDECREF(__pyx_t_4);
  { PyObject *__pyx_type, *__pyx_value, *__pyx_tb;
    __Pyx_ErrFetch(&__pyx_type, &__pyx_value, &__pyx_tb);
    __Pyx_SafeReleaseBuffer(&__pyx_bstruct_out);
  __Pyx_ErrRestore(__pyx_type, __pyx_value, __pyx_tb);}
  __Pyx_AddTraceback("atom_.hann_", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __pyx_r = 0;
  goto __pyx_L2;
  __pyx_L0:;
  __Pyx_SafeReleaseBuffer(&__pyx_bstruct_out);
  __pyx_L2:;
  __Pyx_XDECREF((PyObject *)__pyx_v_out);
  __Pyx_XGIVEREF((PyObject *)__pyx_r);
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

/* "atom_.pyx":100
 * @cython.profile(False)
 * @cython.boundscheck(False)
 * cpdef np.ndarray[np.float_t, ndim=1, mode="c"] hann_(float phi, int N, float omega, float chirp):             # <<<<<<<<<<<<<<
 *     '''A hann atom where
 *        phi := initial phase
 */

static PyObject *__pyx_pf_5atom__3hann_(PyObject *__pyx_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
static char __pyx_doc_5atom__3hann_[] = "A hann atom where\n       phi := initial phase\n       N := scale, i.e. length\n       omega := normalized frequency";
static PyObject *__pyx_pf_5atom__3hann_(PyObject *__pyx_self, PyObject *__pyx_args, PyObject *__pyx_kwds) {
  float __pyx_v_phi;
  int __pyx_v_N;
  float __pyx_v_omega;
  float __pyx_v_chirp;
  PyObject *__pyx_r = NULL;
  __Pyx_RefNannyDeclarations
  PyObject *__pyx_t_1 = NULL;
  int __pyx_lineno = 0;
  const char *__pyx_filename = NULL;
  int __pyx_clineno = 0;
  static PyObject **__pyx_pyargnames[] = {&__pyx_n_s__phi,&__pyx_n_s__N,&__pyx_n_s__omega,&__pyx_n_s__chirp,0};
  __Pyx_RefNannySetupContext("hann_");
  __pyx_self = __pyx_self;
  {
    PyObject* values[4] = {0,0,0,0};
    if (unlikely(__pyx_kwds)) {
      Py_ssize_t kw_args;
      switch (PyTuple_GET_SIZE(__pyx_args)) {
        case  4: values[3] = PyTuple_GET_ITEM(__pyx_args, 3);
        case  3: values[2] = PyTuple_GET_ITEM(__pyx_args, 2);
        case  2: values[1] = PyTuple_GET_ITEM(__pyx_args, 1);
        case  1: values[0] = PyTuple_GET_ITEM(__pyx_args, 0);
        case  0: break;
        default: goto __pyx_L5_argtuple_error;
      }
      kw_args = PyDict_Size(__pyx_kwds);
      switch (PyTuple_GET_SIZE(__pyx_args)) {
        case  0:
        values[0] = PyDict_GetItem(__pyx_kwds, __pyx_n_s__phi);
        if (likely(values[0])) kw_args--;
        else goto __pyx_L5_argtuple_error;
        case  1:
        values[1] = PyDict_GetItem(__pyx_kwds, __pyx_n_s__N);
        if (likely(values[1])) kw_args--;
        else {
          __Pyx_RaiseArgtupleInvalid("hann_", 1, 4, 4, 1); {__pyx_filename = __pyx_f[0]; __pyx_lineno = 100; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
        }
        case  2:
        values[2] = PyDict_GetItem(__pyx_kwds, __pyx_n_s__omega);
        if (likely(values[2])) kw_args--;
        else {
          __Pyx_RaiseArgtupleInvalid("hann_", 1, 4, 4, 2); {__pyx_filename = __pyx_f[0]; __pyx_lineno = 100; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
        }
        case  3:
        values[3] = PyDict_GetItem(__pyx_kwds, __pyx_n_s__chirp);
        if (likely(values[3])) kw_args--;
        else {
          __Pyx_RaiseArgtupleInvalid("hann_", 1, 4, 4, 3); {__pyx_filename = __pyx_f[0]; __pyx_lineno = 100; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
        }
      }
      if (unlikely(kw_args > 0)) {
        if (unlikely(__Pyx_ParseOptionalKeywords(__pyx_kwds, __pyx_pyargnames, 0, values, PyTuple_GET_SIZE(__pyx_args), "hann_") < 0)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 100; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
      }
    } else if (PyTuple_GET_SIZE(__pyx_args) != 4) {
      goto __pyx_L5_argtuple_error;
    } else {
      values[0] = PyTuple_GET_ITEM(__pyx_args, 0);
      values[1] = PyTuple_GET_ITEM(__pyx_args, 1);
      values[2] = PyTuple_GET_ITEM(__pyx_args, 2);
      values[3] = PyTuple_GET_ITEM(__pyx_args, 3);
    }
    __pyx_v_phi = __pyx_PyFloat_AsDouble(values[0]); if (unlikely((__pyx_v_phi == (float)-1) && PyErr_Occurred())) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 100; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
    __pyx_v_N = __Pyx_PyInt_AsInt(values[1]); if (unlikely((__pyx_v_N == (int)-1) && PyErr_Occurred())) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 100; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
    __pyx_v_omega = __pyx_PyFloat_AsDouble(values[2]); if (unlikely((__pyx_v_omega == (float)-1) && PyErr_Occurred())) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 100; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
    __pyx_v_chirp = __pyx_PyFloat_AsDouble(values[3]); if (unlikely((__pyx_v_chirp == (float)-1) && PyErr_Occurred())) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 100; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
  }
  goto __pyx_L4_argument_unpacking_done;
  __pyx_L5_argtuple_error:;
  __Pyx_RaiseArgtupleInvalid("hann_", 1, 4, 4, PyTuple_GET_SIZE(__pyx_args)); {__pyx_filename = __pyx_f[0]; __pyx_lineno = 100; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
  __pyx_L3_error:;
  __Pyx_AddTraceback("atom_.hann_", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __Pyx_RefNannyFinishContext();
  return NULL;
  __pyx_L4_argument_unpacking_done:;
  __Pyx_XDECREF(__pyx_r);
  __pyx_t_1 = ((PyObject *)__pyx_f_5atom__hann_(__pyx_v_phi, __pyx_v_N, __pyx_v_omega, __pyx_v_chirp, 0)); if (unlikely(!__pyx_t_1)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 100; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(__pyx_t_1);
  __pyx_r = __pyx_t_1;
  __pyx_t_1 = 0;
  goto __pyx_L0;

  __pyx_r = Py_None; __Pyx_INCREF(Py_None);
  goto __pyx_L0;
  __pyx_L1_error:;
  __Pyx_XDECREF(__pyx_t_1);
  __Pyx_AddTraceback("atom_.hann_", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __pyx_r = NULL;
  __pyx_L0:;
  __Pyx_XGIVEREF(__pyx_r);
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

/* "atom_.pyx":118
 * @cython.profile(False)
 * @cython.boundscheck(False)
 * cpdef np.ndarray[np.float_t, ndim=1, mode="c"] hannFM_(float phi, int N, float omega, float chirp, float omega_m=0., float phi_m=0., float depth=0.):             # <<<<<<<<<<<<<<
 *     '''A hann atom where
 *        phi := initial phase
 */

static PyObject *__pyx_pf_5atom__4hannFM_(PyObject *__pyx_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
static PyArrayObject *__pyx_f_5atom__hannFM_(float __pyx_v_phi, int __pyx_v_N, float __pyx_v_omega, float __pyx_v_chirp, int __pyx_skip_dispatch, struct __pyx_opt_args_5atom__hannFM_ *__pyx_optional_args) {
  float __pyx_v_omega_m = ((float)0.);
  float __pyx_v_phi_m = ((float)0.);
  float __pyx_v_depth = ((float)0.);
  PyArrayObject *__pyx_v_out = 0;
  int __pyx_v_i;
  Py_buffer __pyx_bstruct_out;
  Py_ssize_t __pyx_bstride_0_out = 0;
  Py_ssize_t __pyx_bshape_0_out = 0;
  PyArrayObject *__pyx_r = NULL;
  __Pyx_RefNannyDeclarations
  PyObject *__pyx_t_1 = NULL;
  PyObject *__pyx_t_2 = NULL;
  PyObject *__pyx_t_3 = NULL;
  PyObject *__pyx_t_4 = NULL;
  PyArrayObject *__pyx_t_5 = NULL;
  int __pyx_t_6;
  int __pyx_t_7;
  int __pyx_t_8;
  int __pyx_lineno = 0;
  const char *__pyx_filename = NULL;
  int __pyx_clineno = 0;
  __Pyx_RefNannySetupContext("hannFM_");
  if (__pyx_optional_args) {
    if (__pyx_optional_args->__pyx_n > 0) {
      __pyx_v_omega_m = __pyx_optional_args->omega_m;
      if (__pyx_optional_args->__pyx_n > 1) {
        __pyx_v_phi_m = __pyx_optional_args->phi_m;
        if (__pyx_optional_args->__pyx_n > 2) {
          __pyx_v_depth = __pyx_optional_args->depth;
        }
      }
    }
  }
  __pyx_bstruct_out.buf = NULL;

  /* "atom_.pyx":124
 *        omega := normalized frequency'''
 * 
 *     cdef np.ndarray[np.float_t, ndim=1, mode="c"] out = np.zeros(N, dtype=float)             # <<<<<<<<<<<<<<
 *     cdef int i
 * 
 */
  __pyx_t_1 = __Pyx_GetName(__pyx_m, __pyx_n_s__np); if (unlikely(!__pyx_t_1)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 124; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(__pyx_t_1);
  __pyx_t_2 = PyObject_GetAttr(__pyx_t_1, __pyx_n_s__zeros); if (unlikely(!__pyx_t_2)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 124; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(__pyx_t_2);
  __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;
  __pyx_t_1 = PyInt_FromLong(__pyx_v_N); if (unlikely(!__pyx_t_1)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 124; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(__pyx_t_1);
  __pyx_t_3 = PyTuple_New(1); if (unlikely(!__pyx_t_3)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 124; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(((PyObject *)__pyx_t_3));
  PyTuple_SET_ITEM(__pyx_t_3, 0, __pyx_t_1);
  __Pyx_GIVEREF(__pyx_t_1);
  __pyx_t_1 = 0;
  __pyx_t_1 = PyDict_New(); if (unlikely(!__pyx_t_1)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 124; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(((PyObject *)__pyx_t_1));
  if (PyDict_SetItem(__pyx_t_1, ((PyObject *)__pyx_n_s__dtype), ((PyObject *)((PyObject*)(&PyFloat_Type)))) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 124; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __pyx_t_4 = PyEval_CallObjectWithKeywords(__pyx_t_2, ((PyObject *)__pyx_t_3), ((PyObject *)__pyx_t_1)); if (unlikely(!__pyx_t_4)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 124; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(__pyx_t_4);
  __Pyx_DECREF(__pyx_t_2); __pyx_t_2 = 0;
  __Pyx_DECREF(((PyObject *)__pyx_t_3)); __pyx_t_3 = 0;
  __Pyx_DECREF(((PyObject *)__pyx_t_1)); __pyx_t_1 = 0;
  if (!(likely(((__pyx_t_4) == Py_None) || likely(__Pyx_TypeTest(__pyx_t_4, __pyx_ptype_5numpy_ndarray))))) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 124; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __pyx_t_5 = ((PyArrayObject *)__pyx_t_4);
  {
    __Pyx_BufFmt_StackElem __pyx_stack[1];
    if (unlikely(__Pyx_GetBufferAndValidate(&__pyx_bstruct_out, (PyObject*)__pyx_t_5, &__Pyx_TypeInfo_nn___pyx_t_5numpy_float_t, PyBUF_FORMAT| PyBUF_C_CONTIGUOUS| PyBUF_WRITABLE, 1, 0, __pyx_stack) == -1)) {
      __pyx_v_out = ((PyArrayObject *)Py_None); __Pyx_INCREF(Py_None); __pyx_bstruct_out.buf = NULL;
      {__pyx_filename = __pyx_f[0]; __pyx_lineno = 124; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
    } else {__pyx_bstride_0_out = __pyx_bstruct_out.strides[0];
      __pyx_bshape_0_out = __pyx_bstruct_out.shape[0];
    }
  }
  __pyx_t_5 = 0;
  __pyx_v_out = ((PyArrayObject *)__pyx_t_4);
  __pyx_t_4 = 0;

  /* "atom_.pyx":127
 *     cdef int i
 * 
 *     for i in range(N):             # <<<<<<<<<<<<<<
 *         out[i] = hann(i, N) * realSinusoidFM(i, omega, chirp, phi, omega_m, phi_m, depth)
 * 
 */
  __pyx_t_6 = __pyx_v_N;
  for (__pyx_t_7 = 0; __pyx_t_7 < __pyx_t_6; __pyx_t_7+=1) {
    __pyx_v_i = __pyx_t_7;

    /* "atom_.pyx":128
 * 
 *     for i in range(N):
 *         out[i] = hann(i, N) * realSinusoidFM(i, omega, chirp, phi, omega_m, phi_m, depth)             # <<<<<<<<<<<<<<
 * 
 *     return out
 */
    __pyx_t_4 = PyFloat_FromDouble(__pyx_v_depth); if (unlikely(!__pyx_t_4)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 128; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
    __Pyx_GOTREF(__pyx_t_4);
    __pyx_t_8 = __pyx_v_i;
    if (__pyx_t_8 < 0) __pyx_t_8 += __pyx_bshape_0_out;
    *__Pyx_BufPtrCContig1d(__pyx_t_5numpy_float_t *, __pyx_bstruct_out.buf, __pyx_t_8, __pyx_bstride_0_out) = (__pyx_f_5atom__hann(__pyx_v_i, __pyx_v_N) * __pyx_f_5atom__realSinusoidFM(__pyx_v_i, __pyx_v_omega, __pyx_v_chirp, __pyx_v_phi, __pyx_v_omega_m, __pyx_v_phi_m, __pyx_t_4));
    __Pyx_DECREF(__pyx_t_4); __pyx_t_4 = 0;
  }

  /* "atom_.pyx":130
 *         out[i] = hann(i, N) * realSinusoidFM(i, omega, chirp, phi, omega_m, phi_m, depth)
 * 
 *     return out             # <<<<<<<<<<<<<<
 * 
 * #Blackman for python
 */
  __Pyx_XDECREF(((PyObject *)__pyx_r));
  __Pyx_INCREF(((PyObject *)__pyx_v_out));
  __pyx_r = ((PyArrayObject *)__pyx_v_out);
  goto __pyx_L0;

  __pyx_r = ((PyArrayObject *)Py_None); __Pyx_INCREF(Py_None);
  goto __pyx_L0;
  __pyx_L1_error:;
  __Pyx_XDECREF(__pyx_t_1);
  __Pyx_XDECREF(__pyx_t_2);
  __Pyx_XDECREF(__pyx_t_3);
  __Pyx_XDECREF(__pyx_t_4);
  { PyObject *__pyx_type, *__pyx_value, *__pyx_tb;
    __Pyx_ErrFetch(&__pyx_type, &__pyx_value, &__pyx_tb);
    __Pyx_SafeReleaseBuffer(&__pyx_bstruct_out);
  __Pyx_ErrRestore(__pyx_type, __pyx_value, __pyx_tb);}
  __Pyx_AddTraceback("atom_.hannFM_", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __pyx_r = 0;
  goto __pyx_L2;
  __pyx_L0:;
  __Pyx_SafeReleaseBuffer(&__pyx_bstruct_out);
  __pyx_L2:;
  __Pyx_XDECREF((PyObject *)__pyx_v_out);
  __Pyx_XGIVEREF((PyObject *)__pyx_r);
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

/* "atom_.pyx":118
 * @cython.profile(False)
 * @cython.boundscheck(False)
 * cpdef np.ndarray[np.float_t, ndim=1, mode="c"] hannFM_(float phi, int N, float omega, float chirp, float omega_m=0., float phi_m=0., float depth=0.):             # <<<<<<<<<<<<<<
 *     '''A hann atom where
 *        phi := initial phase
 */

static PyObject *__pyx_pf_5atom__4hannFM_(PyObject *__pyx_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
static char __pyx_doc_5atom__4hannFM_[] = "A hann atom where\n       phi := initial phase\n       N := scale, i.e. length\n       omega := normalized frequency";
static PyObject *__pyx_pf_5atom__4hannFM_(PyObject *__pyx_self, PyObject *__pyx_args, PyObject *__pyx_kwds) {
  float __pyx_v_phi;
  int __pyx_v_N;
  float __pyx_v_omega;
  float __pyx_v_chirp;
  float __pyx_v_omega_m;
  float __pyx_v_phi_m;
  float __pyx_v_depth;
  PyObject *__pyx_r = NULL;
  __Pyx_RefNannyDeclarations
  PyObject *__pyx_t_1 = NULL;
  struct __pyx_opt_args_5atom__hannFM_ __pyx_t_2;
  int __pyx_lineno = 0;
  const char *__pyx_filename = NULL;
  int __pyx_clineno = 0;
  static PyObject **__pyx_pyargnames[] = {&__pyx_n_s__phi,&__pyx_n_s__N,&__pyx_n_s__omega,&__pyx_n_s__chirp,&__pyx_n_s__omega_m,&__pyx_n_s__phi_m,&__pyx_n_s__depth,0};
  __Pyx_RefNannySetupContext("hannFM_");
  __pyx_self = __pyx_self;
  {
    PyObject* values[7] = {0,0,0,0,0,0,0};
    if (unlikely(__pyx_kwds)) {
      Py_ssize_t kw_args;
      switch (PyTuple_GET_SIZE(__pyx_args)) {
        case  7: values[6] = PyTuple_GET_ITEM(__pyx_args, 6);
        case  6: values[5] = PyTuple_GET_ITEM(__pyx_args, 5);
        case  5: values[4] = PyTuple_GET_ITEM(__pyx_args, 4);
        case  4: values[3] = PyTuple_GET_ITEM(__pyx_args, 3);
        case  3: values[2] = PyTuple_GET_ITEM(__pyx_args, 2);
        case  2: values[1] = PyTuple_GET_ITEM(__pyx_args, 1);
        case  1: values[0] = PyTuple_GET_ITEM(__pyx_args, 0);
        case  0: break;
        default: goto __pyx_L5_argtuple_error;
      }
      kw_args = PyDict_Size(__pyx_kwds);
      switch (PyTuple_GET_SIZE(__pyx_args)) {
        case  0:
        values[0] = PyDict_GetItem(__pyx_kwds, __pyx_n_s__phi);
        if (likely(values[0])) kw_args--;
        else goto __pyx_L5_argtuple_error;
        case  1:
        values[1] = PyDict_GetItem(__pyx_kwds, __pyx_n_s__N);
        if (likely(values[1])) kw_args--;
        else {
          __Pyx_RaiseArgtupleInvalid("hannFM_", 0, 4, 7, 1); {__pyx_filename = __pyx_f[0]; __pyx_lineno = 118; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
        }
        case  2:
        values[2] = PyDict_GetItem(__pyx_kwds, __pyx_n_s__omega);
        if (likely(values[2])) kw_args--;
        else {
          __Pyx_RaiseArgtupleInvalid("hannFM_", 0, 4, 7, 2); {__pyx_filename = __pyx_f[0]; __pyx_lineno = 118; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
        }
        case  3:
        values[3] = PyDict_GetItem(__pyx_kwds, __pyx_n_s__chirp);
        if (likely(values[3])) kw_args--;
        else {
          __Pyx_RaiseArgtupleInvalid("hannFM_", 0, 4, 7, 3); {__pyx_filename = __pyx_f[0]; __pyx_lineno = 118; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
        }
        case  4:
        if (kw_args > 0) {
          PyObject* value = PyDict_GetItem(__pyx_kwds, __pyx_n_s__omega_m);
          if (value) { values[4] = value; kw_args--; }
        }
        case  5:
        if (kw_args > 0) {
          PyObject* value = PyDict_GetItem(__pyx_kwds, __pyx_n_s__phi_m);
          if (value) { values[5] = value; kw_args--; }
        }
        case  6:
        if (kw_args > 0) {
          PyObject* value = PyDict_GetItem(__pyx_kwds, __pyx_n_s__depth);
          if (value) { values[6] = value; kw_args--; }
        }
      }
      if (unlikely(kw_args > 0)) {
        if (unlikely(__Pyx_ParseOptionalKeywords(__pyx_kwds, __pyx_pyargnames, 0, values, PyTuple_GET_SIZE(__pyx_args), "hannFM_") < 0)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 118; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
      }
    } else {
      switch (PyTuple_GET_SIZE(__pyx_args)) {
        case  7: values[6] = PyTuple_GET_ITEM(__pyx_args, 6);
        case  6: values[5] = PyTuple_GET_ITEM(__pyx_args, 5);
        case  5: values[4] = PyTuple_GET_ITEM(__pyx_args, 4);
        case  4: values[3] = PyTuple_GET_ITEM(__pyx_args, 3);
        values[2] = PyTuple_GET_ITEM(__pyx_args, 2);
        values[1] = PyTuple_GET_ITEM(__pyx_args, 1);
        values[0] = PyTuple_GET_ITEM(__pyx_args, 0);
        break;
        default: goto __pyx_L5_argtuple_error;
      }
    }
    __pyx_v_phi = __pyx_PyFloat_AsDouble(values[0]); if (unlikely((__pyx_v_phi == (float)-1) && PyErr_Occurred())) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 118; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
    __pyx_v_N = __Pyx_PyInt_AsInt(values[1]); if (unlikely((__pyx_v_N == (int)-1) && PyErr_Occurred())) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 118; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
    __pyx_v_omega = __pyx_PyFloat_AsDouble(values[2]); if (unlikely((__pyx_v_omega == (float)-1) && PyErr_Occurred())) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 118; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
    __pyx_v_chirp = __pyx_PyFloat_AsDouble(values[3]); if (unlikely((__pyx_v_chirp == (float)-1) && PyErr_Occurred())) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 118; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
    if (values[4]) {
      __pyx_v_omega_m = __pyx_PyFloat_AsDouble(values[4]); if (unlikely((__pyx_v_omega_m == (float)-1) && PyErr_Occurred())) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 118; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
    } else {
      __pyx_v_omega_m = ((float)0.);
    }
    if (values[5]) {
      __pyx_v_phi_m = __pyx_PyFloat_AsDouble(values[5]); if (unlikely((__pyx_v_phi_m == (float)-1) && PyErr_Occurred())) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 118; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
    } else {
      __pyx_v_phi_m = ((float)0.);
    }
    if (values[6]) {
      __pyx_v_depth = __pyx_PyFloat_AsDouble(values[6]); if (unlikely((__pyx_v_depth == (float)-1) && PyErr_Occurred())) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 118; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
    } else {
      __pyx_v_depth = ((float)0.);
    }
  }
  goto __pyx_L4_argument_unpacking_done;
  __pyx_L5_argtuple_error:;
  __Pyx_RaiseArgtupleInvalid("hannFM_", 0, 4, 7, PyTuple_GET_SIZE(__pyx_args)); {__pyx_filename = __pyx_f[0]; __pyx_lineno = 118; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
  __pyx_L3_error:;
  __Pyx_AddTraceback("atom_.hannFM_", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __Pyx_RefNannyFinishContext();
  return NULL;
  __pyx_L4_argument_unpacking_done:;
  __Pyx_XDECREF(__pyx_r);
  __pyx_t_2.__pyx_n = 3;
  __pyx_t_2.omega_m = __pyx_v_omega_m;
  __pyx_t_2.phi_m = __pyx_v_phi_m;
  __pyx_t_2.depth = __pyx_v_depth;
  __pyx_t_1 = ((PyObject *)__pyx_f_5atom__hannFM_(__pyx_v_phi, __pyx_v_N, __pyx_v_omega, __pyx_v_chirp, 0, &__pyx_t_2)); if (unlikely(!__pyx_t_1)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 118; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(__pyx_t_1);
  __pyx_r = __pyx_t_1;
  __pyx_t_1 = 0;
  goto __pyx_L0;

  __pyx_r = Py_None; __Pyx_INCREF(Py_None);
  goto __pyx_L0;
  __pyx_L1_error:;
  __Pyx_XDECREF(__pyx_t_1);
  __Pyx_AddTraceback("atom_.hannFM_", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __pyx_r = NULL;
  __pyx_L0:;
  __Pyx_XGIVEREF(__pyx_r);
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

/* "atom_.pyx":136
 * @cython.profile(False)
 * @cython.boundscheck(False)
 * cpdef np.ndarray[np.float_t, ndim=1, mode="c"] blackman_(float phi, int N, float omega, float chirp):             # <<<<<<<<<<<<<<
 *     '''A blackman atom where
 *        phi := initial phase
 */

static PyObject *__pyx_pf_5atom__5blackman_(PyObject *__pyx_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
static PyArrayObject *__pyx_f_5atom__blackman_(float __pyx_v_phi, int __pyx_v_N, float __pyx_v_omega, float __pyx_v_chirp, int __pyx_skip_dispatch) {
  PyArrayObject *__pyx_v_out = 0;
  int __pyx_v_i;
  Py_buffer __pyx_bstruct_out;
  Py_ssize_t __pyx_bstride_0_out = 0;
  Py_ssize_t __pyx_bshape_0_out = 0;
  PyArrayObject *__pyx_r = NULL;
  __Pyx_RefNannyDeclarations
  PyObject *__pyx_t_1 = NULL;
  PyObject *__pyx_t_2 = NULL;
  PyObject *__pyx_t_3 = NULL;
  PyObject *__pyx_t_4 = NULL;
  PyArrayObject *__pyx_t_5 = NULL;
  int __pyx_t_6;
  int __pyx_t_7;
  int __pyx_t_8;
  int __pyx_lineno = 0;
  const char *__pyx_filename = NULL;
  int __pyx_clineno = 0;
  __Pyx_RefNannySetupContext("blackman_");
  __pyx_bstruct_out.buf = NULL;

  /* "atom_.pyx":142
 *        omega := normalized frequency'''
 * 
 *     cdef np.ndarray[np.float_t, ndim=1, mode="c"] out = np.zeros(N, dtype=float)             # <<<<<<<<<<<<<<
 *     cdef int i
 * 
 */
  __pyx_t_1 = __Pyx_GetName(__pyx_m, __pyx_n_s__np); if (unlikely(!__pyx_t_1)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 142; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(__pyx_t_1);
  __pyx_t_2 = PyObject_GetAttr(__pyx_t_1, __pyx_n_s__zeros); if (unlikely(!__pyx_t_2)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 142; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(__pyx_t_2);
  __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;
  __pyx_t_1 = PyInt_FromLong(__pyx_v_N); if (unlikely(!__pyx_t_1)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 142; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(__pyx_t_1);
  __pyx_t_3 = PyTuple_New(1); if (unlikely(!__pyx_t_3)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 142; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(((PyObject *)__pyx_t_3));
  PyTuple_SET_ITEM(__pyx_t_3, 0, __pyx_t_1);
  __Pyx_GIVEREF(__pyx_t_1);
  __pyx_t_1 = 0;
  __pyx_t_1 = PyDict_New(); if (unlikely(!__pyx_t_1)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 142; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(((PyObject *)__pyx_t_1));
  if (PyDict_SetItem(__pyx_t_1, ((PyObject *)__pyx_n_s__dtype), ((PyObject *)((PyObject*)(&PyFloat_Type)))) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 142; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __pyx_t_4 = PyEval_CallObjectWithKeywords(__pyx_t_2, ((PyObject *)__pyx_t_3), ((PyObject *)__pyx_t_1)); if (unlikely(!__pyx_t_4)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 142; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(__pyx_t_4);
  __Pyx_DECREF(__pyx_t_2); __pyx_t_2 = 0;
  __Pyx_DECREF(((PyObject *)__pyx_t_3)); __pyx_t_3 = 0;
  __Pyx_DECREF(((PyObject *)__pyx_t_1)); __pyx_t_1 = 0;
  if (!(likely(((__pyx_t_4) == Py_None) || likely(__Pyx_TypeTest(__pyx_t_4, __pyx_ptype_5numpy_ndarray))))) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 142; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __pyx_t_5 = ((PyArrayObject *)__pyx_t_4);
  {
    __Pyx_BufFmt_StackElem __pyx_stack[1];
    if (unlikely(__Pyx_GetBufferAndValidate(&__pyx_bstruct_out, (PyObject*)__pyx_t_5, &__Pyx_TypeInfo_nn___pyx_t_5numpy_float_t, PyBUF_FORMAT| PyBUF_C_CONTIGUOUS| PyBUF_WRITABLE, 1, 0, __pyx_stack) == -1)) {
      __pyx_v_out = ((PyArrayObject *)Py_None); __Pyx_INCREF(Py_None); __pyx_bstruct_out.buf = NULL;
      {__pyx_filename = __pyx_f[0]; __pyx_lineno = 142; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
    } else {__pyx_bstride_0_out = __pyx_bstruct_out.strides[0];
      __pyx_bshape_0_out = __pyx_bstruct_out.shape[0];
    }
  }
  __pyx_t_5 = 0;
  __pyx_v_out = ((PyArrayObject *)__pyx_t_4);
  __pyx_t_4 = 0;

  /* "atom_.pyx":145
 *     cdef int i
 * 
 *     for i in range(N):             # <<<<<<<<<<<<<<
 *         out[i] = blackman(i, N) * realSinusoid(i, omega, chirp/N, phi)
 * 
 */
  __pyx_t_6 = __pyx_v_N;
  for (__pyx_t_7 = 0; __pyx_t_7 < __pyx_t_6; __pyx_t_7+=1) {
    __pyx_v_i = __pyx_t_7;

    /* "atom_.pyx":146
 * 
 *     for i in range(N):
 *         out[i] = blackman(i, N) * realSinusoid(i, omega, chirp/N, phi)             # <<<<<<<<<<<<<<
 * 
 *     return out
 */
    if (unlikely(__pyx_v_N == 0)) {
      PyErr_Format(PyExc_ZeroDivisionError, "float division");
      {__pyx_filename = __pyx_f[0]; __pyx_lineno = 146; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
    }
    __pyx_t_8 = __pyx_v_i;
    if (__pyx_t_8 < 0) __pyx_t_8 += __pyx_bshape_0_out;
    *__Pyx_BufPtrCContig1d(__pyx_t_5numpy_float_t *, __pyx_bstruct_out.buf, __pyx_t_8, __pyx_bstride_0_out) = (__pyx_f_5atom__blackman(__pyx_v_i, __pyx_v_N) * __pyx_f_5atom__realSinusoid(__pyx_v_i, __pyx_v_omega, (__pyx_v_chirp / __pyx_v_N), __pyx_v_phi));
  }

  /* "atom_.pyx":148
 *         out[i] = blackman(i, N) * realSinusoid(i, omega, chirp/N, phi)
 * 
 *     return out             # <<<<<<<<<<<<<<
 * 
 * #BlackmanFM for python
 */
  __Pyx_XDECREF(((PyObject *)__pyx_r));
  __Pyx_INCREF(((PyObject *)__pyx_v_out));
  __pyx_r = ((PyArrayObject *)__pyx_v_out);
  goto __pyx_L0;

  __pyx_r = ((PyArrayObject *)Py_None); __Pyx_INCREF(Py_None);
  goto __pyx_L0;
  __pyx_L1_error:;
  __Pyx_XDECREF(__pyx_t_1);
  __Pyx_XDECREF(__pyx_t_2);
  __Pyx_XDECREF(__pyx_t_3);
  __Pyx_XDECREF(__pyx_t_4);
  { PyObject *__pyx_type, *__pyx_value, *__pyx_tb;
    __Pyx_ErrFetch(&__pyx_type, &__pyx_value, &__pyx_tb);
    __Pyx_SafeReleaseBuffer(&__pyx_bstruct_out);
  __Pyx_ErrRestore(__pyx_type, __pyx_value, __pyx_tb);}
  __Pyx_AddTraceback("atom_.blackman_", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __pyx_r = 0;
  goto __pyx_L2;
  __pyx_L0:;
  __Pyx_SafeReleaseBuffer(&__pyx_bstruct_out);
  __pyx_L2:;
  __Pyx_XDECREF((PyObject *)__pyx_v_out);
  __Pyx_XGIVEREF((PyObject *)__pyx_r);
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

/* "atom_.pyx":136
 * @cython.profile(False)
 * @cython.boundscheck(False)
 * cpdef np.ndarray[np.float_t, ndim=1, mode="c"] blackman_(float phi, int N, float omega, float chirp):             # <<<<<<<<<<<<<<
 *     '''A blackman atom where
 *        phi := initial phase
 */

static PyObject *__pyx_pf_5atom__5blackman_(PyObject *__pyx_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
static char __pyx_doc_5atom__5blackman_[] = "A blackman atom where\n       phi := initial phase\n       N := scale, i.e. length\n       omega := normalized frequency";
static PyObject *__pyx_pf_5atom__5blackman_(PyObject *__pyx_self, PyObject *__pyx_args, PyObject *__pyx_kwds) {
  float __pyx_v_phi;
  int __pyx_v_N;
  float __pyx_v_omega;
  float __pyx_v_chirp;
  PyObject *__pyx_r = NULL;
  __Pyx_RefNannyDeclarations
  PyObject *__pyx_t_1 = NULL;
  int __pyx_lineno = 0;
  const char *__pyx_filename = NULL;
  int __pyx_clineno = 0;
  static PyObject **__pyx_pyargnames[] = {&__pyx_n_s__phi,&__pyx_n_s__N,&__pyx_n_s__omega,&__pyx_n_s__chirp,0};
  __Pyx_RefNannySetupContext("blackman_");
  __pyx_self = __pyx_self;
  {
    PyObject* values[4] = {0,0,0,0};
    if (unlikely(__pyx_kwds)) {
      Py_ssize_t kw_args;
      switch (PyTuple_GET_SIZE(__pyx_args)) {
        case  4: values[3] = PyTuple_GET_ITEM(__pyx_args, 3);
        case  3: values[2] = PyTuple_GET_ITEM(__pyx_args, 2);
        case  2: values[1] = PyTuple_GET_ITEM(__pyx_args, 1);
        case  1: values[0] = PyTuple_GET_ITEM(__pyx_args, 0);
        case  0: break;
        default: goto __pyx_L5_argtuple_error;
      }
      kw_args = PyDict_Size(__pyx_kwds);
      switch (PyTuple_GET_SIZE(__pyx_args)) {
        case  0:
        values[0] = PyDict_GetItem(__pyx_kwds, __pyx_n_s__phi);
        if (likely(values[0])) kw_args--;
        else goto __pyx_L5_argtuple_error;
        case  1:
        values[1] = PyDict_GetItem(__pyx_kwds, __pyx_n_s__N);
        if (likely(values[1])) kw_args--;
        else {
          __Pyx_RaiseArgtupleInvalid("blackman_", 1, 4, 4, 1); {__pyx_filename = __pyx_f[0]; __pyx_lineno = 136; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
        }
        case  2:
        values[2] = PyDict_GetItem(__pyx_kwds, __pyx_n_s__omega);
        if (likely(values[2])) kw_args--;
        else {
          __Pyx_RaiseArgtupleInvalid("blackman_", 1, 4, 4, 2); {__pyx_filename = __pyx_f[0]; __pyx_lineno = 136; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
        }
        case  3:
        values[3] = PyDict_GetItem(__pyx_kwds, __pyx_n_s__chirp);
        if (likely(values[3])) kw_args--;
        else {
          __Pyx_RaiseArgtupleInvalid("blackman_", 1, 4, 4, 3); {__pyx_filename = __pyx_f[0]; __pyx_lineno = 136; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
        }
      }
      if (unlikely(kw_args > 0)) {
        if (unlikely(__Pyx_ParseOptionalKeywords(__pyx_kwds, __pyx_pyargnames, 0, values, PyTuple_GET_SIZE(__pyx_args), "blackman_") < 0)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 136; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
      }
    } else if (PyTuple_GET_SIZE(__pyx_args) != 4) {
      goto __pyx_L5_argtuple_error;
    } else {
      values[0] = PyTuple_GET_ITEM(__pyx_args, 0);
      values[1] = PyTuple_GET_ITEM(__pyx_args, 1);
      values[2] = PyTuple_GET_ITEM(__pyx_args, 2);
      values[3] = PyTuple_GET_ITEM(__pyx_args, 3);
    }
    __pyx_v_phi = __pyx_PyFloat_AsDouble(values[0]); if (unlikely((__pyx_v_phi == (float)-1) && PyErr_Occurred())) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 136; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
    __pyx_v_N = __Pyx_PyInt_AsInt(values[1]); if (unlikely((__pyx_v_N == (int)-1) && PyErr_Occurred())) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 136; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
    __pyx_v_omega = __pyx_PyFloat_AsDouble(values[2]); if (unlikely((__pyx_v_omega == (float)-1) && PyErr_Occurred())) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 136; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
    __pyx_v_chirp = __pyx_PyFloat_AsDouble(values[3]); if (unlikely((__pyx_v_chirp == (float)-1) && PyErr_Occurred())) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 136; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
  }
  goto __pyx_L4_argument_unpacking_done;
  __pyx_L5_argtuple_error:;
  __Pyx_RaiseArgtupleInvalid("blackman_", 1, 4, 4, PyTuple_GET_SIZE(__pyx_args)); {__pyx_filename = __pyx_f[0]; __pyx_lineno = 136; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
  __pyx_L3_error:;
  __Pyx_AddTraceback("atom_.blackman_", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __Pyx_RefNannyFinishContext();
  return NULL;
  __pyx_L4_argument_unpacking_done:;
  __Pyx_XDECREF(__pyx_r);
  __pyx_t_1 = ((PyObject *)__pyx_f_5atom__blackman_(__pyx_v_phi, __pyx_v_N, __pyx_v_omega, __pyx_v_chirp, 0)); if (unlikely(!__pyx_t_1)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 136; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(__pyx_t_1);
  __pyx_r = __pyx_t_1;
  __pyx_t_1 = 0;
  goto __pyx_L0;

  __pyx_r = Py_None; __Pyx_INCREF(Py_None);
  goto __pyx_L0;
  __pyx_L1_error:;
  __Pyx_XDECREF(__pyx_t_1);
  __Pyx_AddTraceback("atom_.blackman_", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __pyx_r = NULL;
  __pyx_L0:;
  __Pyx_XGIVEREF(__pyx_r);
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

/* "atom_.pyx":154
 * @cython.profile(False)
 * @cython.boundscheck(False)
 * cpdef np.ndarray[np.float_t, ndim=1, mode="c"] blackmanFM_(float phi, int N, float omega, float chirp, float omega_m=0., float phi_m=0., float depth=0.):             # <<<<<<<<<<<<<<
 *     '''A blackman FM atom where
 *        phi := initial phase
 */

static PyObject *__pyx_pf_5atom__6blackmanFM_(PyObject *__pyx_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
static PyArrayObject *__pyx_f_5atom__blackmanFM_(float __pyx_v_phi, int __pyx_v_N, float __pyx_v_omega, float __pyx_v_chirp, int __pyx_skip_dispatch, struct __pyx_opt_args_5atom__blackmanFM_ *__pyx_optional_args) {
  float __pyx_v_omega_m = ((float)0.);
  float __pyx_v_phi_m = ((float)0.);
  float __pyx_v_depth = ((float)0.);
  PyArrayObject *__pyx_v_out = 0;
  int __pyx_v_i;
  Py_buffer __pyx_bstruct_out;
  Py_ssize_t __pyx_bstride_0_out = 0;
  Py_ssize_t __pyx_bshape_0_out = 0;
  PyArrayObject *__pyx_r = NULL;
  __Pyx_RefNannyDeclarations
  PyObject *__pyx_t_1 = NULL;
  PyObject *__pyx_t_2 = NULL;
  PyObject *__pyx_t_3 = NULL;
  PyObject *__pyx_t_4 = NULL;
  PyArrayObject *__pyx_t_5 = NULL;
  int __pyx_t_6;
  int __pyx_t_7;
  int __pyx_t_8;
  int __pyx_lineno = 0;
  const char *__pyx_filename = NULL;
  int __pyx_clineno = 0;
  __Pyx_RefNannySetupContext("blackmanFM_");
  if (__pyx_optional_args) {
    if (__pyx_optional_args->__pyx_n > 0) {
      __pyx_v_omega_m = __pyx_optional_args->omega_m;
      if (__pyx_optional_args->__pyx_n > 1) {
        __pyx_v_phi_m = __pyx_optional_args->phi_m;
        if (__pyx_optional_args->__pyx_n > 2) {
          __pyx_v_depth = __pyx_optional_args->depth;
        }
      }
    }
  }
  __pyx_bstruct_out.buf = NULL;

  /* "atom_.pyx":160
 *        omega := normalized frequency'''
 * 
 *     cdef np.ndarray[np.float_t, ndim=1, mode="c"] out = np.zeros(N, dtype=float)             # <<<<<<<<<<<<<<
 *     cdef int i
 * 
 */
  __pyx_t_1 = __Pyx_GetName(__pyx_m, __pyx_n_s__np); if (unlikely(!__pyx_t_1)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 160; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(__pyx_t_1);
  __pyx_t_2 = PyObject_GetAttr(__pyx_t_1, __pyx_n_s__zeros); if (unlikely(!__pyx_t_2)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 160; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(__pyx_t_2);
  __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;
  __pyx_t_1 = PyInt_FromLong(__pyx_v_N); if (unlikely(!__pyx_t_1)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 160; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(__pyx_t_1);
  __pyx_t_3 = PyTuple_New(1); if (unlikely(!__pyx_t_3)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 160; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(((PyObject *)__pyx_t_3));
  PyTuple_SET_ITEM(__pyx_t_3, 0, __pyx_t_1);
  __Pyx_GIVEREF(__pyx_t_1);
  __pyx_t_1 = 0;
  __pyx_t_1 = PyDict_New(); if (unlikely(!__pyx_t_1)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 160; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(((PyObject *)__pyx_t_1));
  if (PyDict_SetItem(__pyx_t_1, ((PyObject *)__pyx_n_s__dtype), ((PyObject *)((PyObject*)(&PyFloat_Type)))) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 160; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __pyx_t_4 = PyEval_CallObjectWithKeywords(__pyx_t_2, ((PyObject *)__pyx_t_3), ((PyObject *)__pyx_t_1)); if (unlikely(!__pyx_t_4)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 160; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(__pyx_t_4);
  __Pyx_DECREF(__pyx_t_2); __pyx_t_2 = 0;
  __Pyx_DECREF(((PyObject *)__pyx_t_3)); __pyx_t_3 = 0;
  __Pyx_DECREF(((PyObject *)__pyx_t_1)); __pyx_t_1 = 0;
  if (!(likely(((__pyx_t_4) == Py_None) || likely(__Pyx_TypeTest(__pyx_t_4, __pyx_ptype_5numpy_ndarray))))) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 160; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __pyx_t_5 = ((PyArrayObject *)__pyx_t_4);
  {
    __Pyx_BufFmt_StackElem __pyx_stack[1];
    if (unlikely(__Pyx_GetBufferAndValidate(&__pyx_bstruct_out, (PyObject*)__pyx_t_5, &__Pyx_TypeInfo_nn___pyx_t_5numpy_float_t, PyBUF_FORMAT| PyBUF_C_CONTIGUOUS| PyBUF_WRITABLE, 1, 0, __pyx_stack) == -1)) {
      __pyx_v_out = ((PyArrayObject *)Py_None); __Pyx_INCREF(Py_None); __pyx_bstruct_out.buf = NULL;
      {__pyx_filename = __pyx_f[0]; __pyx_lineno = 160; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
    } else {__pyx_bstride_0_out = __pyx_bstruct_out.strides[0];
      __pyx_bshape_0_out = __pyx_bstruct_out.shape[0];
    }
  }
  __pyx_t_5 = 0;
  __pyx_v_out = ((PyArrayObject *)__pyx_t_4);
  __pyx_t_4 = 0;

  /* "atom_.pyx":163
 *     cdef int i
 * 
 *     for i in range(N):             # <<<<<<<<<<<<<<
 *         out[i] = blackman(i, N) * realSinusoidFM(i, omega, chirp, phi, omega_m, phi_m, depth)
 * 
 */
  __pyx_t_6 = __pyx_v_N;
  for (__pyx_t_7 = 0; __pyx_t_7 < __pyx_t_6; __pyx_t_7+=1) {
    __pyx_v_i = __pyx_t_7;

    /* "atom_.pyx":164
 * 
 *     for i in range(N):
 *         out[i] = blackman(i, N) * realSinusoidFM(i, omega, chirp, phi, omega_m, phi_m, depth)             # <<<<<<<<<<<<<<
 * 
 *     return out
 */
    __pyx_t_4 = PyFloat_FromDouble(__pyx_v_depth); if (unlikely(!__pyx_t_4)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 164; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
    __Pyx_GOTREF(__pyx_t_4);
    __pyx_t_8 = __pyx_v_i;
    if (__pyx_t_8 < 0) __pyx_t_8 += __pyx_bshape_0_out;
    *__Pyx_BufPtrCContig1d(__pyx_t_5numpy_float_t *, __pyx_bstruct_out.buf, __pyx_t_8, __pyx_bstride_0_out) = (__pyx_f_5atom__blackman(__pyx_v_i, __pyx_v_N) * __pyx_f_5atom__realSinusoidFM(__pyx_v_i, __pyx_v_omega, __pyx_v_chirp, __pyx_v_phi, __pyx_v_omega_m, __pyx_v_phi_m, __pyx_t_4));
    __Pyx_DECREF(__pyx_t_4); __pyx_t_4 = 0;
  }

  /* "atom_.pyx":166
 *         out[i] = blackman(i, N) * realSinusoidFM(i, omega, chirp, phi, omega_m, phi_m, depth)
 * 
 *     return out             # <<<<<<<<<<<<<<
 * 
 * 
 */
  __Pyx_XDECREF(((PyObject *)__pyx_r));
  __Pyx_INCREF(((PyObject *)__pyx_v_out));
  __pyx_r = ((PyArrayObject *)__pyx_v_out);
  goto __pyx_L0;

  __pyx_r = ((PyArrayObject *)Py_None); __Pyx_INCREF(Py_None);
  goto __pyx_L0;
  __pyx_L1_error:;
  __Pyx_XDECREF(__pyx_t_1);
  __Pyx_XDECREF(__pyx_t_2);
  __Pyx_XDECREF(__pyx_t_3);
  __Pyx_XDECREF(__pyx_t_4);
  { PyObject *__pyx_type, *__pyx_value, *__pyx_tb;
    __Pyx_ErrFetch(&__pyx_type, &__pyx_value, &__pyx_tb);
    __Pyx_SafeReleaseBuffer(&__pyx_bstruct_out);
  __Pyx_ErrRestore(__pyx_type, __pyx_value, __pyx_tb);}
  __Pyx_AddTraceback("atom_.blackmanFM_", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __pyx_r = 0;
  goto __pyx_L2;
  __pyx_L0:;
  __Pyx_SafeReleaseBuffer(&__pyx_bstruct_out);
  __pyx_L2:;
  __Pyx_XDECREF((PyObject *)__pyx_v_out);
  __Pyx_XGIVEREF((PyObject *)__pyx_r);
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

/* "atom_.pyx":154
 * @cython.profile(False)
 * @cython.boundscheck(False)
 * cpdef np.ndarray[np.float_t, ndim=1, mode="c"] blackmanFM_(float phi, int N, float omega, float chirp, float omega_m=0., float phi_m=0., float depth=0.):             # <<<<<<<<<<<<<<
 *     '''A blackman FM atom where
 *        phi := initial phase
 */

static PyObject *__pyx_pf_5atom__6blackmanFM_(PyObject *__pyx_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
static char __pyx_doc_5atom__6blackmanFM_[] = "A blackman FM atom where\n       phi := initial phase\n       N := scale, i.e. length\n       omega := normalized frequency";
static PyObject *__pyx_pf_5atom__6blackmanFM_(PyObject *__pyx_self, PyObject *__pyx_args, PyObject *__pyx_kwds) {
  float __pyx_v_phi;
  int __pyx_v_N;
  float __pyx_v_omega;
  float __pyx_v_chirp;
  float __pyx_v_omega_m;
  float __pyx_v_phi_m;
  float __pyx_v_depth;
  PyObject *__pyx_r = NULL;
  __Pyx_RefNannyDeclarations
  PyObject *__pyx_t_1 = NULL;
  struct __pyx_opt_args_5atom__blackmanFM_ __pyx_t_2;
  int __pyx_lineno = 0;
  const char *__pyx_filename = NULL;
  int __pyx_clineno = 0;
  static PyObject **__pyx_pyargnames[] = {&__pyx_n_s__phi,&__pyx_n_s__N,&__pyx_n_s__omega,&__pyx_n_s__chirp,&__pyx_n_s__omega_m,&__pyx_n_s__phi_m,&__pyx_n_s__depth,0};
  __Pyx_RefNannySetupContext("blackmanFM_");
  __pyx_self = __pyx_self;
  {
    PyObject* values[7] = {0,0,0,0,0,0,0};
    if (unlikely(__pyx_kwds)) {
      Py_ssize_t kw_args;
      switch (PyTuple_GET_SIZE(__pyx_args)) {
        case  7: values[6] = PyTuple_GET_ITEM(__pyx_args, 6);
        case  6: values[5] = PyTuple_GET_ITEM(__pyx_args, 5);
        case  5: values[4] = PyTuple_GET_ITEM(__pyx_args, 4);
        case  4: values[3] = PyTuple_GET_ITEM(__pyx_args, 3);
        case  3: values[2] = PyTuple_GET_ITEM(__pyx_args, 2);
        case  2: values[1] = PyTuple_GET_ITEM(__pyx_args, 1);
        case  1: values[0] = PyTuple_GET_ITEM(__pyx_args, 0);
        case  0: break;
        default: goto __pyx_L5_argtuple_error;
      }
      kw_args = PyDict_Size(__pyx_kwds);
      switch (PyTuple_GET_SIZE(__pyx_args)) {
        case  0:
        values[0] = PyDict_GetItem(__pyx_kwds, __pyx_n_s__phi);
        if (likely(values[0])) kw_args--;
        else goto __pyx_L5_argtuple_error;
        case  1:
        values[1] = PyDict_GetItem(__pyx_kwds, __pyx_n_s__N);
        if (likely(values[1])) kw_args--;
        else {
          __Pyx_RaiseArgtupleInvalid("blackmanFM_", 0, 4, 7, 1); {__pyx_filename = __pyx_f[0]; __pyx_lineno = 154; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
        }
        case  2:
        values[2] = PyDict_GetItem(__pyx_kwds, __pyx_n_s__omega);
        if (likely(values[2])) kw_args--;
        else {
          __Pyx_RaiseArgtupleInvalid("blackmanFM_", 0, 4, 7, 2); {__pyx_filename = __pyx_f[0]; __pyx_lineno = 154; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
        }
        case  3:
        values[3] = PyDict_GetItem(__pyx_kwds, __pyx_n_s__chirp);
        if (likely(values[3])) kw_args--;
        else {
          __Pyx_RaiseArgtupleInvalid("blackmanFM_", 0, 4, 7, 3); {__pyx_filename = __pyx_f[0]; __pyx_lineno = 154; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
        }
        case  4:
        if (kw_args > 0) {
          PyObject* value = PyDict_GetItem(__pyx_kwds, __pyx_n_s__omega_m);
          if (value) { values[4] = value; kw_args--; }
        }
        case  5:
        if (kw_args > 0) {
          PyObject* value = PyDict_GetItem(__pyx_kwds, __pyx_n_s__phi_m);
          if (value) { values[5] = value; kw_args--; }
        }
        case  6:
        if (kw_args > 0) {
          PyObject* value = PyDict_GetItem(__pyx_kwds, __pyx_n_s__depth);
          if (value) { values[6] = value; kw_args--; }
        }
      }
      if (unlikely(kw_args > 0)) {
        if (unlikely(__Pyx_ParseOptionalKeywords(__pyx_kwds, __pyx_pyargnames, 0, values, PyTuple_GET_SIZE(__pyx_args), "blackmanFM_") < 0)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 154; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
      }
    } else {
      switch (PyTuple_GET_SIZE(__pyx_args)) {
        case  7: values[6] = PyTuple_GET_ITEM(__pyx_args, 6);
        case  6: values[5] = PyTuple_GET_ITEM(__pyx_args, 5);
        case  5: values[4] = PyTuple_GET_ITEM(__pyx_args, 4);
        case  4: values[3] = PyTuple_GET_ITEM(__pyx_args, 3);
        values[2] = PyTuple_GET_ITEM(__pyx_args, 2);
        values[1] = PyTuple_GET_ITEM(__pyx_args, 1);
        values[0] = PyTuple_GET_ITEM(__pyx_args, 0);
        break;
        default: goto __pyx_L5_argtuple_error;
      }
    }
    __pyx_v_phi = __pyx_PyFloat_AsDouble(values[0]); if (unlikely((__pyx_v_phi == (float)-1) && PyErr_Occurred())) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 154; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
    __pyx_v_N = __Pyx_PyInt_AsInt(values[1]); if (unlikely((__pyx_v_N == (int)-1) && PyErr_Occurred())) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 154; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
    __pyx_v_omega = __pyx_PyFloat_AsDouble(values[2]); if (unlikely((__pyx_v_omega == (float)-1) && PyErr_Occurred())) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 154; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
    __pyx_v_chirp = __pyx_PyFloat_AsDouble(values[3]); if (unlikely((__pyx_v_chirp == (float)-1) && PyErr_Occurred())) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 154; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
    if (values[4]) {
      __pyx_v_omega_m = __pyx_PyFloat_AsDouble(values[4]); if (unlikely((__pyx_v_omega_m == (float)-1) && PyErr_Occurred())) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 154; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
    } else {
      __pyx_v_omega_m = ((float)0.);
    }
    if (values[5]) {
      __pyx_v_phi_m = __pyx_PyFloat_AsDouble(values[5]); if (unlikely((__pyx_v_phi_m == (float)-1) && PyErr_Occurred())) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 154; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
    } else {
      __pyx_v_phi_m = ((float)0.);
    }
    if (values[6]) {
      __pyx_v_depth = __pyx_PyFloat_AsDouble(values[6]); if (unlikely((__pyx_v_depth == (float)-1) && PyErr_Occurred())) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 154; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
    } else {
      __pyx_v_depth = ((float)0.);
    }
  }
  goto __pyx_L4_argument_unpacking_done;
  __pyx_L5_argtuple_error:;
  __Pyx_RaiseArgtupleInvalid("blackmanFM_", 0, 4, 7, PyTuple_GET_SIZE(__pyx_args)); {__pyx_filename = __pyx_f[0]; __pyx_lineno = 154; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
  __pyx_L3_error:;
  __Pyx_AddTraceback("atom_.blackmanFM_", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __Pyx_RefNannyFinishContext();
  return NULL;
  __pyx_L4_argument_unpacking_done:;
  __Pyx_XDECREF(__pyx_r);
  __pyx_t_2.__pyx_n = 3;
  __pyx_t_2.omega_m = __pyx_v_omega_m;
  __pyx_t_2.phi_m = __pyx_v_phi_m;
  __pyx_t_2.depth = __pyx_v_depth;
  __pyx_t_1 = ((PyObject *)__pyx_f_5atom__blackmanFM_(__pyx_v_phi, __pyx_v_N, __pyx_v_omega, __pyx_v_chirp, 0, &__pyx_t_2)); if (unlikely(!__pyx_t_1)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 154; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(__pyx_t_1);
  __pyx_r = __pyx_t_1;
  __pyx_t_1 = 0;
  goto __pyx_L0;

  __pyx_r = Py_None; __Pyx_INCREF(Py_None);
  goto __pyx_L0;
  __pyx_L1_error:;
  __Pyx_XDECREF(__pyx_t_1);
  __Pyx_AddTraceback("atom_.blackmanFM_", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __pyx_r = NULL;
  __pyx_L0:;
  __Pyx_XGIVEREF(__pyx_r);
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

/* "atom_.pyx":173
 * @cython.profile(False)
 * @cython.boundscheck(False)
 * cpdef np.ndarray[np.float_t, ndim=1, mode="c"] gamma_(float phi, int N, float omega, float chirp, float order, float bandwidth):             # <<<<<<<<<<<<<<
 * 
 *     cdef np.ndarray[np.float_t, ndim=1, mode="c"] out = np.zeros(N, dtype=float)
 */

static PyObject *__pyx_pf_5atom__7gamma_(PyObject *__pyx_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
static PyArrayObject *__pyx_f_5atom__gamma_(float __pyx_v_phi, int __pyx_v_N, float __pyx_v_omega, float __pyx_v_chirp, float __pyx_v_order, float __pyx_v_bandwidth, int __pyx_skip_dispatch) {
  PyArrayObject *__pyx_v_out = 0;
  int __pyx_v_i;
  Py_buffer __pyx_bstruct_out;
  Py_ssize_t __pyx_bstride_0_out = 0;
  Py_ssize_t __pyx_bshape_0_out = 0;
  PyArrayObject *__pyx_r = NULL;
  __Pyx_RefNannyDeclarations
  PyObject *__pyx_t_1 = NULL;
  PyObject *__pyx_t_2 = NULL;
  PyObject *__pyx_t_3 = NULL;
  PyObject *__pyx_t_4 = NULL;
  PyArrayObject *__pyx_t_5 = NULL;
  int __pyx_t_6;
  int __pyx_t_7;
  int __pyx_t_8;
  int __pyx_lineno = 0;
  const char *__pyx_filename = NULL;
  int __pyx_clineno = 0;
  __Pyx_RefNannySetupContext("gamma_");
  __pyx_bstruct_out.buf = NULL;

  /* "atom_.pyx":175
 * cpdef np.ndarray[np.float_t, ndim=1, mode="c"] gamma_(float phi, int N, float omega, float chirp, float order, float bandwidth):
 * 
 *     cdef np.ndarray[np.float_t, ndim=1, mode="c"] out = np.zeros(N, dtype=float)             # <<<<<<<<<<<<<<
 *     cdef int i
 * 
 */
  __pyx_t_1 = __Pyx_GetName(__pyx_m, __pyx_n_s__np); if (unlikely(!__pyx_t_1)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 175; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(__pyx_t_1);
  __pyx_t_2 = PyObject_GetAttr(__pyx_t_1, __pyx_n_s__zeros); if (unlikely(!__pyx_t_2)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 175; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(__pyx_t_2);
  __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;
  __pyx_t_1 = PyInt_FromLong(__pyx_v_N); if (unlikely(!__pyx_t_1)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 175; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(__pyx_t_1);
  __pyx_t_3 = PyTuple_New(1); if (unlikely(!__pyx_t_3)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 175; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(((PyObject *)__pyx_t_3));
  PyTuple_SET_ITEM(__pyx_t_3, 0, __pyx_t_1);
  __Pyx_GIVEREF(__pyx_t_1);
  __pyx_t_1 = 0;
  __pyx_t_1 = PyDict_New(); if (unlikely(!__pyx_t_1)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 175; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(((PyObject *)__pyx_t_1));
  if (PyDict_SetItem(__pyx_t_1, ((PyObject *)__pyx_n_s__dtype), ((PyObject *)((PyObject*)(&PyFloat_Type)))) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 175; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __pyx_t_4 = PyEval_CallObjectWithKeywords(__pyx_t_2, ((PyObject *)__pyx_t_3), ((PyObject *)__pyx_t_1)); if (unlikely(!__pyx_t_4)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 175; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(__pyx_t_4);
  __Pyx_DECREF(__pyx_t_2); __pyx_t_2 = 0;
  __Pyx_DECREF(((PyObject *)__pyx_t_3)); __pyx_t_3 = 0;
  __Pyx_DECREF(((PyObject *)__pyx_t_1)); __pyx_t_1 = 0;
  if (!(likely(((__pyx_t_4) == Py_None) || likely(__Pyx_TypeTest(__pyx_t_4, __pyx_ptype_5numpy_ndarray))))) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 175; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __pyx_t_5 = ((PyArrayObject *)__pyx_t_4);
  {
    __Pyx_BufFmt_StackElem __pyx_stack[1];
    if (unlikely(__Pyx_GetBufferAndValidate(&__pyx_bstruct_out, (PyObject*)__pyx_t_5, &__Pyx_TypeInfo_nn___pyx_t_5numpy_float_t, PyBUF_FORMAT| PyBUF_C_CONTIGUOUS| PyBUF_WRITABLE, 1, 0, __pyx_stack) == -1)) {
      __pyx_v_out = ((PyArrayObject *)Py_None); __Pyx_INCREF(Py_None); __pyx_bstruct_out.buf = NULL;
      {__pyx_filename = __pyx_f[0]; __pyx_lineno = 175; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
    } else {__pyx_bstride_0_out = __pyx_bstruct_out.strides[0];
      __pyx_bshape_0_out = __pyx_bstruct_out.shape[0];
    }
  }
  __pyx_t_5 = 0;
  __pyx_v_out = ((PyArrayObject *)__pyx_t_4);
  __pyx_t_4 = 0;

  /* "atom_.pyx":178
 *     cdef int i
 * 
 *     for i in range(N):             # <<<<<<<<<<<<<<
 *         out[i] = (i**order-1) * exp(-2 * pi * bandwidth * i) * realSinusoid(i, omega, chirp/N, phi)
 * 
 */
  __pyx_t_6 = __pyx_v_N;
  for (__pyx_t_7 = 0; __pyx_t_7 < __pyx_t_6; __pyx_t_7+=1) {
    __pyx_v_i = __pyx_t_7;

    /* "atom_.pyx":179
 * 
 *     for i in range(N):
 *         out[i] = (i**order-1) * exp(-2 * pi * bandwidth * i) * realSinusoid(i, omega, chirp/N, phi)             # <<<<<<<<<<<<<<
 * 
 *     return out
 */
    if (unlikely(__pyx_v_N == 0)) {
      PyErr_Format(PyExc_ZeroDivisionError, "float division");
      {__pyx_filename = __pyx_f[0]; __pyx_lineno = 179; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
    }
    __pyx_t_8 = __pyx_v_i;
    if (__pyx_t_8 < 0) __pyx_t_8 += __pyx_bshape_0_out;
    *__Pyx_BufPtrCContig1d(__pyx_t_5numpy_float_t *, __pyx_bstruct_out.buf, __pyx_t_8, __pyx_bstride_0_out) = (((powf(((float)__pyx_v_i), __pyx_v_order) - 1.0) * exp((((-2.0 * __pyx_v_5atom__pi) * __pyx_v_bandwidth) * __pyx_v_i))) * __pyx_f_5atom__realSinusoid(__pyx_v_i, __pyx_v_omega, (__pyx_v_chirp / __pyx_v_N), __pyx_v_phi));
  }

  /* "atom_.pyx":181
 *         out[i] = (i**order-1) * exp(-2 * pi * bandwidth * i) * realSinusoid(i, omega, chirp/N, phi)
 * 
 *     return out             # <<<<<<<<<<<<<<
 * 
 * #GammaFM for python
 */
  __Pyx_XDECREF(((PyObject *)__pyx_r));
  __Pyx_INCREF(((PyObject *)__pyx_v_out));
  __pyx_r = ((PyArrayObject *)__pyx_v_out);
  goto __pyx_L0;

  __pyx_r = ((PyArrayObject *)Py_None); __Pyx_INCREF(Py_None);
  goto __pyx_L0;
  __pyx_L1_error:;
  __Pyx_XDECREF(__pyx_t_1);
  __Pyx_XDECREF(__pyx_t_2);
  __Pyx_XDECREF(__pyx_t_3);
  __Pyx_XDECREF(__pyx_t_4);
  { PyObject *__pyx_type, *__pyx_value, *__pyx_tb;
    __Pyx_ErrFetch(&__pyx_type, &__pyx_value, &__pyx_tb);
    __Pyx_SafeReleaseBuffer(&__pyx_bstruct_out);
  __Pyx_ErrRestore(__pyx_type, __pyx_value, __pyx_tb);}
  __Pyx_AddTraceback("atom_.gamma_", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __pyx_r = 0;
  goto __pyx_L2;
  __pyx_L0:;
  __Pyx_SafeReleaseBuffer(&__pyx_bstruct_out);
  __pyx_L2:;
  __Pyx_XDECREF((PyObject *)__pyx_v_out);
  __Pyx_XGIVEREF((PyObject *)__pyx_r);
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

/* "atom_.pyx":173
 * @cython.profile(False)
 * @cython.boundscheck(False)
 * cpdef np.ndarray[np.float_t, ndim=1, mode="c"] gamma_(float phi, int N, float omega, float chirp, float order, float bandwidth):             # <<<<<<<<<<<<<<
 * 
 *     cdef np.ndarray[np.float_t, ndim=1, mode="c"] out = np.zeros(N, dtype=float)
 */

static PyObject *__pyx_pf_5atom__7gamma_(PyObject *__pyx_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
static PyObject *__pyx_pf_5atom__7gamma_(PyObject *__pyx_self, PyObject *__pyx_args, PyObject *__pyx_kwds) {
  float __pyx_v_phi;
  int __pyx_v_N;
  float __pyx_v_omega;
  float __pyx_v_chirp;
  float __pyx_v_order;
  float __pyx_v_bandwidth;
  PyObject *__pyx_r = NULL;
  __Pyx_RefNannyDeclarations
  PyObject *__pyx_t_1 = NULL;
  int __pyx_lineno = 0;
  const char *__pyx_filename = NULL;
  int __pyx_clineno = 0;
  static PyObject **__pyx_pyargnames[] = {&__pyx_n_s__phi,&__pyx_n_s__N,&__pyx_n_s__omega,&__pyx_n_s__chirp,&__pyx_n_s__order,&__pyx_n_s__bandwidth,0};
  __Pyx_RefNannySetupContext("gamma_");
  __pyx_self = __pyx_self;
  {
    PyObject* values[6] = {0,0,0,0,0,0};
    if (unlikely(__pyx_kwds)) {
      Py_ssize_t kw_args;
      switch (PyTuple_GET_SIZE(__pyx_args)) {
        case  6: values[5] = PyTuple_GET_ITEM(__pyx_args, 5);
        case  5: values[4] = PyTuple_GET_ITEM(__pyx_args, 4);
        case  4: values[3] = PyTuple_GET_ITEM(__pyx_args, 3);
        case  3: values[2] = PyTuple_GET_ITEM(__pyx_args, 2);
        case  2: values[1] = PyTuple_GET_ITEM(__pyx_args, 1);
        case  1: values[0] = PyTuple_GET_ITEM(__pyx_args, 0);
        case  0: break;
        default: goto __pyx_L5_argtuple_error;
      }
      kw_args = PyDict_Size(__pyx_kwds);
      switch (PyTuple_GET_SIZE(__pyx_args)) {
        case  0:
        values[0] = PyDict_GetItem(__pyx_kwds, __pyx_n_s__phi);
        if (likely(values[0])) kw_args--;
        else goto __pyx_L5_argtuple_error;
        case  1:
        values[1] = PyDict_GetItem(__pyx_kwds, __pyx_n_s__N);
        if (likely(values[1])) kw_args--;
        else {
          __Pyx_RaiseArgtupleInvalid("gamma_", 1, 6, 6, 1); {__pyx_filename = __pyx_f[0]; __pyx_lineno = 173; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
        }
        case  2:
        values[2] = PyDict_GetItem(__pyx_kwds, __pyx_n_s__omega);
        if (likely(values[2])) kw_args--;
        else {
          __Pyx_RaiseArgtupleInvalid("gamma_", 1, 6, 6, 2); {__pyx_filename = __pyx_f[0]; __pyx_lineno = 173; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
        }
        case  3:
        values[3] = PyDict_GetItem(__pyx_kwds, __pyx_n_s__chirp);
        if (likely(values[3])) kw_args--;
        else {
          __Pyx_RaiseArgtupleInvalid("gamma_", 1, 6, 6, 3); {__pyx_filename = __pyx_f[0]; __pyx_lineno = 173; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
        }
        case  4:
        values[4] = PyDict_GetItem(__pyx_kwds, __pyx_n_s__order);
        if (likely(values[4])) kw_args--;
        else {
          __Pyx_RaiseArgtupleInvalid("gamma_", 1, 6, 6, 4); {__pyx_filename = __pyx_f[0]; __pyx_lineno = 173; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
        }
        case  5:
        values[5] = PyDict_GetItem(__pyx_kwds, __pyx_n_s__bandwidth);
        if (likely(values[5])) kw_args--;
        else {
          __Pyx_RaiseArgtupleInvalid("gamma_", 1, 6, 6, 5); {__pyx_filename = __pyx_f[0]; __pyx_lineno = 173; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
        }
      }
      if (unlikely(kw_args > 0)) {
        if (unlikely(__Pyx_ParseOptionalKeywords(__pyx_kwds, __pyx_pyargnames, 0, values, PyTuple_GET_SIZE(__pyx_args), "gamma_") < 0)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 173; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
      }
    } else if (PyTuple_GET_SIZE(__pyx_args) != 6) {
      goto __pyx_L5_argtuple_error;
    } else {
      values[0] = PyTuple_GET_ITEM(__pyx_args, 0);
      values[1] = PyTuple_GET_ITEM(__pyx_args, 1);
      values[2] = PyTuple_GET_ITEM(__pyx_args, 2);
      values[3] = PyTuple_GET_ITEM(__pyx_args, 3);
      values[4] = PyTuple_GET_ITEM(__pyx_args, 4);
      values[5] = PyTuple_GET_ITEM(__pyx_args, 5);
    }
    __pyx_v_phi = __pyx_PyFloat_AsDouble(values[0]); if (unlikely((__pyx_v_phi == (float)-1) && PyErr_Occurred())) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 173; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
    __pyx_v_N = __Pyx_PyInt_AsInt(values[1]); if (unlikely((__pyx_v_N == (int)-1) && PyErr_Occurred())) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 173; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
    __pyx_v_omega = __pyx_PyFloat_AsDouble(values[2]); if (unlikely((__pyx_v_omega == (float)-1) && PyErr_Occurred())) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 173; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
    __pyx_v_chirp = __pyx_PyFloat_AsDouble(values[3]); if (unlikely((__pyx_v_chirp == (float)-1) && PyErr_Occurred())) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 173; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
    __pyx_v_order = __pyx_PyFloat_AsDouble(values[4]); if (unlikely((__pyx_v_order == (float)-1) && PyErr_Occurred())) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 173; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
    __pyx_v_bandwidth = __pyx_PyFloat_AsDouble(values[5]); if (unlikely((__pyx_v_bandwidth == (float)-1) && PyErr_Occurred())) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 173; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
  }
  goto __pyx_L4_argument_unpacking_done;
  __pyx_L5_argtuple_error:;
  __Pyx_RaiseArgtupleInvalid("gamma_", 1, 6, 6, PyTuple_GET_SIZE(__pyx_args)); {__pyx_filename = __pyx_f[0]; __pyx_lineno = 173; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
  __pyx_L3_error:;
  __Pyx_AddTraceback("atom_.gamma_", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __Pyx_RefNannyFinishContext();
  return NULL;
  __pyx_L4_argument_unpacking_done:;
  __Pyx_XDECREF(__pyx_r);
  __pyx_t_1 = ((PyObject *)__pyx_f_5atom__gamma_(__pyx_v_phi, __pyx_v_N, __pyx_v_omega, __pyx_v_chirp, __pyx_v_order, __pyx_v_bandwidth, 0)); if (unlikely(!__pyx_t_1)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 173; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(__pyx_t_1);
  __pyx_r = __pyx_t_1;
  __pyx_t_1 = 0;
  goto __pyx_L0;

  __pyx_r = Py_None; __Pyx_INCREF(Py_None);
  goto __pyx_L0;
  __pyx_L1_error:;
  __Pyx_XDECREF(__pyx_t_1);
  __Pyx_AddTraceback("atom_.gamma_", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __pyx_r = NULL;
  __pyx_L0:;
  __Pyx_XGIVEREF(__pyx_r);
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

/* "atom_.pyx":187
 * @cython.profile(False)
 * @cython.boundscheck(False)
 * cpdef np.ndarray[np.float_t, ndim=1, mode="c"] gammaFM_(float phi, int N, float omega, float chirp, float order, float bandwidth, float omega_m=0., float phi_m=0., float depth=0.):             # <<<<<<<<<<<<<<
 * 
 *     cdef np.ndarray[np.float_t, ndim=1, mode="c"] out = np.zeros(N, dtype=float)
 */

static PyObject *__pyx_pf_5atom__8gammaFM_(PyObject *__pyx_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
static PyArrayObject *__pyx_f_5atom__gammaFM_(float __pyx_v_phi, int __pyx_v_N, float __pyx_v_omega, float __pyx_v_chirp, float __pyx_v_order, float __pyx_v_bandwidth, int __pyx_skip_dispatch, struct __pyx_opt_args_5atom__gammaFM_ *__pyx_optional_args) {
  float __pyx_v_omega_m = ((float)0.);
  float __pyx_v_phi_m = ((float)0.);
  float __pyx_v_depth = ((float)0.);
  PyArrayObject *__pyx_v_out = 0;
  int __pyx_v_i;
  Py_buffer __pyx_bstruct_out;
  Py_ssize_t __pyx_bstride_0_out = 0;
  Py_ssize_t __pyx_bshape_0_out = 0;
  PyArrayObject *__pyx_r = NULL;
  __Pyx_RefNannyDeclarations
  PyObject *__pyx_t_1 = NULL;
  PyObject *__pyx_t_2 = NULL;
  PyObject *__pyx_t_3 = NULL;
  PyObject *__pyx_t_4 = NULL;
  PyArrayObject *__pyx_t_5 = NULL;
  int __pyx_t_6;
  int __pyx_t_7;
  int __pyx_t_8;
  int __pyx_lineno = 0;
  const char *__pyx_filename = NULL;
  int __pyx_clineno = 0;
  __Pyx_RefNannySetupContext("gammaFM_");
  if (__pyx_optional_args) {
    if (__pyx_optional_args->__pyx_n > 0) {
      __pyx_v_omega_m = __pyx_optional_args->omega_m;
      if (__pyx_optional_args->__pyx_n > 1) {
        __pyx_v_phi_m = __pyx_optional_args->phi_m;
        if (__pyx_optional_args->__pyx_n > 2) {
          __pyx_v_depth = __pyx_optional_args->depth;
        }
      }
    }
  }
  __pyx_bstruct_out.buf = NULL;

  /* "atom_.pyx":189
 * cpdef np.ndarray[np.float_t, ndim=1, mode="c"] gammaFM_(float phi, int N, float omega, float chirp, float order, float bandwidth, float omega_m=0., float phi_m=0., float depth=0.):
 * 
 *     cdef np.ndarray[np.float_t, ndim=1, mode="c"] out = np.zeros(N, dtype=float)             # <<<<<<<<<<<<<<
 *     cdef int i
 * 
 */
  __pyx_t_1 = __Pyx_GetName(__pyx_m, __pyx_n_s__np); if (unlikely(!__pyx_t_1)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 189; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(__pyx_t_1);
  __pyx_t_2 = PyObject_GetAttr(__pyx_t_1, __pyx_n_s__zeros); if (unlikely(!__pyx_t_2)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 189; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(__pyx_t_2);
  __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;
  __pyx_t_1 = PyInt_FromLong(__pyx_v_N); if (unlikely(!__pyx_t_1)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 189; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(__pyx_t_1);
  __pyx_t_3 = PyTuple_New(1); if (unlikely(!__pyx_t_3)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 189; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(((PyObject *)__pyx_t_3));
  PyTuple_SET_ITEM(__pyx_t_3, 0, __pyx_t_1);
  __Pyx_GIVEREF(__pyx_t_1);
  __pyx_t_1 = 0;
  __pyx_t_1 = PyDict_New(); if (unlikely(!__pyx_t_1)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 189; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(((PyObject *)__pyx_t_1));
  if (PyDict_SetItem(__pyx_t_1, ((PyObject *)__pyx_n_s__dtype), ((PyObject *)((PyObject*)(&PyFloat_Type)))) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 189; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __pyx_t_4 = PyEval_CallObjectWithKeywords(__pyx_t_2, ((PyObject *)__pyx_t_3), ((PyObject *)__pyx_t_1)); if (unlikely(!__pyx_t_4)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 189; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(__pyx_t_4);
  __Pyx_DECREF(__pyx_t_2); __pyx_t_2 = 0;
  __Pyx_DECREF(((PyObject *)__pyx_t_3)); __pyx_t_3 = 0;
  __Pyx_DECREF(((PyObject *)__pyx_t_1)); __pyx_t_1 = 0;
  if (!(likely(((__pyx_t_4) == Py_None) || likely(__Pyx_TypeTest(__pyx_t_4, __pyx_ptype_5numpy_ndarray))))) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 189; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __pyx_t_5 = ((PyArrayObject *)__pyx_t_4);
  {
    __Pyx_BufFmt_StackElem __pyx_stack[1];
    if (unlikely(__Pyx_GetBufferAndValidate(&__pyx_bstruct_out, (PyObject*)__pyx_t_5, &__Pyx_TypeInfo_nn___pyx_t_5numpy_float_t, PyBUF_FORMAT| PyBUF_C_CONTIGUOUS| PyBUF_WRITABLE, 1, 0, __pyx_stack) == -1)) {
      __pyx_v_out = ((PyArrayObject *)Py_None); __Pyx_INCREF(Py_None); __pyx_bstruct_out.buf = NULL;
      {__pyx_filename = __pyx_f[0]; __pyx_lineno = 189; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
    } else {__pyx_bstride_0_out = __pyx_bstruct_out.strides[0];
      __pyx_bshape_0_out = __pyx_bstruct_out.shape[0];
    }
  }
  __pyx_t_5 = 0;
  __pyx_v_out = ((PyArrayObject *)__pyx_t_4);
  __pyx_t_4 = 0;

  /* "atom_.pyx":192
 *     cdef int i
 * 
 *     for i in range(N):             # <<<<<<<<<<<<<<
 *         out[i] = (i**order-1) * exp(-2 * pi * bandwidth * i) * realSinusoidFM(i, omega, chirp, phi, omega_m, phi_m, depth)
 * 
 */
  __pyx_t_6 = __pyx_v_N;
  for (__pyx_t_7 = 0; __pyx_t_7 < __pyx_t_6; __pyx_t_7+=1) {
    __pyx_v_i = __pyx_t_7;

    /* "atom_.pyx":193
 * 
 *     for i in range(N):
 *         out[i] = (i**order-1) * exp(-2 * pi * bandwidth * i) * realSinusoidFM(i, omega, chirp, phi, omega_m, phi_m, depth)             # <<<<<<<<<<<<<<
 * 
 *     return out
 */
    __pyx_t_4 = PyFloat_FromDouble(__pyx_v_depth); if (unlikely(!__pyx_t_4)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 193; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
    __Pyx_GOTREF(__pyx_t_4);
    __pyx_t_8 = __pyx_v_i;
    if (__pyx_t_8 < 0) __pyx_t_8 += __pyx_bshape_0_out;
    *__Pyx_BufPtrCContig1d(__pyx_t_5numpy_float_t *, __pyx_bstruct_out.buf, __pyx_t_8, __pyx_bstride_0_out) = (((powf(((float)__pyx_v_i), __pyx_v_order) - 1.0) * exp((((-2.0 * __pyx_v_5atom__pi) * __pyx_v_bandwidth) * __pyx_v_i))) * __pyx_f_5atom__realSinusoidFM(__pyx_v_i, __pyx_v_omega, __pyx_v_chirp, __pyx_v_phi, __pyx_v_omega_m, __pyx_v_phi_m, __pyx_t_4));
    __Pyx_DECREF(__pyx_t_4); __pyx_t_4 = 0;
  }

  /* "atom_.pyx":195
 *         out[i] = (i**order-1) * exp(-2 * pi * bandwidth * i) * realSinusoidFM(i, omega, chirp, phi, omega_m, phi_m, depth)
 * 
 *     return out             # <<<<<<<<<<<<<<
 * 
 * 
 */
  __Pyx_XDECREF(((PyObject *)__pyx_r));
  __Pyx_INCREF(((PyObject *)__pyx_v_out));
  __pyx_r = ((PyArrayObject *)__pyx_v_out);
  goto __pyx_L0;

  __pyx_r = ((PyArrayObject *)Py_None); __Pyx_INCREF(Py_None);
  goto __pyx_L0;
  __pyx_L1_error:;
  __Pyx_XDECREF(__pyx_t_1);
  __Pyx_XDECREF(__pyx_t_2);
  __Pyx_XDECREF(__pyx_t_3);
  __Pyx_XDECREF(__pyx_t_4);
  { PyObject *__pyx_type, *__pyx_value, *__pyx_tb;
    __Pyx_ErrFetch(&__pyx_type, &__pyx_value, &__pyx_tb);
    __Pyx_SafeReleaseBuffer(&__pyx_bstruct_out);
  __Pyx_ErrRestore(__pyx_type, __pyx_value, __pyx_tb);}
  __Pyx_AddTraceback("atom_.gammaFM_", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __pyx_r = 0;
  goto __pyx_L2;
  __pyx_L0:;
  __Pyx_SafeReleaseBuffer(&__pyx_bstruct_out);
  __pyx_L2:;
  __Pyx_XDECREF((PyObject *)__pyx_v_out);
  __Pyx_XGIVEREF((PyObject *)__pyx_r);
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

/* "atom_.pyx":187
 * @cython.profile(False)
 * @cython.boundscheck(False)
 * cpdef np.ndarray[np.float_t, ndim=1, mode="c"] gammaFM_(float phi, int N, float omega, float chirp, float order, float bandwidth, float omega_m=0., float phi_m=0., float depth=0.):             # <<<<<<<<<<<<<<
 * 
 *     cdef np.ndarray[np.float_t, ndim=1, mode="c"] out = np.zeros(N, dtype=float)
 */

static PyObject *__pyx_pf_5atom__8gammaFM_(PyObject *__pyx_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
static PyObject *__pyx_pf_5atom__8gammaFM_(PyObject *__pyx_self, PyObject *__pyx_args, PyObject *__pyx_kwds) {
  float __pyx_v_phi;
  int __pyx_v_N;
  float __pyx_v_omega;
  float __pyx_v_chirp;
  float __pyx_v_order;
  float __pyx_v_bandwidth;
  float __pyx_v_omega_m;
  float __pyx_v_phi_m;
  float __pyx_v_depth;
  PyObject *__pyx_r = NULL;
  __Pyx_RefNannyDeclarations
  PyObject *__pyx_t_1 = NULL;
  struct __pyx_opt_args_5atom__gammaFM_ __pyx_t_2;
  int __pyx_lineno = 0;
  const char *__pyx_filename = NULL;
  int __pyx_clineno = 0;
  static PyObject **__pyx_pyargnames[] = {&__pyx_n_s__phi,&__pyx_n_s__N,&__pyx_n_s__omega,&__pyx_n_s__chirp,&__pyx_n_s__order,&__pyx_n_s__bandwidth,&__pyx_n_s__omega_m,&__pyx_n_s__phi_m,&__pyx_n_s__depth,0};
  __Pyx_RefNannySetupContext("gammaFM_");
  __pyx_self = __pyx_self;
  {
    PyObject* values[9] = {0,0,0,0,0,0,0,0,0};
    if (unlikely(__pyx_kwds)) {
      Py_ssize_t kw_args;
      switch (PyTuple_GET_SIZE(__pyx_args)) {
        case  9: values[8] = PyTuple_GET_ITEM(__pyx_args, 8);
        case  8: values[7] = PyTuple_GET_ITEM(__pyx_args, 7);
        case  7: values[6] = PyTuple_GET_ITEM(__pyx_args, 6);
        case  6: values[5] = PyTuple_GET_ITEM(__pyx_args, 5);
        case  5: values[4] = PyTuple_GET_ITEM(__pyx_args, 4);
        case  4: values[3] = PyTuple_GET_ITEM(__pyx_args, 3);
        case  3: values[2] = PyTuple_GET_ITEM(__pyx_args, 2);
        case  2: values[1] = PyTuple_GET_ITEM(__pyx_args, 1);
        case  1: values[0] = PyTuple_GET_ITEM(__pyx_args, 0);
        case  0: break;
        default: goto __pyx_L5_argtuple_error;
      }
      kw_args = PyDict_Size(__pyx_kwds);
      switch (PyTuple_GET_SIZE(__pyx_args)) {
        case  0:
        values[0] = PyDict_GetItem(__pyx_kwds, __pyx_n_s__phi);
        if (likely(values[0])) kw_args--;
        else goto __pyx_L5_argtuple_error;
        case  1:
        values[1] = PyDict_GetItem(__pyx_kwds, __pyx_n_s__N);
        if (likely(values[1])) kw_args--;
        else {
          __Pyx_RaiseArgtupleInvalid("gammaFM_", 0, 6, 9, 1); {__pyx_filename = __pyx_f[0]; __pyx_lineno = 187; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
        }
        case  2:
        values[2] = PyDict_GetItem(__pyx_kwds, __pyx_n_s__omega);
        if (likely(values[2])) kw_args--;
        else {
          __Pyx_RaiseArgtupleInvalid("gammaFM_", 0, 6, 9, 2); {__pyx_filename = __pyx_f[0]; __pyx_lineno = 187; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
        }
        case  3:
        values[3] = PyDict_GetItem(__pyx_kwds, __pyx_n_s__chirp);
        if (likely(values[3])) kw_args--;
        else {
          __Pyx_RaiseArgtupleInvalid("gammaFM_", 0, 6, 9, 3); {__pyx_filename = __pyx_f[0]; __pyx_lineno = 187; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
        }
        case  4:
        values[4] = PyDict_GetItem(__pyx_kwds, __pyx_n_s__order);
        if (likely(values[4])) kw_args--;
        else {
          __Pyx_RaiseArgtupleInvalid("gammaFM_", 0, 6, 9, 4); {__pyx_filename = __pyx_f[0]; __pyx_lineno = 187; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
        }
        case  5:
        values[5] = PyDict_GetItem(__pyx_kwds, __pyx_n_s__bandwidth);
        if (likely(values[5])) kw_args--;
        else {
          __Pyx_RaiseArgtupleInvalid("gammaFM_", 0, 6, 9, 5); {__pyx_filename = __pyx_f[0]; __pyx_lineno = 187; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
        }
        case  6:
        if (kw_args > 0) {
          PyObject* value = PyDict_GetItem(__pyx_kwds, __pyx_n_s__omega_m);
          if (value) { values[6] = value; kw_args--; }
        }
        case  7:
        if (kw_args > 0) {
          PyObject* value = PyDict_GetItem(__pyx_kwds, __pyx_n_s__phi_m);
          if (value) { values[7] = value; kw_args--; }
        }
        case  8:
        if (kw_args > 0) {
          PyObject* value = PyDict_GetItem(__pyx_kwds, __pyx_n_s__depth);
          if (value) { values[8] = value; kw_args--; }
        }
      }
      if (unlikely(kw_args > 0)) {
        if (unlikely(__Pyx_ParseOptionalKeywords(__pyx_kwds, __pyx_pyargnames, 0, values, PyTuple_GET_SIZE(__pyx_args), "gammaFM_") < 0)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 187; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
      }
    } else {
      switch (PyTuple_GET_SIZE(__pyx_args)) {
        case  9: values[8] = PyTuple_GET_ITEM(__pyx_args, 8);
        case  8: values[7] = PyTuple_GET_ITEM(__pyx_args, 7);
        case  7: values[6] = PyTuple_GET_ITEM(__pyx_args, 6);
        case  6: values[5] = PyTuple_GET_ITEM(__pyx_args, 5);
        values[4] = PyTuple_GET_ITEM(__pyx_args, 4);
        values[3] = PyTuple_GET_ITEM(__pyx_args, 3);
        values[2] = PyTuple_GET_ITEM(__pyx_args, 2);
        values[1] = PyTuple_GET_ITEM(__pyx_args, 1);
        values[0] = PyTuple_GET_ITEM(__pyx_args, 0);
        break;
        default: goto __pyx_L5_argtuple_error;
      }
    }
    __pyx_v_phi = __pyx_PyFloat_AsDouble(values[0]); if (unlikely((__pyx_v_phi == (float)-1) && PyErr_Occurred())) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 187; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
    __pyx_v_N = __Pyx_PyInt_AsInt(values[1]); if (unlikely((__pyx_v_N == (int)-1) && PyErr_Occurred())) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 187; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
    __pyx_v_omega = __pyx_PyFloat_AsDouble(values[2]); if (unlikely((__pyx_v_omega == (float)-1) && PyErr_Occurred())) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 187; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
    __pyx_v_chirp = __pyx_PyFloat_AsDouble(values[3]); if (unlikely((__pyx_v_chirp == (float)-1) && PyErr_Occurred())) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 187; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
    __pyx_v_order = __pyx_PyFloat_AsDouble(values[4]); if (unlikely((__pyx_v_order == (float)-1) && PyErr_Occurred())) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 187; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
    __pyx_v_bandwidth = __pyx_PyFloat_AsDouble(values[5]); if (unlikely((__pyx_v_bandwidth == (float)-1) && PyErr_Occurred())) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 187; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
    if (values[6]) {
      __pyx_v_omega_m = __pyx_PyFloat_AsDouble(values[6]); if (unlikely((__pyx_v_omega_m == (float)-1) && PyErr_Occurred())) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 187; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
    } else {
      __pyx_v_omega_m = ((float)0.);
    }
    if (values[7]) {
      __pyx_v_phi_m = __pyx_PyFloat_AsDouble(values[7]); if (unlikely((__pyx_v_phi_m == (float)-1) && PyErr_Occurred())) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 187; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
    } else {
      __pyx_v_phi_m = ((float)0.);
    }
    if (values[8]) {
      __pyx_v_depth = __pyx_PyFloat_AsDouble(values[8]); if (unlikely((__pyx_v_depth == (float)-1) && PyErr_Occurred())) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 187; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
    } else {
      __pyx_v_depth = ((float)0.);
    }
  }
  goto __pyx_L4_argument_unpacking_done;
  __pyx_L5_argtuple_error:;
  __Pyx_RaiseArgtupleInvalid("gammaFM_", 0, 6, 9, PyTuple_GET_SIZE(__pyx_args)); {__pyx_filename = __pyx_f[0]; __pyx_lineno = 187; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
  __pyx_L3_error:;
  __Pyx_AddTraceback("atom_.gammaFM_", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __Pyx_RefNannyFinishContext();
  return NULL;
  __pyx_L4_argument_unpacking_done:;
  __Pyx_XDECREF(__pyx_r);
  __pyx_t_2.__pyx_n = 3;
  __pyx_t_2.omega_m = __pyx_v_omega_m;
  __pyx_t_2.phi_m = __pyx_v_phi_m;
  __pyx_t_2.depth = __pyx_v_depth;
  __pyx_t_1 = ((PyObject *)__pyx_f_5atom__gammaFM_(__pyx_v_phi, __pyx_v_N, __pyx_v_omega, __pyx_v_chirp, __pyx_v_order, __pyx_v_bandwidth, 0, &__pyx_t_2)); if (unlikely(!__pyx_t_1)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 187; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(__pyx_t_1);
  __pyx_r = __pyx_t_1;
  __pyx_t_1 = 0;
  goto __pyx_L0;

  __pyx_r = Py_None; __Pyx_INCREF(Py_None);
  goto __pyx_L0;
  __pyx_L1_error:;
  __Pyx_XDECREF(__pyx_t_1);
  __Pyx_AddTraceback("atom_.gammaFM_", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __pyx_r = NULL;
  __pyx_L0:;
  __Pyx_XGIVEREF(__pyx_r);
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

/* "atom_.pyx":202
 * @cython.profile(False)
 * @cython.boundscheck(False)
 * cpdef np.ndarray[np.float_t, ndim=1, mode="c"] damped_(float phi, int N, float omega, float chirp, float damp):             # <<<<<<<<<<<<<<
 * 
 *     cdef np.ndarray[np.float_t, ndim=1, mode="c"] out = np.zeros(N, dtype=float)
 */

static PyObject *__pyx_pf_5atom__9damped_(PyObject *__pyx_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
static PyArrayObject *__pyx_f_5atom__damped_(float __pyx_v_phi, int __pyx_v_N, float __pyx_v_omega, float __pyx_v_chirp, float __pyx_v_damp, int __pyx_skip_dispatch) {
  PyArrayObject *__pyx_v_out = 0;
  int __pyx_v_i;
  Py_buffer __pyx_bstruct_out;
  Py_ssize_t __pyx_bstride_0_out = 0;
  Py_ssize_t __pyx_bshape_0_out = 0;
  PyArrayObject *__pyx_r = NULL;
  __Pyx_RefNannyDeclarations
  PyObject *__pyx_t_1 = NULL;
  PyObject *__pyx_t_2 = NULL;
  PyObject *__pyx_t_3 = NULL;
  PyObject *__pyx_t_4 = NULL;
  PyArrayObject *__pyx_t_5 = NULL;
  int __pyx_t_6;
  int __pyx_t_7;
  int __pyx_t_8;
  int __pyx_lineno = 0;
  const char *__pyx_filename = NULL;
  int __pyx_clineno = 0;
  __Pyx_RefNannySetupContext("damped_");
  __pyx_bstruct_out.buf = NULL;

  /* "atom_.pyx":204
 * cpdef np.ndarray[np.float_t, ndim=1, mode="c"] damped_(float phi, int N, float omega, float chirp, float damp):
 * 
 *     cdef np.ndarray[np.float_t, ndim=1, mode="c"] out = np.zeros(N, dtype=float)             # <<<<<<<<<<<<<<
 *     cdef int i
 * 
 */
  __pyx_t_1 = __Pyx_GetName(__pyx_m, __pyx_n_s__np); if (unlikely(!__pyx_t_1)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 204; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(__pyx_t_1);
  __pyx_t_2 = PyObject_GetAttr(__pyx_t_1, __pyx_n_s__zeros); if (unlikely(!__pyx_t_2)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 204; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(__pyx_t_2);
  __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;
  __pyx_t_1 = PyInt_FromLong(__pyx_v_N); if (unlikely(!__pyx_t_1)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 204; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(__pyx_t_1);
  __pyx_t_3 = PyTuple_New(1); if (unlikely(!__pyx_t_3)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 204; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(((PyObject *)__pyx_t_3));
  PyTuple_SET_ITEM(__pyx_t_3, 0, __pyx_t_1);
  __Pyx_GIVEREF(__pyx_t_1);
  __pyx_t_1 = 0;
  __pyx_t_1 = PyDict_New(); if (unlikely(!__pyx_t_1)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 204; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(((PyObject *)__pyx_t_1));
  if (PyDict_SetItem(__pyx_t_1, ((PyObject *)__pyx_n_s__dtype), ((PyObject *)((PyObject*)(&PyFloat_Type)))) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 204; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __pyx_t_4 = PyEval_CallObjectWithKeywords(__pyx_t_2, ((PyObject *)__pyx_t_3), ((PyObject *)__pyx_t_1)); if (unlikely(!__pyx_t_4)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 204; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(__pyx_t_4);
  __Pyx_DECREF(__pyx_t_2); __pyx_t_2 = 0;
  __Pyx_DECREF(((PyObject *)__pyx_t_3)); __pyx_t_3 = 0;
  __Pyx_DECREF(((PyObject *)__pyx_t_1)); __pyx_t_1 = 0;
  if (!(likely(((__pyx_t_4) == Py_None) || likely(__Pyx_TypeTest(__pyx_t_4, __pyx_ptype_5numpy_ndarray))))) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 204; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __pyx_t_5 = ((PyArrayObject *)__pyx_t_4);
  {
    __Pyx_BufFmt_StackElem __pyx_stack[1];
    if (unlikely(__Pyx_GetBufferAndValidate(&__pyx_bstruct_out, (PyObject*)__pyx_t_5, &__Pyx_TypeInfo_nn___pyx_t_5numpy_float_t, PyBUF_FORMAT| PyBUF_C_CONTIGUOUS| PyBUF_WRITABLE, 1, 0, __pyx_stack) == -1)) {
      __pyx_v_out = ((PyArrayObject *)Py_None); __Pyx_INCREF(Py_None); __pyx_bstruct_out.buf = NULL;
      {__pyx_filename = __pyx_f[0]; __pyx_lineno = 204; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
    } else {__pyx_bstride_0_out = __pyx_bstruct_out.strides[0];
      __pyx_bshape_0_out = __pyx_bstruct_out.shape[0];
    }
  }
  __pyx_t_5 = 0;
  __pyx_v_out = ((PyArrayObject *)__pyx_t_4);
  __pyx_t_4 = 0;

  /* "atom_.pyx":207
 *     cdef int i
 * 
 *     for i in range(N):             # <<<<<<<<<<<<<<
 *         out[i] = exp(-damp * i) * realSinusoid(i, omega, chirp/N, phi)
 * 
 */
  __pyx_t_6 = __pyx_v_N;
  for (__pyx_t_7 = 0; __pyx_t_7 < __pyx_t_6; __pyx_t_7+=1) {
    __pyx_v_i = __pyx_t_7;

    /* "atom_.pyx":208
 * 
 *     for i in range(N):
 *         out[i] = exp(-damp * i) * realSinusoid(i, omega, chirp/N, phi)             # <<<<<<<<<<<<<<
 * 
 *     return out
 */
    if (unlikely(__pyx_v_N == 0)) {
      PyErr_Format(PyExc_ZeroDivisionError, "float division");
      {__pyx_filename = __pyx_f[0]; __pyx_lineno = 208; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
    }
    __pyx_t_8 = __pyx_v_i;
    if (__pyx_t_8 < 0) __pyx_t_8 += __pyx_bshape_0_out;
    *__Pyx_BufPtrCContig1d(__pyx_t_5numpy_float_t *, __pyx_bstruct_out.buf, __pyx_t_8, __pyx_bstride_0_out) = (exp(((-__pyx_v_damp) * __pyx_v_i)) * __pyx_f_5atom__realSinusoid(__pyx_v_i, __pyx_v_omega, (__pyx_v_chirp / __pyx_v_N), __pyx_v_phi));
  }

  /* "atom_.pyx":210
 *         out[i] = exp(-damp * i) * realSinusoid(i, omega, chirp/N, phi)
 * 
 *     return out             # <<<<<<<<<<<<<<
 * 
 * #DampedFM for python
 */
  __Pyx_XDECREF(((PyObject *)__pyx_r));
  __Pyx_INCREF(((PyObject *)__pyx_v_out));
  __pyx_r = ((PyArrayObject *)__pyx_v_out);
  goto __pyx_L0;

  __pyx_r = ((PyArrayObject *)Py_None); __Pyx_INCREF(Py_None);
  goto __pyx_L0;
  __pyx_L1_error:;
  __Pyx_XDECREF(__pyx_t_1);
  __Pyx_XDECREF(__pyx_t_2);
  __Pyx_XDECREF(__pyx_t_3);
  __Pyx_XDECREF(__pyx_t_4);
  { PyObject *__pyx_type, *__pyx_value, *__pyx_tb;
    __Pyx_ErrFetch(&__pyx_type, &__pyx_value, &__pyx_tb);
    __Pyx_SafeReleaseBuffer(&__pyx_bstruct_out);
  __Pyx_ErrRestore(__pyx_type, __pyx_value, __pyx_tb);}
  __Pyx_AddTraceback("atom_.damped_", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __pyx_r = 0;
  goto __pyx_L2;
  __pyx_L0:;
  __Pyx_SafeReleaseBuffer(&__pyx_bstruct_out);
  __pyx_L2:;
  __Pyx_XDECREF((PyObject *)__pyx_v_out);
  __Pyx_XGIVEREF((PyObject *)__pyx_r);
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

/* "atom_.pyx":202
 * @cython.profile(False)
 * @cython.boundscheck(False)
 * cpdef np.ndarray[np.float_t, ndim=1, mode="c"] damped_(float phi, int N, float omega, float chirp, float damp):             # <<<<<<<<<<<<<<
 * 
 *     cdef np.ndarray[np.float_t, ndim=1, mode="c"] out = np.zeros(N, dtype=float)
 */

static PyObject *__pyx_pf_5atom__9damped_(PyObject *__pyx_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
static PyObject *__pyx_pf_5atom__9damped_(PyObject *__pyx_self, PyObject *__pyx_args, PyObject *__pyx_kwds) {
  float __pyx_v_phi;
  int __pyx_v_N;
  float __pyx_v_omega;
  float __pyx_v_chirp;
  float __pyx_v_damp;
  PyObject *__pyx_r = NULL;
  __Pyx_RefNannyDeclarations
  PyObject *__pyx_t_1 = NULL;
  int __pyx_lineno = 0;
  const char *__pyx_filename = NULL;
  int __pyx_clineno = 0;
  static PyObject **__pyx_pyargnames[] = {&__pyx_n_s__phi,&__pyx_n_s__N,&__pyx_n_s__omega,&__pyx_n_s__chirp,&__pyx_n_s__damp,0};
  __Pyx_RefNannySetupContext("damped_");
  __pyx_self = __pyx_self;
  {
    PyObject* values[5] = {0,0,0,0,0};
    if (unlikely(__pyx_kwds)) {
      Py_ssize_t kw_args;
      switch (PyTuple_GET_SIZE(__pyx_args)) {
        case  5: values[4] = PyTuple_GET_ITEM(__pyx_args, 4);
        case  4: values[3] = PyTuple_GET_ITEM(__pyx_args, 3);
        case  3: values[2] = PyTuple_GET_ITEM(__pyx_args, 2);
        case  2: values[1] = PyTuple_GET_ITEM(__pyx_args, 1);
        case  1: values[0] = PyTuple_GET_ITEM(__pyx_args, 0);
        case  0: break;
        default: goto __pyx_L5_argtuple_error;
      }
      kw_args = PyDict_Size(__pyx_kwds);
      switch (PyTuple_GET_SIZE(__pyx_args)) {
        case  0:
        values[0] = PyDict_GetItem(__pyx_kwds, __pyx_n_s__phi);
        if (likely(values[0])) kw_args--;
        else goto __pyx_L5_argtuple_error;
        case  1:
        values[1] = PyDict_GetItem(__pyx_kwds, __pyx_n_s__N);
        if (likely(values[1])) kw_args--;
        else {
          __Pyx_RaiseArgtupleInvalid("damped_", 1, 5, 5, 1); {__pyx_filename = __pyx_f[0]; __pyx_lineno = 202; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
        }
        case  2:
        values[2] = PyDict_GetItem(__pyx_kwds, __pyx_n_s__omega);
        if (likely(values[2])) kw_args--;
        else {
          __Pyx_RaiseArgtupleInvalid("damped_", 1, 5, 5, 2); {__pyx_filename = __pyx_f[0]; __pyx_lineno = 202; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
        }
        case  3:
        values[3] = PyDict_GetItem(__pyx_kwds, __pyx_n_s__chirp);
        if (likely(values[3])) kw_args--;
        else {
          __Pyx_RaiseArgtupleInvalid("damped_", 1, 5, 5, 3); {__pyx_filename = __pyx_f[0]; __pyx_lineno = 202; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
        }
        case  4:
        values[4] = PyDict_GetItem(__pyx_kwds, __pyx_n_s__damp);
        if (likely(values[4])) kw_args--;
        else {
          __Pyx_RaiseArgtupleInvalid("damped_", 1, 5, 5, 4); {__pyx_filename = __pyx_f[0]; __pyx_lineno = 202; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
        }
      }
      if (unlikely(kw_args > 0)) {
        if (unlikely(__Pyx_ParseOptionalKeywords(__pyx_kwds, __pyx_pyargnames, 0, values, PyTuple_GET_SIZE(__pyx_args), "damped_") < 0)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 202; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
      }
    } else if (PyTuple_GET_SIZE(__pyx_args) != 5) {
      goto __pyx_L5_argtuple_error;
    } else {
      values[0] = PyTuple_GET_ITEM(__pyx_args, 0);
      values[1] = PyTuple_GET_ITEM(__pyx_args, 1);
      values[2] = PyTuple_GET_ITEM(__pyx_args, 2);
      values[3] = PyTuple_GET_ITEM(__pyx_args, 3);
      values[4] = PyTuple_GET_ITEM(__pyx_args, 4);
    }
    __pyx_v_phi = __pyx_PyFloat_AsDouble(values[0]); if (unlikely((__pyx_v_phi == (float)-1) && PyErr_Occurred())) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 202; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
    __pyx_v_N = __Pyx_PyInt_AsInt(values[1]); if (unlikely((__pyx_v_N == (int)-1) && PyErr_Occurred())) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 202; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
    __pyx_v_omega = __pyx_PyFloat_AsDouble(values[2]); if (unlikely((__pyx_v_omega == (float)-1) && PyErr_Occurred())) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 202; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
    __pyx_v_chirp = __pyx_PyFloat_AsDouble(values[3]); if (unlikely((__pyx_v_chirp == (float)-1) && PyErr_Occurred())) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 202; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
    __pyx_v_damp = __pyx_PyFloat_AsDouble(values[4]); if (unlikely((__pyx_v_damp == (float)-1) && PyErr_Occurred())) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 202; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
  }
  goto __pyx_L4_argument_unpacking_done;
  __pyx_L5_argtuple_error:;
  __Pyx_RaiseArgtupleInvalid("damped_", 1, 5, 5, PyTuple_GET_SIZE(__pyx_args)); {__pyx_filename = __pyx_f[0]; __pyx_lineno = 202; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
  __pyx_L3_error:;
  __Pyx_AddTraceback("atom_.damped_", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __Pyx_RefNannyFinishContext();
  return NULL;
  __pyx_L4_argument_unpacking_done:;
  __Pyx_XDECREF(__pyx_r);
  __pyx_t_1 = ((PyObject *)__pyx_f_5atom__damped_(__pyx_v_phi, __pyx_v_N, __pyx_v_omega, __pyx_v_chirp, __pyx_v_damp, 0)); if (unlikely(!__pyx_t_1)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 202; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(__pyx_t_1);
  __pyx_r = __pyx_t_1;
  __pyx_t_1 = 0;
  goto __pyx_L0;

  __pyx_r = Py_None; __Pyx_INCREF(Py_None);
  goto __pyx_L0;
  __pyx_L1_error:;
  __Pyx_XDECREF(__pyx_t_1);
  __Pyx_AddTraceback("atom_.damped_", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __pyx_r = NULL;
  __pyx_L0:;
  __Pyx_XGIVEREF(__pyx_r);
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

/* "atom_.pyx":216
 * @cython.profile(False)
 * @cython.boundscheck(False)
 * cpdef np.ndarray[np.float_t, ndim=1, mode="c"] dampedFM_(float phi, int N, float omega, float chirp, float damp, float omega_m=0., float phi_m=0., float depth=0.):             # <<<<<<<<<<<<<<
 * 
 *     cdef np.ndarray[np.float_t, ndim=1, mode="c"] out = np.zeros(N, dtype=float)
 */

static PyObject *__pyx_pf_5atom__10dampedFM_(PyObject *__pyx_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
static PyArrayObject *__pyx_f_5atom__dampedFM_(float __pyx_v_phi, int __pyx_v_N, float __pyx_v_omega, float __pyx_v_chirp, float __pyx_v_damp, int __pyx_skip_dispatch, struct __pyx_opt_args_5atom__dampedFM_ *__pyx_optional_args) {
  float __pyx_v_omega_m = ((float)0.);
  float __pyx_v_phi_m = ((float)0.);
  float __pyx_v_depth = ((float)0.);
  PyArrayObject *__pyx_v_out = 0;
  int __pyx_v_i;
  Py_buffer __pyx_bstruct_out;
  Py_ssize_t __pyx_bstride_0_out = 0;
  Py_ssize_t __pyx_bshape_0_out = 0;
  PyArrayObject *__pyx_r = NULL;
  __Pyx_RefNannyDeclarations
  PyObject *__pyx_t_1 = NULL;
  PyObject *__pyx_t_2 = NULL;
  PyObject *__pyx_t_3 = NULL;
  PyObject *__pyx_t_4 = NULL;
  PyArrayObject *__pyx_t_5 = NULL;
  int __pyx_t_6;
  int __pyx_t_7;
  int __pyx_t_8;
  int __pyx_lineno = 0;
  const char *__pyx_filename = NULL;
  int __pyx_clineno = 0;
  __Pyx_RefNannySetupContext("dampedFM_");
  if (__pyx_optional_args) {
    if (__pyx_optional_args->__pyx_n > 0) {
      __pyx_v_omega_m = __pyx_optional_args->omega_m;
      if (__pyx_optional_args->__pyx_n > 1) {
        __pyx_v_phi_m = __pyx_optional_args->phi_m;
        if (__pyx_optional_args->__pyx_n > 2) {
          __pyx_v_depth = __pyx_optional_args->depth;
        }
      }
    }
  }
  __pyx_bstruct_out.buf = NULL;

  /* "atom_.pyx":218
 * cpdef np.ndarray[np.float_t, ndim=1, mode="c"] dampedFM_(float phi, int N, float omega, float chirp, float damp, float omega_m=0., float phi_m=0., float depth=0.):
 * 
 *     cdef np.ndarray[np.float_t, ndim=1, mode="c"] out = np.zeros(N, dtype=float)             # <<<<<<<<<<<<<<
 *     cdef int i
 * 
 */
  __pyx_t_1 = __Pyx_GetName(__pyx_m, __pyx_n_s__np); if (unlikely(!__pyx_t_1)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 218; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(__pyx_t_1);
  __pyx_t_2 = PyObject_GetAttr(__pyx_t_1, __pyx_n_s__zeros); if (unlikely(!__pyx_t_2)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 218; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(__pyx_t_2);
  __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;
  __pyx_t_1 = PyInt_FromLong(__pyx_v_N); if (unlikely(!__pyx_t_1)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 218; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(__pyx_t_1);
  __pyx_t_3 = PyTuple_New(1); if (unlikely(!__pyx_t_3)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 218; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(((PyObject *)__pyx_t_3));
  PyTuple_SET_ITEM(__pyx_t_3, 0, __pyx_t_1);
  __Pyx_GIVEREF(__pyx_t_1);
  __pyx_t_1 = 0;
  __pyx_t_1 = PyDict_New(); if (unlikely(!__pyx_t_1)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 218; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(((PyObject *)__pyx_t_1));
  if (PyDict_SetItem(__pyx_t_1, ((PyObject *)__pyx_n_s__dtype), ((PyObject *)((PyObject*)(&PyFloat_Type)))) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 218; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __pyx_t_4 = PyEval_CallObjectWithKeywords(__pyx_t_2, ((PyObject *)__pyx_t_3), ((PyObject *)__pyx_t_1)); if (unlikely(!__pyx_t_4)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 218; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(__pyx_t_4);
  __Pyx_DECREF(__pyx_t_2); __pyx_t_2 = 0;
  __Pyx_DECREF(((PyObject *)__pyx_t_3)); __pyx_t_3 = 0;
  __Pyx_DECREF(((PyObject *)__pyx_t_1)); __pyx_t_1 = 0;
  if (!(likely(((__pyx_t_4) == Py_None) || likely(__Pyx_TypeTest(__pyx_t_4, __pyx_ptype_5numpy_ndarray))))) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 218; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __pyx_t_5 = ((PyArrayObject *)__pyx_t_4);
  {
    __Pyx_BufFmt_StackElem __pyx_stack[1];
    if (unlikely(__Pyx_GetBufferAndValidate(&__pyx_bstruct_out, (PyObject*)__pyx_t_5, &__Pyx_TypeInfo_nn___pyx_t_5numpy_float_t, PyBUF_FORMAT| PyBUF_C_CONTIGUOUS| PyBUF_WRITABLE, 1, 0, __pyx_stack) == -1)) {
      __pyx_v_out = ((PyArrayObject *)Py_None); __Pyx_INCREF(Py_None); __pyx_bstruct_out.buf = NULL;
      {__pyx_filename = __pyx_f[0]; __pyx_lineno = 218; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
    } else {__pyx_bstride_0_out = __pyx_bstruct_out.strides[0];
      __pyx_bshape_0_out = __pyx_bstruct_out.shape[0];
    }
  }
  __pyx_t_5 = 0;
  __pyx_v_out = ((PyArrayObject *)__pyx_t_4);
  __pyx_t_4 = 0;

  /* "atom_.pyx":221
 *     cdef int i
 * 
 *     for i in range(N):             # <<<<<<<<<<<<<<
 *         out[i] = exp(-damp * i) * realSinusoidFM(i, omega, chirp, phi, omega_m, phi_m, depth)
 * 
 */
  __pyx_t_6 = __pyx_v_N;
  for (__pyx_t_7 = 0; __pyx_t_7 < __pyx_t_6; __pyx_t_7+=1) {
    __pyx_v_i = __pyx_t_7;

    /* "atom_.pyx":222
 * 
 *     for i in range(N):
 *         out[i] = exp(-damp * i) * realSinusoidFM(i, omega, chirp, phi, omega_m, phi_m, depth)             # <<<<<<<<<<<<<<
 * 
 *     return out
 */
    __pyx_t_4 = PyFloat_FromDouble(__pyx_v_depth); if (unlikely(!__pyx_t_4)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 222; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
    __Pyx_GOTREF(__pyx_t_4);
    __pyx_t_8 = __pyx_v_i;
    if (__pyx_t_8 < 0) __pyx_t_8 += __pyx_bshape_0_out;
    *__Pyx_BufPtrCContig1d(__pyx_t_5numpy_float_t *, __pyx_bstruct_out.buf, __pyx_t_8, __pyx_bstride_0_out) = (exp(((-__pyx_v_damp) * __pyx_v_i)) * __pyx_f_5atom__realSinusoidFM(__pyx_v_i, __pyx_v_omega, __pyx_v_chirp, __pyx_v_phi, __pyx_v_omega_m, __pyx_v_phi_m, __pyx_t_4));
    __Pyx_DECREF(__pyx_t_4); __pyx_t_4 = 0;
  }

  /* "atom_.pyx":224
 *         out[i] = exp(-damp * i) * realSinusoidFM(i, omega, chirp, phi, omega_m, phi_m, depth)
 * 
 *     return out             # <<<<<<<<<<<<<<
 * 
 * 
 */
  __Pyx_XDECREF(((PyObject *)__pyx_r));
  __Pyx_INCREF(((PyObject *)__pyx_v_out));
  __pyx_r = ((PyArrayObject *)__pyx_v_out);
  goto __pyx_L0;

  __pyx_r = ((PyArrayObject *)Py_None); __Pyx_INCREF(Py_None);
  goto __pyx_L0;
  __pyx_L1_error:;
  __Pyx_XDECREF(__pyx_t_1);
  __Pyx_XDECREF(__pyx_t_2);
  __Pyx_XDECREF(__pyx_t_3);
  __Pyx_XDECREF(__pyx_t_4);
  { PyObject *__pyx_type, *__pyx_value, *__pyx_tb;
    __Pyx_ErrFetch(&__pyx_type, &__pyx_value, &__pyx_tb);
    __Pyx_SafeReleaseBuffer(&__pyx_bstruct_out);
  __Pyx_ErrRestore(__pyx_type, __pyx_value, __pyx_tb);}
  __Pyx_AddTraceback("atom_.dampedFM_", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __pyx_r = 0;
  goto __pyx_L2;
  __pyx_L0:;
  __Pyx_SafeReleaseBuffer(&__pyx_bstruct_out);
  __pyx_L2:;
  __Pyx_XDECREF((PyObject *)__pyx_v_out);
  __Pyx_XGIVEREF((PyObject *)__pyx_r);
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

/* "atom_.pyx":216
 * @cython.profile(False)
 * @cython.boundscheck(False)
 * cpdef np.ndarray[np.float_t, ndim=1, mode="c"] dampedFM_(float phi, int N, float omega, float chirp, float damp, float omega_m=0., float phi_m=0., float depth=0.):             # <<<<<<<<<<<<<<
 * 
 *     cdef np.ndarray[np.float_t, ndim=1, mode="c"] out = np.zeros(N, dtype=float)
 */

static PyObject *__pyx_pf_5atom__10dampedFM_(PyObject *__pyx_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
static PyObject *__pyx_pf_5atom__10dampedFM_(PyObject *__pyx_self, PyObject *__pyx_args, PyObject *__pyx_kwds) {
  float __pyx_v_phi;
  int __pyx_v_N;
  float __pyx_v_omega;
  float __pyx_v_chirp;
  float __pyx_v_damp;
  float __pyx_v_omega_m;
  float __pyx_v_phi_m;
  float __pyx_v_depth;
  PyObject *__pyx_r = NULL;
  __Pyx_RefNannyDeclarations
  PyObject *__pyx_t_1 = NULL;
  struct __pyx_opt_args_5atom__dampedFM_ __pyx_t_2;
  int __pyx_lineno = 0;
  const char *__pyx_filename = NULL;
  int __pyx_clineno = 0;
  static PyObject **__pyx_pyargnames[] = {&__pyx_n_s__phi,&__pyx_n_s__N,&__pyx_n_s__omega,&__pyx_n_s__chirp,&__pyx_n_s__damp,&__pyx_n_s__omega_m,&__pyx_n_s__phi_m,&__pyx_n_s__depth,0};
  __Pyx_RefNannySetupContext("dampedFM_");
  __pyx_self = __pyx_self;
  {
    PyObject* values[8] = {0,0,0,0,0,0,0,0};
    if (unlikely(__pyx_kwds)) {
      Py_ssize_t kw_args;
      switch (PyTuple_GET_SIZE(__pyx_args)) {
        case  8: values[7] = PyTuple_GET_ITEM(__pyx_args, 7);
        case  7: values[6] = PyTuple_GET_ITEM(__pyx_args, 6);
        case  6: values[5] = PyTuple_GET_ITEM(__pyx_args, 5);
        case  5: values[4] = PyTuple_GET_ITEM(__pyx_args, 4);
        case  4: values[3] = PyTuple_GET_ITEM(__pyx_args, 3);
        case  3: values[2] = PyTuple_GET_ITEM(__pyx_args, 2);
        case  2: values[1] = PyTuple_GET_ITEM(__pyx_args, 1);
        case  1: values[0] = PyTuple_GET_ITEM(__pyx_args, 0);
        case  0: break;
        default: goto __pyx_L5_argtuple_error;
      }
      kw_args = PyDict_Size(__pyx_kwds);
      switch (PyTuple_GET_SIZE(__pyx_args)) {
        case  0:
        values[0] = PyDict_GetItem(__pyx_kwds, __pyx_n_s__phi);
        if (likely(values[0])) kw_args--;
        else goto __pyx_L5_argtuple_error;
        case  1:
        values[1] = PyDict_GetItem(__pyx_kwds, __pyx_n_s__N);
        if (likely(values[1])) kw_args--;
        else {
          __Pyx_RaiseArgtupleInvalid("dampedFM_", 0, 5, 8, 1); {__pyx_filename = __pyx_f[0]; __pyx_lineno = 216; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
        }
        case  2:
        values[2] = PyDict_GetItem(__pyx_kwds, __pyx_n_s__omega);
        if (likely(values[2])) kw_args--;
        else {
          __Pyx_RaiseArgtupleInvalid("dampedFM_", 0, 5, 8, 2); {__pyx_filename = __pyx_f[0]; __pyx_lineno = 216; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
        }
        case  3:
        values[3] = PyDict_GetItem(__pyx_kwds, __pyx_n_s__chirp);
        if (likely(values[3])) kw_args--;
        else {
          __Pyx_RaiseArgtupleInvalid("dampedFM_", 0, 5, 8, 3); {__pyx_filename = __pyx_f[0]; __pyx_lineno = 216; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
        }
        case  4:
        values[4] = PyDict_GetItem(__pyx_kwds, __pyx_n_s__damp);
        if (likely(values[4])) kw_args--;
        else {
          __Pyx_RaiseArgtupleInvalid("dampedFM_", 0, 5, 8, 4); {__pyx_filename = __pyx_f[0]; __pyx_lineno = 216; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
        }
        case  5:
        if (kw_args > 0) {
          PyObject* value = PyDict_GetItem(__pyx_kwds, __pyx_n_s__omega_m);
          if (value) { values[5] = value; kw_args--; }
        }
        case  6:
        if (kw_args > 0) {
          PyObject* value = PyDict_GetItem(__pyx_kwds, __pyx_n_s__phi_m);
          if (value) { values[6] = value; kw_args--; }
        }
        case  7:
        if (kw_args > 0) {
          PyObject* value = PyDict_GetItem(__pyx_kwds, __pyx_n_s__depth);
          if (value) { values[7] = value; kw_args--; }
        }
      }
      if (unlikely(kw_args > 0)) {
        if (unlikely(__Pyx_ParseOptionalKeywords(__pyx_kwds, __pyx_pyargnames, 0, values, PyTuple_GET_SIZE(__pyx_args), "dampedFM_") < 0)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 216; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
      }
    } else {
      switch (PyTuple_GET_SIZE(__pyx_args)) {
        case  8: values[7] = PyTuple_GET_ITEM(__pyx_args, 7);
        case  7: values[6] = PyTuple_GET_ITEM(__pyx_args, 6);
        case  6: values[5] = PyTuple_GET_ITEM(__pyx_args, 5);
        case  5: values[4] = PyTuple_GET_ITEM(__pyx_args, 4);
        values[3] = PyTuple_GET_ITEM(__pyx_args, 3);
        values[2] = PyTuple_GET_ITEM(__pyx_args, 2);
        values[1] = PyTuple_GET_ITEM(__pyx_args, 1);
        values[0] = PyTuple_GET_ITEM(__pyx_args, 0);
        break;
        default: goto __pyx_L5_argtuple_error;
      }
    }
    __pyx_v_phi = __pyx_PyFloat_AsDouble(values[0]); if (unlikely((__pyx_v_phi == (float)-1) && PyErr_Occurred())) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 216; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
    __pyx_v_N = __Pyx_PyInt_AsInt(values[1]); if (unlikely((__pyx_v_N == (int)-1) && PyErr_Occurred())) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 216; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
    __pyx_v_omega = __pyx_PyFloat_AsDouble(values[2]); if (unlikely((__pyx_v_omega == (float)-1) && PyErr_Occurred())) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 216; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
    __pyx_v_chirp = __pyx_PyFloat_AsDouble(values[3]); if (unlikely((__pyx_v_chirp == (float)-1) && PyErr_Occurred())) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 216; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
    __pyx_v_damp = __pyx_PyFloat_AsDouble(values[4]); if (unlikely((__pyx_v_damp == (float)-1) && PyErr_Occurred())) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 216; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
    if (values[5]) {
      __pyx_v_omega_m = __pyx_PyFloat_AsDouble(values[5]); if (unlikely((__pyx_v_omega_m == (float)-1) && PyErr_Occurred())) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 216; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
    } else {
      __pyx_v_omega_m = ((float)0.);
    }
    if (values[6]) {
      __pyx_v_phi_m = __pyx_PyFloat_AsDouble(values[6]); if (unlikely((__pyx_v_phi_m == (float)-1) && PyErr_Occurred())) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 216; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
    } else {
      __pyx_v_phi_m = ((float)0.);
    }
    if (values[7]) {
      __pyx_v_depth = __pyx_PyFloat_AsDouble(values[7]); if (unlikely((__pyx_v_depth == (float)-1) && PyErr_Occurred())) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 216; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
    } else {
      __pyx_v_depth = ((float)0.);
    }
  }
  goto __pyx_L4_argument_unpacking_done;
  __pyx_L5_argtuple_error:;
  __Pyx_RaiseArgtupleInvalid("dampedFM_", 0, 5, 8, PyTuple_GET_SIZE(__pyx_args)); {__pyx_filename = __pyx_f[0]; __pyx_lineno = 216; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
  __pyx_L3_error:;
  __Pyx_AddTraceback("atom_.dampedFM_", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __Pyx_RefNannyFinishContext();
  return NULL;
  __pyx_L4_argument_unpacking_done:;
  __Pyx_XDECREF(__pyx_r);
  __pyx_t_2.__pyx_n = 3;
  __pyx_t_2.omega_m = __pyx_v_omega_m;
  __pyx_t_2.phi_m = __pyx_v_phi_m;
  __pyx_t_2.depth = __pyx_v_depth;
  __pyx_t_1 = ((PyObject *)__pyx_f_5atom__dampedFM_(__pyx_v_phi, __pyx_v_N, __pyx_v_omega, __pyx_v_chirp, __pyx_v_damp, 0, &__pyx_t_2)); if (unlikely(!__pyx_t_1)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 216; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(__pyx_t_1);
  __pyx_r = __pyx_t_1;
  __pyx_t_1 = 0;
  goto __pyx_L0;

  __pyx_r = Py_None; __Pyx_INCREF(Py_None);
  goto __pyx_L0;
  __pyx_L1_error:;
  __Pyx_XDECREF(__pyx_t_1);
  __Pyx_AddTraceback("atom_.dampedFM_", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __pyx_r = NULL;
  __pyx_L0:;
  __Pyx_XGIVEREF(__pyx_r);
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

/* "atom_.pyx":232
 * @cython.profile(False)
 * @cython.boundscheck(False)
 * cpdef np.ndarray[np.float_t, ndim=1, mode="c"] fof_(float phi, int N, float omega, float chirp, int rise_n, int decay_n):             # <<<<<<<<<<<<<<
 * 
 *     cdef np.ndarray[np.float_t, ndim=1, mode='c'] out = np.zeros(N, dtype=float)
 */

static PyObject *__pyx_pf_5atom__11fof_(PyObject *__pyx_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
static PyArrayObject *__pyx_f_5atom__fof_(float __pyx_v_phi, int __pyx_v_N, float __pyx_v_omega, float __pyx_v_chirp, int __pyx_v_rise_n, int __pyx_v_decay_n, int __pyx_skip_dispatch) {
  PyArrayObject *__pyx_v_out = 0;
  int __pyx_v_t;
  float __pyx_v_op;
  float __pyx_v_factor;
  float __pyx_v_p;
  float __pyx_v_a;
  int __pyx_v_i;
  Py_buffer __pyx_bstruct_out;
  Py_ssize_t __pyx_bstride_0_out = 0;
  Py_ssize_t __pyx_bshape_0_out = 0;
  PyArrayObject *__pyx_r = NULL;
  __Pyx_RefNannyDeclarations
  PyObject *__pyx_t_1 = NULL;
  PyObject *__pyx_t_2 = NULL;
  PyObject *__pyx_t_3 = NULL;
  PyObject *__pyx_t_4 = NULL;
  PyArrayObject *__pyx_t_5 = NULL;
  double __pyx_t_6;
  int __pyx_t_7;
  int __pyx_t_8;
  PyObject *__pyx_t_9 = NULL;
  __pyx_t_5numpy_float_t __pyx_t_10;
  int __pyx_t_11;
  long __pyx_t_12;
  long __pyx_t_13;
  float __pyx_t_14;
  long __pyx_t_15;
  int __pyx_t_16;
  int __pyx_t_17;
  int __pyx_t_18;
  int __pyx_t_19;
  int __pyx_lineno = 0;
  const char *__pyx_filename = NULL;
  int __pyx_clineno = 0;
  __Pyx_RefNannySetupContext("fof_");
  __pyx_bstruct_out.buf = NULL;

  /* "atom_.pyx":234
 * cpdef np.ndarray[np.float_t, ndim=1, mode="c"] fof_(float phi, int N, float omega, float chirp, int rise_n, int decay_n):
 * 
 *     cdef np.ndarray[np.float_t, ndim=1, mode='c'] out = np.zeros(N, dtype=float)             # <<<<<<<<<<<<<<
 *     cdef int t
 *     cdef float op = log(decay_n) / decay_n
 */
  __pyx_t_1 = __Pyx_GetName(__pyx_m, __pyx_n_s__np); if (unlikely(!__pyx_t_1)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 234; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(__pyx_t_1);
  __pyx_t_2 = PyObject_GetAttr(__pyx_t_1, __pyx_n_s__zeros); if (unlikely(!__pyx_t_2)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 234; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(__pyx_t_2);
  __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;
  __pyx_t_1 = PyInt_FromLong(__pyx_v_N); if (unlikely(!__pyx_t_1)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 234; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(__pyx_t_1);
  __pyx_t_3 = PyTuple_New(1); if (unlikely(!__pyx_t_3)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 234; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(((PyObject *)__pyx_t_3));
  PyTuple_SET_ITEM(__pyx_t_3, 0, __pyx_t_1);
  __Pyx_GIVEREF(__pyx_t_1);
  __pyx_t_1 = 0;
  __pyx_t_1 = PyDict_New(); if (unlikely(!__pyx_t_1)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 234; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(((PyObject *)__pyx_t_1));
  if (PyDict_SetItem(__pyx_t_1, ((PyObject *)__pyx_n_s__dtype), ((PyObject *)((PyObject*)(&PyFloat_Type)))) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 234; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __pyx_t_4 = PyEval_CallObjectWithKeywords(__pyx_t_2, ((PyObject *)__pyx_t_3), ((PyObject *)__pyx_t_1)); if (unlikely(!__pyx_t_4)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 234; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(__pyx_t_4);
  __Pyx_DECREF(__pyx_t_2); __pyx_t_2 = 0;
  __Pyx_DECREF(((PyObject *)__pyx_t_3)); __pyx_t_3 = 0;
  __Pyx_DECREF(((PyObject *)__pyx_t_1)); __pyx_t_1 = 0;
  if (!(likely(((__pyx_t_4) == Py_None) || likely(__Pyx_TypeTest(__pyx_t_4, __pyx_ptype_5numpy_ndarray))))) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 234; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __pyx_t_5 = ((PyArrayObject *)__pyx_t_4);
  {
    __Pyx_BufFmt_StackElem __pyx_stack[1];
    if (unlikely(__Pyx_GetBufferAndValidate(&__pyx_bstruct_out, (PyObject*)__pyx_t_5, &__Pyx_TypeInfo_nn___pyx_t_5numpy_float_t, PyBUF_FORMAT| PyBUF_C_CONTIGUOUS| PyBUF_WRITABLE, 1, 0, __pyx_stack) == -1)) {
      __pyx_v_out = ((PyArrayObject *)Py_None); __Pyx_INCREF(Py_None); __pyx_bstruct_out.buf = NULL;
      {__pyx_filename = __pyx_f[0]; __pyx_lineno = 234; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
    } else {__pyx_bstride_0_out = __pyx_bstruct_out.strides[0];
      __pyx_bshape_0_out = __pyx_bstruct_out.shape[0];
    }
  }
  __pyx_t_5 = 0;
  __pyx_v_out = ((PyArrayObject *)__pyx_t_4);
  __pyx_t_4 = 0;

  /* "atom_.pyx":236
 *     cdef np.ndarray[np.float_t, ndim=1, mode='c'] out = np.zeros(N, dtype=float)
 *     cdef int t
 *     cdef float op = log(decay_n) / decay_n             # <<<<<<<<<<<<<<
 *     cdef float factor = pi/rise_n
 *     cdef float p, a
 */
  __pyx_t_6 = log(__pyx_v_decay_n);
  if (unlikely(__pyx_v_decay_n == 0)) {
    PyErr_Format(PyExc_ZeroDivisionError, "float division");
    {__pyx_filename = __pyx_f[0]; __pyx_lineno = 236; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  }
  __pyx_v_op = (__pyx_t_6 / __pyx_v_decay_n);

  /* "atom_.pyx":237
 *     cdef int t
 *     cdef float op = log(decay_n) / decay_n
 *     cdef float factor = pi/rise_n             # <<<<<<<<<<<<<<
 *     cdef float p, a
 * 
 */
  if (unlikely(__pyx_v_rise_n == 0)) {
    PyErr_Format(PyExc_ZeroDivisionError, "float division");
    {__pyx_filename = __pyx_f[0]; __pyx_lineno = 237; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  }
  __pyx_v_factor = (__pyx_v_5atom__pi / __pyx_v_rise_n);

  /* "atom_.pyx":240
 *     cdef float p, a
 * 
 *     for t in range(rise_n):             # <<<<<<<<<<<<<<
 *         out[t] = 0.5 * (1. - np.cos(factor * t) * exp(-op * t))
 * 
 */
  __pyx_t_7 = __pyx_v_rise_n;
  for (__pyx_t_8 = 0; __pyx_t_8 < __pyx_t_7; __pyx_t_8+=1) {
    __pyx_v_t = __pyx_t_8;

    /* "atom_.pyx":241
 * 
 *     for t in range(rise_n):
 *         out[t] = 0.5 * (1. - np.cos(factor * t) * exp(-op * t))             # <<<<<<<<<<<<<<
 * 
 *     p = out[rise_n-1]
 */
    __pyx_t_4 = PyFloat_FromDouble(0.5); if (unlikely(!__pyx_t_4)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 241; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
    __Pyx_GOTREF(__pyx_t_4);
    __pyx_t_1 = PyFloat_FromDouble(1.); if (unlikely(!__pyx_t_1)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 241; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
    __Pyx_GOTREF(__pyx_t_1);
    __pyx_t_3 = __Pyx_GetName(__pyx_m, __pyx_n_s__np); if (unlikely(!__pyx_t_3)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 241; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
    __Pyx_GOTREF(__pyx_t_3);
    __pyx_t_2 = PyObject_GetAttr(__pyx_t_3, __pyx_n_s__cos); if (unlikely(!__pyx_t_2)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 241; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
    __Pyx_GOTREF(__pyx_t_2);
    __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
    __pyx_t_3 = PyFloat_FromDouble((__pyx_v_factor * __pyx_v_t)); if (unlikely(!__pyx_t_3)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 241; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
    __Pyx_GOTREF(__pyx_t_3);
    __pyx_t_9 = PyTuple_New(1); if (unlikely(!__pyx_t_9)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 241; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
    __Pyx_GOTREF(((PyObject *)__pyx_t_9));
    PyTuple_SET_ITEM(__pyx_t_9, 0, __pyx_t_3);
    __Pyx_GIVEREF(__pyx_t_3);
    __pyx_t_3 = 0;
    __pyx_t_3 = PyObject_Call(__pyx_t_2, ((PyObject *)__pyx_t_9), NULL); if (unlikely(!__pyx_t_3)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 241; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
    __Pyx_GOTREF(__pyx_t_3);
    __Pyx_DECREF(__pyx_t_2); __pyx_t_2 = 0;
    __Pyx_DECREF(((PyObject *)__pyx_t_9)); __pyx_t_9 = 0;
    __pyx_t_9 = PyFloat_FromDouble(exp(((-__pyx_v_op) * __pyx_v_t))); if (unlikely(!__pyx_t_9)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 241; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
    __Pyx_GOTREF(__pyx_t_9);
    __pyx_t_2 = PyNumber_Multiply(__pyx_t_3, __pyx_t_9); if (unlikely(!__pyx_t_2)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 241; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
    __Pyx_GOTREF(__pyx_t_2);
    __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
    __Pyx_DECREF(__pyx_t_9); __pyx_t_9 = 0;
    __pyx_t_9 = PyNumber_Subtract(__pyx_t_1, __pyx_t_2); if (unlikely(!__pyx_t_9)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 241; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
    __Pyx_GOTREF(__pyx_t_9);
    __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;
    __Pyx_DECREF(__pyx_t_2); __pyx_t_2 = 0;
    __pyx_t_2 = PyNumber_Multiply(__pyx_t_4, __pyx_t_9); if (unlikely(!__pyx_t_2)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 241; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
    __Pyx_GOTREF(__pyx_t_2);
    __Pyx_DECREF(__pyx_t_4); __pyx_t_4 = 0;
    __Pyx_DECREF(__pyx_t_9); __pyx_t_9 = 0;
    __pyx_t_10 = __pyx_PyFloat_AsDouble(__pyx_t_2); if (unlikely((__pyx_t_10 == (npy_double)-1) && PyErr_Occurred())) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 241; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
    __Pyx_DECREF(__pyx_t_2); __pyx_t_2 = 0;
    __pyx_t_11 = __pyx_v_t;
    if (__pyx_t_11 < 0) __pyx_t_11 += __pyx_bshape_0_out;
    *__Pyx_BufPtrCContig1d(__pyx_t_5numpy_float_t *, __pyx_bstruct_out.buf, __pyx_t_11, __pyx_bstride_0_out) = __pyx_t_10;
  }

  /* "atom_.pyx":243
 *         out[t] = 0.5 * (1. - np.cos(factor * t) * exp(-op * t))
 * 
 *     p = out[rise_n-1]             # <<<<<<<<<<<<<<
 *     for t in range(rise_n, N):
 *         out[t-1] = exp(-op*t)
 */
  __pyx_t_12 = (__pyx_v_rise_n - 1);
  if (__pyx_t_12 < 0) __pyx_t_12 += __pyx_bshape_0_out;
  __pyx_v_p = (*__Pyx_BufPtrCContig1d(__pyx_t_5numpy_float_t *, __pyx_bstruct_out.buf, __pyx_t_12, __pyx_bstride_0_out));

  /* "atom_.pyx":244
 * 
 *     p = out[rise_n-1]
 *     for t in range(rise_n, N):             # <<<<<<<<<<<<<<
 *         out[t-1] = exp(-op*t)
 * 
 */
  __pyx_t_7 = __pyx_v_N;
  for (__pyx_t_8 = __pyx_v_rise_n; __pyx_t_8 < __pyx_t_7; __pyx_t_8+=1) {
    __pyx_v_t = __pyx_t_8;

    /* "atom_.pyx":245
 *     p = out[rise_n-1]
 *     for t in range(rise_n, N):
 *         out[t-1] = exp(-op*t)             # <<<<<<<<<<<<<<
 * 
 *     a = max(abs(out[rise_n-1:N-1]))
 */
    __pyx_t_13 = (__pyx_v_t - 1);
    if (__pyx_t_13 < 0) __pyx_t_13 += __pyx_bshape_0_out;
    *__Pyx_BufPtrCContig1d(__pyx_t_5numpy_float_t *, __pyx_bstruct_out.buf, __pyx_t_13, __pyx_bstride_0_out) = exp(((-__pyx_v_op) * __pyx_v_t));
  }

  /* "atom_.pyx":247
 *         out[t-1] = exp(-op*t)
 * 
 *     a = max(abs(out[rise_n-1:N-1]))             # <<<<<<<<<<<<<<
 *     for t in range(rise_n-1, N-1):
 *         out[t] = out[t]/a  * p
 */
  __pyx_t_2 = __Pyx_PySequence_GetSlice(((PyObject *)__pyx_v_out), (__pyx_v_rise_n - 1), (__pyx_v_N - 1)); if (unlikely(!__pyx_t_2)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 247; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(__pyx_t_2);
  __pyx_t_9 = PyNumber_Absolute(__pyx_t_2); if (unlikely(!__pyx_t_9)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 247; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(__pyx_t_9);
  __Pyx_DECREF(__pyx_t_2); __pyx_t_2 = 0;
  __pyx_t_2 = PyTuple_New(1); if (unlikely(!__pyx_t_2)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 247; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(((PyObject *)__pyx_t_2));
  PyTuple_SET_ITEM(__pyx_t_2, 0, __pyx_t_9);
  __Pyx_GIVEREF(__pyx_t_9);
  __pyx_t_9 = 0;
  __pyx_t_9 = PyObject_Call(__pyx_builtin_max, ((PyObject *)__pyx_t_2), NULL); if (unlikely(!__pyx_t_9)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 247; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(__pyx_t_9);
  __Pyx_DECREF(((PyObject *)__pyx_t_2)); __pyx_t_2 = 0;
  __pyx_t_14 = __pyx_PyFloat_AsDouble(__pyx_t_9); if (unlikely((__pyx_t_14 == (float)-1) && PyErr_Occurred())) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 247; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_DECREF(__pyx_t_9); __pyx_t_9 = 0;
  __pyx_v_a = __pyx_t_14;

  /* "atom_.pyx":248
 * 
 *     a = max(abs(out[rise_n-1:N-1]))
 *     for t in range(rise_n-1, N-1):             # <<<<<<<<<<<<<<
 *         out[t] = out[t]/a  * p
 * 
 */
  __pyx_t_15 = (__pyx_v_N - 1);
  for (__pyx_t_7 = (__pyx_v_rise_n - 1); __pyx_t_7 < __pyx_t_15; __pyx_t_7+=1) {
    __pyx_v_t = __pyx_t_7;

    /* "atom_.pyx":249
 *     a = max(abs(out[rise_n-1:N-1]))
 *     for t in range(rise_n-1, N-1):
 *         out[t] = out[t]/a  * p             # <<<<<<<<<<<<<<
 * 
 *     for i in range(N):
 */
    __pyx_t_8 = __pyx_v_t;
    if (__pyx_t_8 < 0) __pyx_t_8 += __pyx_bshape_0_out;
    __pyx_t_10 = (*__Pyx_BufPtrCContig1d(__pyx_t_5numpy_float_t *, __pyx_bstruct_out.buf, __pyx_t_8, __pyx_bstride_0_out));
    if (unlikely(__pyx_v_a == 0)) {
      PyErr_Format(PyExc_ZeroDivisionError, "float division");
      {__pyx_filename = __pyx_f[0]; __pyx_lineno = 249; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
    }
    __pyx_t_16 = __pyx_v_t;
    if (__pyx_t_16 < 0) __pyx_t_16 += __pyx_bshape_0_out;
    *__Pyx_BufPtrCContig1d(__pyx_t_5numpy_float_t *, __pyx_bstruct_out.buf, __pyx_t_16, __pyx_bstride_0_out) = ((__pyx_t_10 / __pyx_v_a) * __pyx_v_p);
  }

  /* "atom_.pyx":251
 *         out[t] = out[t]/a  * p
 * 
 *     for i in range(N):             # <<<<<<<<<<<<<<
 *         out[i] = out[i] * realSinusoid(i, omega, chirp/N, phi)
 * 
 */
  __pyx_t_7 = __pyx_v_N;
  for (__pyx_t_17 = 0; __pyx_t_17 < __pyx_t_7; __pyx_t_17+=1) {
    __pyx_v_i = __pyx_t_17;

    /* "atom_.pyx":252
 * 
 *     for i in range(N):
 *         out[i] = out[i] * realSinusoid(i, omega, chirp/N, phi)             # <<<<<<<<<<<<<<
 * 
 *     return out
 */
    __pyx_t_18 = __pyx_v_i;
    if (__pyx_t_18 < 0) __pyx_t_18 += __pyx_bshape_0_out;
    if (unlikely(__pyx_v_N == 0)) {
      PyErr_Format(PyExc_ZeroDivisionError, "float division");
      {__pyx_filename = __pyx_f[0]; __pyx_lineno = 252; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
    }
    __pyx_t_19 = __pyx_v_i;
    if (__pyx_t_19 < 0) __pyx_t_19 += __pyx_bshape_0_out;
    *__Pyx_BufPtrCContig1d(__pyx_t_5numpy_float_t *, __pyx_bstruct_out.buf, __pyx_t_19, __pyx_bstride_0_out) = ((*__Pyx_BufPtrCContig1d(__pyx_t_5numpy_float_t *, __pyx_bstruct_out.buf, __pyx_t_18, __pyx_bstride_0_out)) * __pyx_f_5atom__realSinusoid(__pyx_v_i, __pyx_v_omega, (__pyx_v_chirp / __pyx_v_N), __pyx_v_phi));
  }

  /* "atom_.pyx":254
 *         out[i] = out[i] * realSinusoid(i, omega, chirp/N, phi)
 * 
 *     return out             # <<<<<<<<<<<<<<
 * 
 * #FOFFM
 */
  __Pyx_XDECREF(((PyObject *)__pyx_r));
  __Pyx_INCREF(((PyObject *)__pyx_v_out));
  __pyx_r = ((PyArrayObject *)__pyx_v_out);
  goto __pyx_L0;

  __pyx_r = ((PyArrayObject *)Py_None); __Pyx_INCREF(Py_None);
  goto __pyx_L0;
  __pyx_L1_error:;
  __Pyx_XDECREF(__pyx_t_1);
  __Pyx_XDECREF(__pyx_t_2);
  __Pyx_XDECREF(__pyx_t_3);
  __Pyx_XDECREF(__pyx_t_4);
  __Pyx_XDECREF(__pyx_t_9);
  { PyObject *__pyx_type, *__pyx_value, *__pyx_tb;
    __Pyx_ErrFetch(&__pyx_type, &__pyx_value, &__pyx_tb);
    __Pyx_SafeReleaseBuffer(&__pyx_bstruct_out);
  __Pyx_ErrRestore(__pyx_type, __pyx_value, __pyx_tb);}
  __Pyx_AddTraceback("atom_.fof_", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __pyx_r = 0;
  goto __pyx_L2;
  __pyx_L0:;
  __Pyx_SafeReleaseBuffer(&__pyx_bstruct_out);
  __pyx_L2:;
  __Pyx_XDECREF((PyObject *)__pyx_v_out);
  __Pyx_XGIVEREF((PyObject *)__pyx_r);
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

/* "atom_.pyx":232
 * @cython.profile(False)
 * @cython.boundscheck(False)
 * cpdef np.ndarray[np.float_t, ndim=1, mode="c"] fof_(float phi, int N, float omega, float chirp, int rise_n, int decay_n):             # <<<<<<<<<<<<<<
 * 
 *     cdef np.ndarray[np.float_t, ndim=1, mode='c'] out = np.zeros(N, dtype=float)
 */

static PyObject *__pyx_pf_5atom__11fof_(PyObject *__pyx_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
static PyObject *__pyx_pf_5atom__11fof_(PyObject *__pyx_self, PyObject *__pyx_args, PyObject *__pyx_kwds) {
  float __pyx_v_phi;
  int __pyx_v_N;
  float __pyx_v_omega;
  float __pyx_v_chirp;
  int __pyx_v_rise_n;
  int __pyx_v_decay_n;
  PyObject *__pyx_r = NULL;
  __Pyx_RefNannyDeclarations
  PyObject *__pyx_t_1 = NULL;
  int __pyx_lineno = 0;
  const char *__pyx_filename = NULL;
  int __pyx_clineno = 0;
  static PyObject **__pyx_pyargnames[] = {&__pyx_n_s__phi,&__pyx_n_s__N,&__pyx_n_s__omega,&__pyx_n_s__chirp,&__pyx_n_s__rise_n,&__pyx_n_s__decay_n,0};
  __Pyx_RefNannySetupContext("fof_");
  __pyx_self = __pyx_self;
  {
    PyObject* values[6] = {0,0,0,0,0,0};
    if (unlikely(__pyx_kwds)) {
      Py_ssize_t kw_args;
      switch (PyTuple_GET_SIZE(__pyx_args)) {
        case  6: values[5] = PyTuple_GET_ITEM(__pyx_args, 5);
        case  5: values[4] = PyTuple_GET_ITEM(__pyx_args, 4);
        case  4: values[3] = PyTuple_GET_ITEM(__pyx_args, 3);
        case  3: values[2] = PyTuple_GET_ITEM(__pyx_args, 2);
        case  2: values[1] = PyTuple_GET_ITEM(__pyx_args, 1);
        case  1: values[0] = PyTuple_GET_ITEM(__pyx_args, 0);
        case  0: break;
        default: goto __pyx_L5_argtuple_error;
      }
      kw_args = PyDict_Size(__pyx_kwds);
      switch (PyTuple_GET_SIZE(__pyx_args)) {
        case  0:
        values[0] = PyDict_GetItem(__pyx_kwds, __pyx_n_s__phi);
        if (likely(values[0])) kw_args--;
        else goto __pyx_L5_argtuple_error;
        case  1:
        values[1] = PyDict_GetItem(__pyx_kwds, __pyx_n_s__N);
        if (likely(values[1])) kw_args--;
        else {
          __Pyx_RaiseArgtupleInvalid("fof_", 1, 6, 6, 1); {__pyx_filename = __pyx_f[0]; __pyx_lineno = 232; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
        }
        case  2:
        values[2] = PyDict_GetItem(__pyx_kwds, __pyx_n_s__omega);
        if (likely(values[2])) kw_args--;
        else {
          __Pyx_RaiseArgtupleInvalid("fof_", 1, 6, 6, 2); {__pyx_filename = __pyx_f[0]; __pyx_lineno = 232; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
        }
        case  3:
        values[3] = PyDict_GetItem(__pyx_kwds, __pyx_n_s__chirp);
        if (likely(values[3])) kw_args--;
        else {
          __Pyx_RaiseArgtupleInvalid("fof_", 1, 6, 6, 3); {__pyx_filename = __pyx_f[0]; __pyx_lineno = 232; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
        }
        case  4:
        values[4] = PyDict_GetItem(__pyx_kwds, __pyx_n_s__rise_n);
        if (likely(values[4])) kw_args--;
        else {
          __Pyx_RaiseArgtupleInvalid("fof_", 1, 6, 6, 4); {__pyx_filename = __pyx_f[0]; __pyx_lineno = 232; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
        }
        case  5:
        values[5] = PyDict_GetItem(__pyx_kwds, __pyx_n_s__decay_n);
        if (likely(values[5])) kw_args--;
        else {
          __Pyx_RaiseArgtupleInvalid("fof_", 1, 6, 6, 5); {__pyx_filename = __pyx_f[0]; __pyx_lineno = 232; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
        }
      }
      if (unlikely(kw_args > 0)) {
        if (unlikely(__Pyx_ParseOptionalKeywords(__pyx_kwds, __pyx_pyargnames, 0, values, PyTuple_GET_SIZE(__pyx_args), "fof_") < 0)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 232; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
      }
    } else if (PyTuple_GET_SIZE(__pyx_args) != 6) {
      goto __pyx_L5_argtuple_error;
    } else {
      values[0] = PyTuple_GET_ITEM(__pyx_args, 0);
      values[1] = PyTuple_GET_ITEM(__pyx_args, 1);
      values[2] = PyTuple_GET_ITEM(__pyx_args, 2);
      values[3] = PyTuple_GET_ITEM(__pyx_args, 3);
      values[4] = PyTuple_GET_ITEM(__pyx_args, 4);
      values[5] = PyTuple_GET_ITEM(__pyx_args, 5);
    }
    __pyx_v_phi = __pyx_PyFloat_AsDouble(values[0]); if (unlikely((__pyx_v_phi == (float)-1) && PyErr_Occurred())) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 232; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
    __pyx_v_N = __Pyx_PyInt_AsInt(values[1]); if (unlikely((__pyx_v_N == (int)-1) && PyErr_Occurred())) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 232; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
    __pyx_v_omega = __pyx_PyFloat_AsDouble(values[2]); if (unlikely((__pyx_v_omega == (float)-1) && PyErr_Occurred())) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 232; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
    __pyx_v_chirp = __pyx_PyFloat_AsDouble(values[3]); if (unlikely((__pyx_v_chirp == (float)-1) && PyErr_Occurred())) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 232; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
    __pyx_v_rise_n = __Pyx_PyInt_AsInt(values[4]); if (unlikely((__pyx_v_rise_n == (int)-1) && PyErr_Occurred())) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 232; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
    __pyx_v_decay_n = __Pyx_PyInt_AsInt(values[5]); if (unlikely((__pyx_v_decay_n == (int)-1) && PyErr_Occurred())) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 232; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
  }
  goto __pyx_L4_argument_unpacking_done;
  __pyx_L5_argtuple_error:;
  __Pyx_RaiseArgtupleInvalid("fof_", 1, 6, 6, PyTuple_GET_SIZE(__pyx_args)); {__pyx_filename = __pyx_f[0]; __pyx_lineno = 232; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
  __pyx_L3_error:;
  __Pyx_AddTraceback("atom_.fof_", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __Pyx_RefNannyFinishContext();
  return NULL;
  __pyx_L4_argument_unpacking_done:;
  __Pyx_XDECREF(__pyx_r);
  __pyx_t_1 = ((PyObject *)__pyx_f_5atom__fof_(__pyx_v_phi, __pyx_v_N, __pyx_v_omega, __pyx_v_chirp, __pyx_v_rise_n, __pyx_v_decay_n, 0)); if (unlikely(!__pyx_t_1)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 232; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(__pyx_t_1);
  __pyx_r = __pyx_t_1;
  __pyx_t_1 = 0;
  goto __pyx_L0;

  __pyx_r = Py_None; __Pyx_INCREF(Py_None);
  goto __pyx_L0;
  __pyx_L1_error:;
  __Pyx_XDECREF(__pyx_t_1);
  __Pyx_AddTraceback("atom_.fof_", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __pyx_r = NULL;
  __pyx_L0:;
  __Pyx_XGIVEREF(__pyx_r);
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

/* "atom_.pyx":260
 * @cython.profile(False)
 * @cython.boundscheck(False)
 * cpdef np.ndarray[np.float_t, ndim=1, mode="c"] fofFM_(float phi, int N, float omega, float chirp, int rise_n, int decay_n, float omega_m=0., float phi_m=0., float depth=0.):             # <<<<<<<<<<<<<<
 * 
 *     cdef np.ndarray[np.float_t, ndim=1, mode='c'] out = np.zeros(N, dtype=float)
 */

static PyObject *__pyx_pf_5atom__12fofFM_(PyObject *__pyx_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
static PyArrayObject *__pyx_f_5atom__fofFM_(float __pyx_v_phi, int __pyx_v_N, float __pyx_v_omega, float __pyx_v_chirp, int __pyx_v_rise_n, int __pyx_v_decay_n, int __pyx_skip_dispatch, struct __pyx_opt_args_5atom__fofFM_ *__pyx_optional_args) {
  float __pyx_v_omega_m = ((float)0.);
  float __pyx_v_phi_m = ((float)0.);
  float __pyx_v_depth = ((float)0.);
  PyArrayObject *__pyx_v_out = 0;
  int __pyx_v_t;
  float __pyx_v_op;
  float __pyx_v_factor;
  float __pyx_v_p;
  float __pyx_v_a;
  int __pyx_v_i;
  Py_buffer __pyx_bstruct_out;
  Py_ssize_t __pyx_bstride_0_out = 0;
  Py_ssize_t __pyx_bshape_0_out = 0;
  PyArrayObject *__pyx_r = NULL;
  __Pyx_RefNannyDeclarations
  PyObject *__pyx_t_1 = NULL;
  PyObject *__pyx_t_2 = NULL;
  PyObject *__pyx_t_3 = NULL;
  PyObject *__pyx_t_4 = NULL;
  PyArrayObject *__pyx_t_5 = NULL;
  double __pyx_t_6;
  int __pyx_t_7;
  int __pyx_t_8;
  PyObject *__pyx_t_9 = NULL;
  __pyx_t_5numpy_float_t __pyx_t_10;
  int __pyx_t_11;
  long __pyx_t_12;
  long __pyx_t_13;
  float __pyx_t_14;
  long __pyx_t_15;
  int __pyx_t_16;
  int __pyx_t_17;
  int __pyx_t_18;
  int __pyx_t_19;
  int __pyx_lineno = 0;
  const char *__pyx_filename = NULL;
  int __pyx_clineno = 0;
  __Pyx_RefNannySetupContext("fofFM_");
  if (__pyx_optional_args) {
    if (__pyx_optional_args->__pyx_n > 0) {
      __pyx_v_omega_m = __pyx_optional_args->omega_m;
      if (__pyx_optional_args->__pyx_n > 1) {
        __pyx_v_phi_m = __pyx_optional_args->phi_m;
        if (__pyx_optional_args->__pyx_n > 2) {
          __pyx_v_depth = __pyx_optional_args->depth;
        }
      }
    }
  }
  __pyx_bstruct_out.buf = NULL;

  /* "atom_.pyx":262
 * cpdef np.ndarray[np.float_t, ndim=1, mode="c"] fofFM_(float phi, int N, float omega, float chirp, int rise_n, int decay_n, float omega_m=0., float phi_m=0., float depth=0.):
 * 
 *     cdef np.ndarray[np.float_t, ndim=1, mode='c'] out = np.zeros(N, dtype=float)             # <<<<<<<<<<<<<<
 *     cdef int t
 *     cdef float op = log(decay_n) / decay_n
 */
  __pyx_t_1 = __Pyx_GetName(__pyx_m, __pyx_n_s__np); if (unlikely(!__pyx_t_1)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 262; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(__pyx_t_1);
  __pyx_t_2 = PyObject_GetAttr(__pyx_t_1, __pyx_n_s__zeros); if (unlikely(!__pyx_t_2)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 262; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(__pyx_t_2);
  __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;
  __pyx_t_1 = PyInt_FromLong(__pyx_v_N); if (unlikely(!__pyx_t_1)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 262; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(__pyx_t_1);
  __pyx_t_3 = PyTuple_New(1); if (unlikely(!__pyx_t_3)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 262; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(((PyObject *)__pyx_t_3));
  PyTuple_SET_ITEM(__pyx_t_3, 0, __pyx_t_1);
  __Pyx_GIVEREF(__pyx_t_1);
  __pyx_t_1 = 0;
  __pyx_t_1 = PyDict_New(); if (unlikely(!__pyx_t_1)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 262; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(((PyObject *)__pyx_t_1));
  if (PyDict_SetItem(__pyx_t_1, ((PyObject *)__pyx_n_s__dtype), ((PyObject *)((PyObject*)(&PyFloat_Type)))) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 262; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __pyx_t_4 = PyEval_CallObjectWithKeywords(__pyx_t_2, ((PyObject *)__pyx_t_3), ((PyObject *)__pyx_t_1)); if (unlikely(!__pyx_t_4)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 262; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(__pyx_t_4);
  __Pyx_DECREF(__pyx_t_2); __pyx_t_2 = 0;
  __Pyx_DECREF(((PyObject *)__pyx_t_3)); __pyx_t_3 = 0;
  __Pyx_DECREF(((PyObject *)__pyx_t_1)); __pyx_t_1 = 0;
  if (!(likely(((__pyx_t_4) == Py_None) || likely(__Pyx_TypeTest(__pyx_t_4, __pyx_ptype_5numpy_ndarray))))) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 262; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __pyx_t_5 = ((PyArrayObject *)__pyx_t_4);
  {
    __Pyx_BufFmt_StackElem __pyx_stack[1];
    if (unlikely(__Pyx_GetBufferAndValidate(&__pyx_bstruct_out, (PyObject*)__pyx_t_5, &__Pyx_TypeInfo_nn___pyx_t_5numpy_float_t, PyBUF_FORMAT| PyBUF_C_CONTIGUOUS| PyBUF_WRITABLE, 1, 0, __pyx_stack) == -1)) {
      __pyx_v_out = ((PyArrayObject *)Py_None); __Pyx_INCREF(Py_None); __pyx_bstruct_out.buf = NULL;
      {__pyx_filename = __pyx_f[0]; __pyx_lineno = 262; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
    } else {__pyx_bstride_0_out = __pyx_bstruct_out.strides[0];
      __pyx_bshape_0_out = __pyx_bstruct_out.shape[0];
    }
  }
  __pyx_t_5 = 0;
  __pyx_v_out = ((PyArrayObject *)__pyx_t_4);
  __pyx_t_4 = 0;

  /* "atom_.pyx":264
 *     cdef np.ndarray[np.float_t, ndim=1, mode='c'] out = np.zeros(N, dtype=float)
 *     cdef int t
 *     cdef float op = log(decay_n) / decay_n             # <<<<<<<<<<<<<<
 *     cdef float factor = pi/rise_n
 *     cdef float p, a
 */
  __pyx_t_6 = log(__pyx_v_decay_n);
  if (unlikely(__pyx_v_decay_n == 0)) {
    PyErr_Format(PyExc_ZeroDivisionError, "float division");
    {__pyx_filename = __pyx_f[0]; __pyx_lineno = 264; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  }
  __pyx_v_op = (__pyx_t_6 / __pyx_v_decay_n);

  /* "atom_.pyx":265
 *     cdef int t
 *     cdef float op = log(decay_n) / decay_n
 *     cdef float factor = pi/rise_n             # <<<<<<<<<<<<<<
 *     cdef float p, a
 * 
 */
  if (unlikely(__pyx_v_rise_n == 0)) {
    PyErr_Format(PyExc_ZeroDivisionError, "float division");
    {__pyx_filename = __pyx_f[0]; __pyx_lineno = 265; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  }
  __pyx_v_factor = (__pyx_v_5atom__pi / __pyx_v_rise_n);

  /* "atom_.pyx":268
 *     cdef float p, a
 * 
 *     for t in range(rise_n):             # <<<<<<<<<<<<<<
 *         out[t] = 0.5 * (1. - np.cos(factor * t) * exp(-op * t))
 * 
 */
  __pyx_t_7 = __pyx_v_rise_n;
  for (__pyx_t_8 = 0; __pyx_t_8 < __pyx_t_7; __pyx_t_8+=1) {
    __pyx_v_t = __pyx_t_8;

    /* "atom_.pyx":269
 * 
 *     for t in range(rise_n):
 *         out[t] = 0.5 * (1. - np.cos(factor * t) * exp(-op * t))             # <<<<<<<<<<<<<<
 * 
 *     p = out[rise_n-1]
 */
    __pyx_t_4 = PyFloat_FromDouble(0.5); if (unlikely(!__pyx_t_4)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 269; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
    __Pyx_GOTREF(__pyx_t_4);
    __pyx_t_1 = PyFloat_FromDouble(1.); if (unlikely(!__pyx_t_1)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 269; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
    __Pyx_GOTREF(__pyx_t_1);
    __pyx_t_3 = __Pyx_GetName(__pyx_m, __pyx_n_s__np); if (unlikely(!__pyx_t_3)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 269; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
    __Pyx_GOTREF(__pyx_t_3);
    __pyx_t_2 = PyObject_GetAttr(__pyx_t_3, __pyx_n_s__cos); if (unlikely(!__pyx_t_2)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 269; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
    __Pyx_GOTREF(__pyx_t_2);
    __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
    __pyx_t_3 = PyFloat_FromDouble((__pyx_v_factor * __pyx_v_t)); if (unlikely(!__pyx_t_3)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 269; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
    __Pyx_GOTREF(__pyx_t_3);
    __pyx_t_9 = PyTuple_New(1); if (unlikely(!__pyx_t_9)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 269; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
    __Pyx_GOTREF(((PyObject *)__pyx_t_9));
    PyTuple_SET_ITEM(__pyx_t_9, 0, __pyx_t_3);
    __Pyx_GIVEREF(__pyx_t_3);
    __pyx_t_3 = 0;
    __pyx_t_3 = PyObject_Call(__pyx_t_2, ((PyObject *)__pyx_t_9), NULL); if (unlikely(!__pyx_t_3)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 269; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
    __Pyx_GOTREF(__pyx_t_3);
    __Pyx_DECREF(__pyx_t_2); __pyx_t_2 = 0;
    __Pyx_DECREF(((PyObject *)__pyx_t_9)); __pyx_t_9 = 0;
    __pyx_t_9 = PyFloat_FromDouble(exp(((-__pyx_v_op) * __pyx_v_t))); if (unlikely(!__pyx_t_9)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 269; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
    __Pyx_GOTREF(__pyx_t_9);
    __pyx_t_2 = PyNumber_Multiply(__pyx_t_3, __pyx_t_9); if (unlikely(!__pyx_t_2)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 269; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
    __Pyx_GOTREF(__pyx_t_2);
    __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
    __Pyx_DECREF(__pyx_t_9); __pyx_t_9 = 0;
    __pyx_t_9 = PyNumber_Subtract(__pyx_t_1, __pyx_t_2); if (unlikely(!__pyx_t_9)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 269; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
    __Pyx_GOTREF(__pyx_t_9);
    __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;
    __Pyx_DECREF(__pyx_t_2); __pyx_t_2 = 0;
    __pyx_t_2 = PyNumber_Multiply(__pyx_t_4, __pyx_t_9); if (unlikely(!__pyx_t_2)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 269; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
    __Pyx_GOTREF(__pyx_t_2);
    __Pyx_DECREF(__pyx_t_4); __pyx_t_4 = 0;
    __Pyx_DECREF(__pyx_t_9); __pyx_t_9 = 0;
    __pyx_t_10 = __pyx_PyFloat_AsDouble(__pyx_t_2); if (unlikely((__pyx_t_10 == (npy_double)-1) && PyErr_Occurred())) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 269; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
    __Pyx_DECREF(__pyx_t_2); __pyx_t_2 = 0;
    __pyx_t_11 = __pyx_v_t;
    if (__pyx_t_11 < 0) __pyx_t_11 += __pyx_bshape_0_out;
    *__Pyx_BufPtrCContig1d(__pyx_t_5numpy_float_t *, __pyx_bstruct_out.buf, __pyx_t_11, __pyx_bstride_0_out) = __pyx_t_10;
  }

  /* "atom_.pyx":271
 *         out[t] = 0.5 * (1. - np.cos(factor * t) * exp(-op * t))
 * 
 *     p = out[rise_n-1]             # <<<<<<<<<<<<<<
 *     for t in range(rise_n, N):
 *         out[t-1] = exp(-op*t)
 */
  __pyx_t_12 = (__pyx_v_rise_n - 1);
  if (__pyx_t_12 < 0) __pyx_t_12 += __pyx_bshape_0_out;
  __pyx_v_p = (*__Pyx_BufPtrCContig1d(__pyx_t_5numpy_float_t *, __pyx_bstruct_out.buf, __pyx_t_12, __pyx_bstride_0_out));

  /* "atom_.pyx":272
 * 
 *     p = out[rise_n-1]
 *     for t in range(rise_n, N):             # <<<<<<<<<<<<<<
 *         out[t-1] = exp(-op*t)
 * 
 */
  __pyx_t_7 = __pyx_v_N;
  for (__pyx_t_8 = __pyx_v_rise_n; __pyx_t_8 < __pyx_t_7; __pyx_t_8+=1) {
    __pyx_v_t = __pyx_t_8;

    /* "atom_.pyx":273
 *     p = out[rise_n-1]
 *     for t in range(rise_n, N):
 *         out[t-1] = exp(-op*t)             # <<<<<<<<<<<<<<
 * 
 *     a = max(abs(out[rise_n-1:N-1]))
 */
    __pyx_t_13 = (__pyx_v_t - 1);
    if (__pyx_t_13 < 0) __pyx_t_13 += __pyx_bshape_0_out;
    *__Pyx_BufPtrCContig1d(__pyx_t_5numpy_float_t *, __pyx_bstruct_out.buf, __pyx_t_13, __pyx_bstride_0_out) = exp(((-__pyx_v_op) * __pyx_v_t));
  }

  /* "atom_.pyx":275
 *         out[t-1] = exp(-op*t)
 * 
 *     a = max(abs(out[rise_n-1:N-1]))             # <<<<<<<<<<<<<<
 *     for t in range(rise_n-1, N-1):
 *         out[t] = out[t]/a  * p
 */
  __pyx_t_2 = __Pyx_PySequence_GetSlice(((PyObject *)__pyx_v_out), (__pyx_v_rise_n - 1), (__pyx_v_N - 1)); if (unlikely(!__pyx_t_2)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 275; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(__pyx_t_2);
  __pyx_t_9 = PyNumber_Absolute(__pyx_t_2); if (unlikely(!__pyx_t_9)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 275; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(__pyx_t_9);
  __Pyx_DECREF(__pyx_t_2); __pyx_t_2 = 0;
  __pyx_t_2 = PyTuple_New(1); if (unlikely(!__pyx_t_2)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 275; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(((PyObject *)__pyx_t_2));
  PyTuple_SET_ITEM(__pyx_t_2, 0, __pyx_t_9);
  __Pyx_GIVEREF(__pyx_t_9);
  __pyx_t_9 = 0;
  __pyx_t_9 = PyObject_Call(__pyx_builtin_max, ((PyObject *)__pyx_t_2), NULL); if (unlikely(!__pyx_t_9)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 275; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(__pyx_t_9);
  __Pyx_DECREF(((PyObject *)__pyx_t_2)); __pyx_t_2 = 0;
  __pyx_t_14 = __pyx_PyFloat_AsDouble(__pyx_t_9); if (unlikely((__pyx_t_14 == (float)-1) && PyErr_Occurred())) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 275; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_DECREF(__pyx_t_9); __pyx_t_9 = 0;
  __pyx_v_a = __pyx_t_14;

  /* "atom_.pyx":276
 * 
 *     a = max(abs(out[rise_n-1:N-1]))
 *     for t in range(rise_n-1, N-1):             # <<<<<<<<<<<<<<
 *         out[t] = out[t]/a  * p
 * 
 */
  __pyx_t_15 = (__pyx_v_N - 1);
  for (__pyx_t_7 = (__pyx_v_rise_n - 1); __pyx_t_7 < __pyx_t_15; __pyx_t_7+=1) {
    __pyx_v_t = __pyx_t_7;

    /* "atom_.pyx":277
 *     a = max(abs(out[rise_n-1:N-1]))
 *     for t in range(rise_n-1, N-1):
 *         out[t] = out[t]/a  * p             # <<<<<<<<<<<<<<
 * 
 *     for i in range(N):
 */
    __pyx_t_8 = __pyx_v_t;
    if (__pyx_t_8 < 0) __pyx_t_8 += __pyx_bshape_0_out;
    __pyx_t_10 = (*__Pyx_BufPtrCContig1d(__pyx_t_5numpy_float_t *, __pyx_bstruct_out.buf, __pyx_t_8, __pyx_bstride_0_out));
    if (unlikely(__pyx_v_a == 0)) {
      PyErr_Format(PyExc_ZeroDivisionError, "float division");
      {__pyx_filename = __pyx_f[0]; __pyx_lineno = 277; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
    }
    __pyx_t_16 = __pyx_v_t;
    if (__pyx_t_16 < 0) __pyx_t_16 += __pyx_bshape_0_out;
    *__Pyx_BufPtrCContig1d(__pyx_t_5numpy_float_t *, __pyx_bstruct_out.buf, __pyx_t_16, __pyx_bstride_0_out) = ((__pyx_t_10 / __pyx_v_a) * __pyx_v_p);
  }

  /* "atom_.pyx":279
 *         out[t] = out[t]/a  * p
 * 
 *     for i in range(N):             # <<<<<<<<<<<<<<
 *         out[i] = out[i] * realSinusoidFM(i, omega, chirp, phi, omega_m, phi_m, depth)
 * 
 */
  __pyx_t_7 = __pyx_v_N;
  for (__pyx_t_17 = 0; __pyx_t_17 < __pyx_t_7; __pyx_t_17+=1) {
    __pyx_v_i = __pyx_t_17;

    /* "atom_.pyx":280
 * 
 *     for i in range(N):
 *         out[i] = out[i] * realSinusoidFM(i, omega, chirp, phi, omega_m, phi_m, depth)             # <<<<<<<<<<<<<<
 * 
 *     return out
 */
    __pyx_t_18 = __pyx_v_i;
    if (__pyx_t_18 < 0) __pyx_t_18 += __pyx_bshape_0_out;
    __pyx_t_9 = PyFloat_FromDouble(__pyx_v_depth); if (unlikely(!__pyx_t_9)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 280; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
    __Pyx_GOTREF(__pyx_t_9);
    __pyx_t_19 = __pyx_v_i;
    if (__pyx_t_19 < 0) __pyx_t_19 += __pyx_bshape_0_out;
    *__Pyx_BufPtrCContig1d(__pyx_t_5numpy_float_t *, __pyx_bstruct_out.buf, __pyx_t_19, __pyx_bstride_0_out) = ((*__Pyx_BufPtrCContig1d(__pyx_t_5numpy_float_t *, __pyx_bstruct_out.buf, __pyx_t_18, __pyx_bstride_0_out)) * __pyx_f_5atom__realSinusoidFM(__pyx_v_i, __pyx_v_omega, __pyx_v_chirp, __pyx_v_phi, __pyx_v_omega_m, __pyx_v_phi_m, __pyx_t_9));
    __Pyx_DECREF(__pyx_t_9); __pyx_t_9 = 0;
  }

  /* "atom_.pyx":282
 *         out[i] = out[i] * realSinusoidFM(i, omega, chirp, phi, omega_m, phi_m, depth)
 * 
 *     return out             # <<<<<<<<<<<<<<
 * 
 * 
 */
  __Pyx_XDECREF(((PyObject *)__pyx_r));
  __Pyx_INCREF(((PyObject *)__pyx_v_out));
  __pyx_r = ((PyArrayObject *)__pyx_v_out);
  goto __pyx_L0;

  __pyx_r = ((PyArrayObject *)Py_None); __Pyx_INCREF(Py_None);
  goto __pyx_L0;
  __pyx_L1_error:;
  __Pyx_XDECREF(__pyx_t_1);
  __Pyx_XDECREF(__pyx_t_2);
  __Pyx_XDECREF(__pyx_t_3);
  __Pyx_XDECREF(__pyx_t_4);
  __Pyx_XDECREF(__pyx_t_9);
  { PyObject *__pyx_type, *__pyx_value, *__pyx_tb;
    __Pyx_ErrFetch(&__pyx_type, &__pyx_value, &__pyx_tb);
    __Pyx_SafeReleaseBuffer(&__pyx_bstruct_out);
  __Pyx_ErrRestore(__pyx_type, __pyx_value, __pyx_tb);}
  __Pyx_AddTraceback("atom_.fofFM_", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __pyx_r = 0;
  goto __pyx_L2;
  __pyx_L0:;
  __Pyx_SafeReleaseBuffer(&__pyx_bstruct_out);
  __pyx_L2:;
  __Pyx_XDECREF((PyObject *)__pyx_v_out);
  __Pyx_XGIVEREF((PyObject *)__pyx_r);
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

/* "atom_.pyx":260
 * @cython.profile(False)
 * @cython.boundscheck(False)
 * cpdef np.ndarray[np.float_t, ndim=1, mode="c"] fofFM_(float phi, int N, float omega, float chirp, int rise_n, int decay_n, float omega_m=0., float phi_m=0., float depth=0.):             # <<<<<<<<<<<<<<
 * 
 *     cdef np.ndarray[np.float_t, ndim=1, mode='c'] out = np.zeros(N, dtype=float)
 */

static PyObject *__pyx_pf_5atom__12fofFM_(PyObject *__pyx_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
static PyObject *__pyx_pf_5atom__12fofFM_(PyObject *__pyx_self, PyObject *__pyx_args, PyObject *__pyx_kwds) {
  float __pyx_v_phi;
  int __pyx_v_N;
  float __pyx_v_omega;
  float __pyx_v_chirp;
  int __pyx_v_rise_n;
  int __pyx_v_decay_n;
  float __pyx_v_omega_m;
  float __pyx_v_phi_m;
  float __pyx_v_depth;
  PyObject *__pyx_r = NULL;
  __Pyx_RefNannyDeclarations
  PyObject *__pyx_t_1 = NULL;
  struct __pyx_opt_args_5atom__fofFM_ __pyx_t_2;
  int __pyx_lineno = 0;
  const char *__pyx_filename = NULL;
  int __pyx_clineno = 0;
  static PyObject **__pyx_pyargnames[] = {&__pyx_n_s__phi,&__pyx_n_s__N,&__pyx_n_s__omega,&__pyx_n_s__chirp,&__pyx_n_s__rise_n,&__pyx_n_s__decay_n,&__pyx_n_s__omega_m,&__pyx_n_s__phi_m,&__pyx_n_s__depth,0};
  __Pyx_RefNannySetupContext("fofFM_");
  __pyx_self = __pyx_self;
  {
    PyObject* values[9] = {0,0,0,0,0,0,0,0,0};
    if (unlikely(__pyx_kwds)) {
      Py_ssize_t kw_args;
      switch (PyTuple_GET_SIZE(__pyx_args)) {
        case  9: values[8] = PyTuple_GET_ITEM(__pyx_args, 8);
        case  8: values[7] = PyTuple_GET_ITEM(__pyx_args, 7);
        case  7: values[6] = PyTuple_GET_ITEM(__pyx_args, 6);
        case  6: values[5] = PyTuple_GET_ITEM(__pyx_args, 5);
        case  5: values[4] = PyTuple_GET_ITEM(__pyx_args, 4);
        case  4: values[3] = PyTuple_GET_ITEM(__pyx_args, 3);
        case  3: values[2] = PyTuple_GET_ITEM(__pyx_args, 2);
        case  2: values[1] = PyTuple_GET_ITEM(__pyx_args, 1);
        case  1: values[0] = PyTuple_GET_ITEM(__pyx_args, 0);
        case  0: break;
        default: goto __pyx_L5_argtuple_error;
      }
      kw_args = PyDict_Size(__pyx_kwds);
      switch (PyTuple_GET_SIZE(__pyx_args)) {
        case  0:
        values[0] = PyDict_GetItem(__pyx_kwds, __pyx_n_s__phi);
        if (likely(values[0])) kw_args--;
        else goto __pyx_L5_argtuple_error;
        case  1:
        values[1] = PyDict_GetItem(__pyx_kwds, __pyx_n_s__N);
        if (likely(values[1])) kw_args--;
        else {
          __Pyx_RaiseArgtupleInvalid("fofFM_", 0, 6, 9, 1); {__pyx_filename = __pyx_f[0]; __pyx_lineno = 260; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
        }
        case  2:
        values[2] = PyDict_GetItem(__pyx_kwds, __pyx_n_s__omega);
        if (likely(values[2])) kw_args--;
        else {
          __Pyx_RaiseArgtupleInvalid("fofFM_", 0, 6, 9, 2); {__pyx_filename = __pyx_f[0]; __pyx_lineno = 260; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
        }
        case  3:
        values[3] = PyDict_GetItem(__pyx_kwds, __pyx_n_s__chirp);
        if (likely(values[3])) kw_args--;
        else {
          __Pyx_RaiseArgtupleInvalid("fofFM_", 0, 6, 9, 3); {__pyx_filename = __pyx_f[0]; __pyx_lineno = 260; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
        }
        case  4:
        values[4] = PyDict_GetItem(__pyx_kwds, __pyx_n_s__rise_n);
        if (likely(values[4])) kw_args--;
        else {
          __Pyx_RaiseArgtupleInvalid("fofFM_", 0, 6, 9, 4); {__pyx_filename = __pyx_f[0]; __pyx_lineno = 260; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
        }
        case  5:
        values[5] = PyDict_GetItem(__pyx_kwds, __pyx_n_s__decay_n);
        if (likely(values[5])) kw_args--;
        else {
          __Pyx_RaiseArgtupleInvalid("fofFM_", 0, 6, 9, 5); {__pyx_filename = __pyx_f[0]; __pyx_lineno = 260; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
        }
        case  6:
        if (kw_args > 0) {
          PyObject* value = PyDict_GetItem(__pyx_kwds, __pyx_n_s__omega_m);
          if (value) { values[6] = value; kw_args--; }
        }
        case  7:
        if (kw_args > 0) {
          PyObject* value = PyDict_GetItem(__pyx_kwds, __pyx_n_s__phi_m);
          if (value) { values[7] = value; kw_args--; }
        }
        case  8:
        if (kw_args > 0) {
          PyObject* value = PyDict_GetItem(__pyx_kwds, __pyx_n_s__depth);
          if (value) { values[8] = value; kw_args--; }
        }
      }
      if (unlikely(kw_args > 0)) {
        if (unlikely(__Pyx_ParseOptionalKeywords(__pyx_kwds, __pyx_pyargnames, 0, values, PyTuple_GET_SIZE(__pyx_args), "fofFM_") < 0)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 260; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
      }
    } else {
      switch (PyTuple_GET_SIZE(__pyx_args)) {
        case  9: values[8] = PyTuple_GET_ITEM(__pyx_args, 8);
        case  8: values[7] = PyTuple_GET_ITEM(__pyx_args, 7);
        case  7: values[6] = PyTuple_GET_ITEM(__pyx_args, 6);
        case  6: values[5] = PyTuple_GET_ITEM(__pyx_args, 5);
        values[4] = PyTuple_GET_ITEM(__pyx_args, 4);
        values[3] = PyTuple_GET_ITEM(__pyx_args, 3);
        values[2] = PyTuple_GET_ITEM(__pyx_args, 2);
        values[1] = PyTuple_GET_ITEM(__pyx_args, 1);
        values[0] = PyTuple_GET_ITEM(__pyx_args, 0);
        break;
        default: goto __pyx_L5_argtuple_error;
      }
    }
    __pyx_v_phi = __pyx_PyFloat_AsDouble(values[0]); if (unlikely((__pyx_v_phi == (float)-1) && PyErr_Occurred())) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 260; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
    __pyx_v_N = __Pyx_PyInt_AsInt(values[1]); if (unlikely((__pyx_v_N == (int)-1) && PyErr_Occurred())) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 260; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
    __pyx_v_omega = __pyx_PyFloat_AsDouble(values[2]); if (unlikely((__pyx_v_omega == (float)-1) && PyErr_Occurred())) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 260; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
    __pyx_v_chirp = __pyx_PyFloat_AsDouble(values[3]); if (unlikely((__pyx_v_chirp == (float)-1) && PyErr_Occurred())) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 260; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
    __pyx_v_rise_n = __Pyx_PyInt_AsInt(values[4]); if (unlikely((__pyx_v_rise_n == (int)-1) && PyErr_Occurred())) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 260; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
    __pyx_v_decay_n = __Pyx_PyInt_AsInt(values[5]); if (unlikely((__pyx_v_decay_n == (int)-1) && PyErr_Occurred())) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 260; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
    if (values[6]) {
      __pyx_v_omega_m = __pyx_PyFloat_AsDouble(values[6]); if (unlikely((__pyx_v_omega_m == (float)-1) && PyErr_Occurred())) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 260; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
    } else {
      __pyx_v_omega_m = ((float)0.);
    }
    if (values[7]) {
      __pyx_v_phi_m = __pyx_PyFloat_AsDouble(values[7]); if (unlikely((__pyx_v_phi_m == (float)-1) && PyErr_Occurred())) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 260; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
    } else {
      __pyx_v_phi_m = ((float)0.);
    }
    if (values[8]) {
      __pyx_v_depth = __pyx_PyFloat_AsDouble(values[8]); if (unlikely((__pyx_v_depth == (float)-1) && PyErr_Occurred())) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 260; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
    } else {
      __pyx_v_depth = ((float)0.);
    }
  }
  goto __pyx_L4_argument_unpacking_done;
  __pyx_L5_argtuple_error:;
  __Pyx_RaiseArgtupleInvalid("fofFM_", 0, 6, 9, PyTuple_GET_SIZE(__pyx_args)); {__pyx_filename = __pyx_f[0]; __pyx_lineno = 260; __pyx_clineno = __LINE__; goto __pyx_L3_error;}
  __pyx_L3_error:;
  __Pyx_AddTraceback("atom_.fofFM_", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __Pyx_RefNannyFinishContext();
  return NULL;
  __pyx_L4_argument_unpacking_done:;
  __Pyx_XDECREF(__pyx_r);
  __pyx_t_2.__pyx_n = 3;
  __pyx_t_2.omega_m = __pyx_v_omega_m;
  __pyx_t_2.phi_m = __pyx_v_phi_m;
  __pyx_t_2.depth = __pyx_v_depth;
  __pyx_t_1 = ((PyObject *)__pyx_f_5atom__fofFM_(__pyx_v_phi, __pyx_v_N, __pyx_v_omega, __pyx_v_chirp, __pyx_v_rise_n, __pyx_v_decay_n, 0, &__pyx_t_2)); if (unlikely(!__pyx_t_1)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 260; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(__pyx_t_1);
  __pyx_r = __pyx_t_1;
  __pyx_t_1 = 0;
  goto __pyx_L0;

  __pyx_r = Py_None; __Pyx_INCREF(Py_None);
  goto __pyx_L0;
  __pyx_L1_error:;
  __Pyx_XDECREF(__pyx_t_1);
  __Pyx_AddTraceback("atom_.fofFM_", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __pyx_r = NULL;
  __pyx_L0:;
  __Pyx_XGIVEREF(__pyx_r);
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

/* "numpy.pxd":190
 *         # experimental exception made for __getbuffer__ and __releasebuffer__
 *         # -- the details of this may change.
 *         def __getbuffer__(ndarray self, Py_buffer* info, int flags):             # <<<<<<<<<<<<<<
 *             # This implementation of getbuffer is geared towards Cython
 *             # requirements, and does not yet fullfill the PEP.
 */

static CYTHON_UNUSED int __pyx_pf_5numpy_7ndarray___getbuffer__(PyObject *__pyx_v_self, Py_buffer *__pyx_v_info, int __pyx_v_flags); /*proto*/
static CYTHON_UNUSED int __pyx_pf_5numpy_7ndarray___getbuffer__(PyObject *__pyx_v_self, Py_buffer *__pyx_v_info, int __pyx_v_flags) {
  int __pyx_v_copy_shape;
  int __pyx_v_i;
  int __pyx_v_ndim;
  int __pyx_v_endian_detector;
  int __pyx_v_little_endian;
  int __pyx_v_t;
  char *__pyx_v_f;
  PyArray_Descr *__pyx_v_descr = 0;
  int __pyx_v_offset;
  int __pyx_v_hasfields;
  int __pyx_r;
  __Pyx_RefNannyDeclarations
  int __pyx_t_1;
  int __pyx_t_2;
  int __pyx_t_3;
  PyObject *__pyx_t_4 = NULL;
  int __pyx_t_5;
  int __pyx_t_6;
  int __pyx_t_7;
  PyObject *__pyx_t_8 = NULL;
  char *__pyx_t_9;
  int __pyx_lineno = 0;
  const char *__pyx_filename = NULL;
  int __pyx_clineno = 0;
  __Pyx_RefNannySetupContext("__getbuffer__");
  if (__pyx_v_info != NULL) {
    __pyx_v_info->obj = Py_None; __Pyx_INCREF(Py_None);
    __Pyx_GIVEREF(__pyx_v_info->obj);
  }

  /* "numpy.pxd":196
 *             # of flags
 * 
 *             if info == NULL: return             # <<<<<<<<<<<<<<
 * 
 *             cdef int copy_shape, i, ndim
 */
  __pyx_t_1 = (__pyx_v_info == NULL);
  if (__pyx_t_1) {
    __pyx_r = 0;
    goto __pyx_L0;
    goto __pyx_L5;
  }
  __pyx_L5:;

  /* "numpy.pxd":199
 * 
 *             cdef int copy_shape, i, ndim
 *             cdef int endian_detector = 1             # <<<<<<<<<<<<<<
 *             cdef bint little_endian = ((<char*>&endian_detector)[0] != 0)
 * 
 */
  __pyx_v_endian_detector = 1;

  /* "numpy.pxd":200
 *             cdef int copy_shape, i, ndim
 *             cdef int endian_detector = 1
 *             cdef bint little_endian = ((<char*>&endian_detector)[0] != 0)             # <<<<<<<<<<<<<<
 * 
 *             ndim = PyArray_NDIM(self)
 */
  __pyx_v_little_endian = ((((char *)(&__pyx_v_endian_detector))[0]) != 0);

  /* "numpy.pxd":202
 *             cdef bint little_endian = ((<char*>&endian_detector)[0] != 0)
 * 
 *             ndim = PyArray_NDIM(self)             # <<<<<<<<<<<<<<
 * 
 *             if sizeof(npy_intp) != sizeof(Py_ssize_t):
 */
  __pyx_v_ndim = PyArray_NDIM(((PyArrayObject *)__pyx_v_self));

  /* "numpy.pxd":204
 *             ndim = PyArray_NDIM(self)
 * 
 *             if sizeof(npy_intp) != sizeof(Py_ssize_t):             # <<<<<<<<<<<<<<
 *                 copy_shape = 1
 *             else:
 */
  __pyx_t_1 = ((sizeof(npy_intp)) != (sizeof(Py_ssize_t)));
  if (__pyx_t_1) {

    /* "numpy.pxd":205
 * 
 *             if sizeof(npy_intp) != sizeof(Py_ssize_t):
 *                 copy_shape = 1             # <<<<<<<<<<<<<<
 *             else:
 *                 copy_shape = 0
 */
    __pyx_v_copy_shape = 1;
    goto __pyx_L6;
  }
  /*else*/ {

    /* "numpy.pxd":207
 *                 copy_shape = 1
 *             else:
 *                 copy_shape = 0             # <<<<<<<<<<<<<<
 * 
 *             if ((flags & pybuf.PyBUF_C_CONTIGUOUS == pybuf.PyBUF_C_CONTIGUOUS)
 */
    __pyx_v_copy_shape = 0;
  }
  __pyx_L6:;

  /* "numpy.pxd":209
 *                 copy_shape = 0
 * 
 *             if ((flags & pybuf.PyBUF_C_CONTIGUOUS == pybuf.PyBUF_C_CONTIGUOUS)             # <<<<<<<<<<<<<<
 *                 and not PyArray_CHKFLAGS(self, NPY_C_CONTIGUOUS)):
 *                 raise ValueError(u"ndarray is not C contiguous")
 */
  __pyx_t_1 = ((__pyx_v_flags & PyBUF_C_CONTIGUOUS) == PyBUF_C_CONTIGUOUS);
  if (__pyx_t_1) {

    /* "numpy.pxd":210
 * 
 *             if ((flags & pybuf.PyBUF_C_CONTIGUOUS == pybuf.PyBUF_C_CONTIGUOUS)
 *                 and not PyArray_CHKFLAGS(self, NPY_C_CONTIGUOUS)):             # <<<<<<<<<<<<<<
 *                 raise ValueError(u"ndarray is not C contiguous")
 * 
 */
    __pyx_t_2 = (!PyArray_CHKFLAGS(((PyArrayObject *)__pyx_v_self), NPY_C_CONTIGUOUS));
    __pyx_t_3 = __pyx_t_2;
  } else {
    __pyx_t_3 = __pyx_t_1;
  }
  if (__pyx_t_3) {

    /* "numpy.pxd":211
 *             if ((flags & pybuf.PyBUF_C_CONTIGUOUS == pybuf.PyBUF_C_CONTIGUOUS)
 *                 and not PyArray_CHKFLAGS(self, NPY_C_CONTIGUOUS)):
 *                 raise ValueError(u"ndarray is not C contiguous")             # <<<<<<<<<<<<<<
 * 
 *             if ((flags & pybuf.PyBUF_F_CONTIGUOUS == pybuf.PyBUF_F_CONTIGUOUS)
 */
    __pyx_t_4 = PyObject_Call(__pyx_builtin_ValueError, ((PyObject *)__pyx_k_tuple_2), NULL); if (unlikely(!__pyx_t_4)) {__pyx_filename = __pyx_f[1]; __pyx_lineno = 211; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
    __Pyx_GOTREF(__pyx_t_4);
    __Pyx_Raise(__pyx_t_4, 0, 0, 0);
    __Pyx_DECREF(__pyx_t_4); __pyx_t_4 = 0;
    {__pyx_filename = __pyx_f[1]; __pyx_lineno = 211; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
    goto __pyx_L7;
  }
  __pyx_L7:;

  /* "numpy.pxd":213
 *                 raise ValueError(u"ndarray is not C contiguous")
 * 
 *             if ((flags & pybuf.PyBUF_F_CONTIGUOUS == pybuf.PyBUF_F_CONTIGUOUS)             # <<<<<<<<<<<<<<
 *                 and not PyArray_CHKFLAGS(self, NPY_F_CONTIGUOUS)):
 *                 raise ValueError(u"ndarray is not Fortran contiguous")
 */
  __pyx_t_3 = ((__pyx_v_flags & PyBUF_F_CONTIGUOUS) == PyBUF_F_CONTIGUOUS);
  if (__pyx_t_3) {

    /* "numpy.pxd":214
 * 
 *             if ((flags & pybuf.PyBUF_F_CONTIGUOUS == pybuf.PyBUF_F_CONTIGUOUS)
 *                 and not PyArray_CHKFLAGS(self, NPY_F_CONTIGUOUS)):             # <<<<<<<<<<<<<<
 *                 raise ValueError(u"ndarray is not Fortran contiguous")
 * 
 */
    __pyx_t_1 = (!PyArray_CHKFLAGS(((PyArrayObject *)__pyx_v_self), NPY_F_CONTIGUOUS));
    __pyx_t_2 = __pyx_t_1;
  } else {
    __pyx_t_2 = __pyx_t_3;
  }
  if (__pyx_t_2) {

    /* "numpy.pxd":215
 *             if ((flags & pybuf.PyBUF_F_CONTIGUOUS == pybuf.PyBUF_F_CONTIGUOUS)
 *                 and not PyArray_CHKFLAGS(self, NPY_F_CONTIGUOUS)):
 *                 raise ValueError(u"ndarray is not Fortran contiguous")             # <<<<<<<<<<<<<<
 * 
 *             info.buf = PyArray_DATA(self)
 */
    __pyx_t_4 = PyObject_Call(__pyx_builtin_ValueError, ((PyObject *)__pyx_k_tuple_4), NULL); if (unlikely(!__pyx_t_4)) {__pyx_filename = __pyx_f[1]; __pyx_lineno = 215; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
    __Pyx_GOTREF(__pyx_t_4);
    __Pyx_Raise(__pyx_t_4, 0, 0, 0);
    __Pyx_DECREF(__pyx_t_4); __pyx_t_4 = 0;
    {__pyx_filename = __pyx_f[1]; __pyx_lineno = 215; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
    goto __pyx_L8;
  }
  __pyx_L8:;

  /* "numpy.pxd":217
 *                 raise ValueError(u"ndarray is not Fortran contiguous")
 * 
 *             info.buf = PyArray_DATA(self)             # <<<<<<<<<<<<<<
 *             info.ndim = ndim
 *             if copy_shape:
 */
  __pyx_v_info->buf = PyArray_DATA(((PyArrayObject *)__pyx_v_self));

  /* "numpy.pxd":218
 * 
 *             info.buf = PyArray_DATA(self)
 *             info.ndim = ndim             # <<<<<<<<<<<<<<
 *             if copy_shape:
 *                 # Allocate new buffer for strides and shape info.
 */
  __pyx_v_info->ndim = __pyx_v_ndim;

  /* "numpy.pxd":219
 *             info.buf = PyArray_DATA(self)
 *             info.ndim = ndim
 *             if copy_shape:             # <<<<<<<<<<<<<<
 *                 # Allocate new buffer for strides and shape info.
 *                 # This is allocated as one block, strides first.
 */
  if (__pyx_v_copy_shape) {

    /* "numpy.pxd":222
 *                 # Allocate new buffer for strides and shape info.
 *                 # This is allocated as one block, strides first.
 *                 info.strides = <Py_ssize_t*>stdlib.malloc(sizeof(Py_ssize_t) * <size_t>ndim * 2)             # <<<<<<<<<<<<<<
 *                 info.shape = info.strides + ndim
 *                 for i in range(ndim):
 */
    __pyx_v_info->strides = ((Py_ssize_t *)malloc((((sizeof(Py_ssize_t)) * ((size_t)__pyx_v_ndim)) * 2)));

    /* "numpy.pxd":223
 *                 # This is allocated as one block, strides first.
 *                 info.strides = <Py_ssize_t*>stdlib.malloc(sizeof(Py_ssize_t) * <size_t>ndim * 2)
 *                 info.shape = info.strides + ndim             # <<<<<<<<<<<<<<
 *                 for i in range(ndim):
 *                     info.strides[i] = PyArray_STRIDES(self)[i]
 */
    __pyx_v_info->shape = (__pyx_v_info->strides + __pyx_v_ndim);

    /* "numpy.pxd":224
 *                 info.strides = <Py_ssize_t*>stdlib.malloc(sizeof(Py_ssize_t) * <size_t>ndim * 2)
 *                 info.shape = info.strides + ndim
 *                 for i in range(ndim):             # <<<<<<<<<<<<<<
 *                     info.strides[i] = PyArray_STRIDES(self)[i]
 *                     info.shape[i] = PyArray_DIMS(self)[i]
 */
    __pyx_t_5 = __pyx_v_ndim;
    for (__pyx_t_6 = 0; __pyx_t_6 < __pyx_t_5; __pyx_t_6+=1) {
      __pyx_v_i = __pyx_t_6;

      /* "numpy.pxd":225
 *                 info.shape = info.strides + ndim
 *                 for i in range(ndim):
 *                     info.strides[i] = PyArray_STRIDES(self)[i]             # <<<<<<<<<<<<<<
 *                     info.shape[i] = PyArray_DIMS(self)[i]
 *             else:
 */
      (__pyx_v_info->strides[__pyx_v_i]) = (PyArray_STRIDES(((PyArrayObject *)__pyx_v_self))[__pyx_v_i]);

      /* "numpy.pxd":226
 *                 for i in range(ndim):
 *                     info.strides[i] = PyArray_STRIDES(self)[i]
 *                     info.shape[i] = PyArray_DIMS(self)[i]             # <<<<<<<<<<<<<<
 *             else:
 *                 info.strides = <Py_ssize_t*>PyArray_STRIDES(self)
 */
      (__pyx_v_info->shape[__pyx_v_i]) = (PyArray_DIMS(((PyArrayObject *)__pyx_v_self))[__pyx_v_i]);
    }
    goto __pyx_L9;
  }
  /*else*/ {

    /* "numpy.pxd":228
 *                     info.shape[i] = PyArray_DIMS(self)[i]
 *             else:
 *                 info.strides = <Py_ssize_t*>PyArray_STRIDES(self)             # <<<<<<<<<<<<<<
 *                 info.shape = <Py_ssize_t*>PyArray_DIMS(self)
 *             info.suboffsets = NULL
 */
    __pyx_v_info->strides = ((Py_ssize_t *)PyArray_STRIDES(((PyArrayObject *)__pyx_v_self)));

    /* "numpy.pxd":229
 *             else:
 *                 info.strides = <Py_ssize_t*>PyArray_STRIDES(self)
 *                 info.shape = <Py_ssize_t*>PyArray_DIMS(self)             # <<<<<<<<<<<<<<
 *             info.suboffsets = NULL
 *             info.itemsize = PyArray_ITEMSIZE(self)
 */
    __pyx_v_info->shape = ((Py_ssize_t *)PyArray_DIMS(((PyArrayObject *)__pyx_v_self)));
  }
  __pyx_L9:;

  /* "numpy.pxd":230
 *                 info.strides = <Py_ssize_t*>PyArray_STRIDES(self)
 *                 info.shape = <Py_ssize_t*>PyArray_DIMS(self)
 *             info.suboffsets = NULL             # <<<<<<<<<<<<<<
 *             info.itemsize = PyArray_ITEMSIZE(self)
 *             info.readonly = not PyArray_ISWRITEABLE(self)
 */
  __pyx_v_info->suboffsets = NULL;

  /* "numpy.pxd":231
 *                 info.shape = <Py_ssize_t*>PyArray_DIMS(self)
 *             info.suboffsets = NULL
 *             info.itemsize = PyArray_ITEMSIZE(self)             # <<<<<<<<<<<<<<
 *             info.readonly = not PyArray_ISWRITEABLE(self)
 * 
 */
  __pyx_v_info->itemsize = PyArray_ITEMSIZE(((PyArrayObject *)__pyx_v_self));

  /* "numpy.pxd":232
 *             info.suboffsets = NULL
 *             info.itemsize = PyArray_ITEMSIZE(self)
 *             info.readonly = not PyArray_ISWRITEABLE(self)             # <<<<<<<<<<<<<<
 * 
 *             cdef int t
 */
  __pyx_v_info->readonly = (!PyArray_ISWRITEABLE(((PyArrayObject *)__pyx_v_self)));

  /* "numpy.pxd":235
 * 
 *             cdef int t
 *             cdef char* f = NULL             # <<<<<<<<<<<<<<
 *             cdef dtype descr = self.descr
 *             cdef list stack
 */
  __pyx_v_f = NULL;

  /* "numpy.pxd":236
 *             cdef int t
 *             cdef char* f = NULL
 *             cdef dtype descr = self.descr             # <<<<<<<<<<<<<<
 *             cdef list stack
 *             cdef int offset
 */
  __Pyx_INCREF(((PyObject *)((PyArrayObject *)__pyx_v_self)->descr));
  __pyx_v_descr = ((PyArrayObject *)__pyx_v_self)->descr;

  /* "numpy.pxd":240
 *             cdef int offset
 * 
 *             cdef bint hasfields = PyDataType_HASFIELDS(descr)             # <<<<<<<<<<<<<<
 * 
 *             if not hasfields and not copy_shape:
 */
  __pyx_v_hasfields = PyDataType_HASFIELDS(__pyx_v_descr);

  /* "numpy.pxd":242
 *             cdef bint hasfields = PyDataType_HASFIELDS(descr)
 * 
 *             if not hasfields and not copy_shape:             # <<<<<<<<<<<<<<
 *                 # do not call releasebuffer
 *                 info.obj = None
 */
  __pyx_t_2 = (!__pyx_v_hasfields);
  if (__pyx_t_2) {
    __pyx_t_3 = (!__pyx_v_copy_shape);
    __pyx_t_1 = __pyx_t_3;
  } else {
    __pyx_t_1 = __pyx_t_2;
  }
  if (__pyx_t_1) {

    /* "numpy.pxd":244
 *             if not hasfields and not copy_shape:
 *                 # do not call releasebuffer
 *                 info.obj = None             # <<<<<<<<<<<<<<
 *             else:
 *                 # need to call releasebuffer
 */
    __Pyx_INCREF(Py_None);
    __Pyx_GIVEREF(Py_None);
    __Pyx_GOTREF(__pyx_v_info->obj);
    __Pyx_DECREF(__pyx_v_info->obj);
    __pyx_v_info->obj = Py_None;
    goto __pyx_L12;
  }
  /*else*/ {

    /* "numpy.pxd":247
 *             else:
 *                 # need to call releasebuffer
 *                 info.obj = self             # <<<<<<<<<<<<<<
 * 
 *             if not hasfields:
 */
    __Pyx_INCREF(__pyx_v_self);
    __Pyx_GIVEREF(__pyx_v_self);
    __Pyx_GOTREF(__pyx_v_info->obj);
    __Pyx_DECREF(__pyx_v_info->obj);
    __pyx_v_info->obj = __pyx_v_self;
  }
  __pyx_L12:;

  /* "numpy.pxd":249
 *                 info.obj = self
 * 
 *             if not hasfields:             # <<<<<<<<<<<<<<
 *                 t = descr.type_num
 *                 if ((descr.byteorder == '>' and little_endian) or
 */
  __pyx_t_1 = (!__pyx_v_hasfields);
  if (__pyx_t_1) {

    /* "numpy.pxd":250
 * 
 *             if not hasfields:
 *                 t = descr.type_num             # <<<<<<<<<<<<<<
 *                 if ((descr.byteorder == '>' and little_endian) or
 *                     (descr.byteorder == '<' and not little_endian)):
 */
    __pyx_v_t = __pyx_v_descr->type_num;

    /* "numpy.pxd":251
 *             if not hasfields:
 *                 t = descr.type_num
 *                 if ((descr.byteorder == '>' and little_endian) or             # <<<<<<<<<<<<<<
 *                     (descr.byteorder == '<' and not little_endian)):
 *                     raise ValueError(u"Non-native byte order not supported")
 */
    __pyx_t_1 = (__pyx_v_descr->byteorder == '>');
    if (__pyx_t_1) {
      __pyx_t_2 = __pyx_v_little_endian;
    } else {
      __pyx_t_2 = __pyx_t_1;
    }
    if (!__pyx_t_2) {

      /* "numpy.pxd":252
 *                 t = descr.type_num
 *                 if ((descr.byteorder == '>' and little_endian) or
 *                     (descr.byteorder == '<' and not little_endian)):             # <<<<<<<<<<<<<<
 *                     raise ValueError(u"Non-native byte order not supported")
 *                 if   t == NPY_BYTE:        f = "b"
 */
      __pyx_t_1 = (__pyx_v_descr->byteorder == '<');
      if (__pyx_t_1) {
        __pyx_t_3 = (!__pyx_v_little_endian);
        __pyx_t_7 = __pyx_t_3;
      } else {
        __pyx_t_7 = __pyx_t_1;
      }
      __pyx_t_1 = __pyx_t_7;
    } else {
      __pyx_t_1 = __pyx_t_2;
    }
    if (__pyx_t_1) {

      /* "numpy.pxd":253
 *                 if ((descr.byteorder == '>' and little_endian) or
 *                     (descr.byteorder == '<' and not little_endian)):
 *                     raise ValueError(u"Non-native byte order not supported")             # <<<<<<<<<<<<<<
 *                 if   t == NPY_BYTE:        f = "b"
 *                 elif t == NPY_UBYTE:       f = "B"
 */
      __pyx_t_4 = PyObject_Call(__pyx_builtin_ValueError, ((PyObject *)__pyx_k_tuple_6), NULL); if (unlikely(!__pyx_t_4)) {__pyx_filename = __pyx_f[1]; __pyx_lineno = 253; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
      __Pyx_GOTREF(__pyx_t_4);
      __Pyx_Raise(__pyx_t_4, 0, 0, 0);
      __Pyx_DECREF(__pyx_t_4); __pyx_t_4 = 0;
      {__pyx_filename = __pyx_f[1]; __pyx_lineno = 253; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
      goto __pyx_L14;
    }
    __pyx_L14:;

    /* "numpy.pxd":254
 *                     (descr.byteorder == '<' and not little_endian)):
 *                     raise ValueError(u"Non-native byte order not supported")
 *                 if   t == NPY_BYTE:        f = "b"             # <<<<<<<<<<<<<<
 *                 elif t == NPY_UBYTE:       f = "B"
 *                 elif t == NPY_SHORT:       f = "h"
 */
    __pyx_t_1 = (__pyx_v_t == NPY_BYTE);
    if (__pyx_t_1) {
      __pyx_v_f = __pyx_k__b;
      goto __pyx_L15;
    }

    /* "numpy.pxd":255
 *                     raise ValueError(u"Non-native byte order not supported")
 *                 if   t == NPY_BYTE:        f = "b"
 *                 elif t == NPY_UBYTE:       f = "B"             # <<<<<<<<<<<<<<
 *                 elif t == NPY_SHORT:       f = "h"
 *                 elif t == NPY_USHORT:      f = "H"
 */
    __pyx_t_1 = (__pyx_v_t == NPY_UBYTE);
    if (__pyx_t_1) {
      __pyx_v_f = __pyx_k__B;
      goto __pyx_L15;
    }

    /* "numpy.pxd":256
 *                 if   t == NPY_BYTE:        f = "b"
 *                 elif t == NPY_UBYTE:       f = "B"
 *                 elif t == NPY_SHORT:       f = "h"             # <<<<<<<<<<<<<<
 *                 elif t == NPY_USHORT:      f = "H"
 *                 elif t == NPY_INT:         f = "i"
 */
    __pyx_t_1 = (__pyx_v_t == NPY_SHORT);
    if (__pyx_t_1) {
      __pyx_v_f = __pyx_k__h;
      goto __pyx_L15;
    }

    /* "numpy.pxd":257
 *                 elif t == NPY_UBYTE:       f = "B"
 *                 elif t == NPY_SHORT:       f = "h"
 *                 elif t == NPY_USHORT:      f = "H"             # <<<<<<<<<<<<<<
 *                 elif t == NPY_INT:         f = "i"
 *                 elif t == NPY_UINT:        f = "I"
 */
    __pyx_t_1 = (__pyx_v_t == NPY_USHORT);
    if (__pyx_t_1) {
      __pyx_v_f = __pyx_k__H;
      goto __pyx_L15;
    }

    /* "numpy.pxd":258
 *                 elif t == NPY_SHORT:       f = "h"
 *                 elif t == NPY_USHORT:      f = "H"
 *                 elif t == NPY_INT:         f = "i"             # <<<<<<<<<<<<<<
 *                 elif t == NPY_UINT:        f = "I"
 *                 elif t == NPY_LONG:        f = "l"
 */
    __pyx_t_1 = (__pyx_v_t == NPY_INT);
    if (__pyx_t_1) {
      __pyx_v_f = __pyx_k__i;
      goto __pyx_L15;
    }

    /* "numpy.pxd":259
 *                 elif t == NPY_USHORT:      f = "H"
 *                 elif t == NPY_INT:         f = "i"
 *                 elif t == NPY_UINT:        f = "I"             # <<<<<<<<<<<<<<
 *                 elif t == NPY_LONG:        f = "l"
 *                 elif t == NPY_ULONG:       f = "L"
 */
    __pyx_t_1 = (__pyx_v_t == NPY_UINT);
    if (__pyx_t_1) {
      __pyx_v_f = __pyx_k__I;
      goto __pyx_L15;
    }

    /* "numpy.pxd":260
 *                 elif t == NPY_INT:         f = "i"
 *                 elif t == NPY_UINT:        f = "I"
 *                 elif t == NPY_LONG:        f = "l"             # <<<<<<<<<<<<<<
 *                 elif t == NPY_ULONG:       f = "L"
 *                 elif t == NPY_LONGLONG:    f = "q"
 */
    __pyx_t_1 = (__pyx_v_t == NPY_LONG);
    if (__pyx_t_1) {
      __pyx_v_f = __pyx_k__l;
      goto __pyx_L15;
    }

    /* "numpy.pxd":261
 *                 elif t == NPY_UINT:        f = "I"
 *                 elif t == NPY_LONG:        f = "l"
 *                 elif t == NPY_ULONG:       f = "L"             # <<<<<<<<<<<<<<
 *                 elif t == NPY_LONGLONG:    f = "q"
 *                 elif t == NPY_ULONGLONG:   f = "Q"
 */
    __pyx_t_1 = (__pyx_v_t == NPY_ULONG);
    if (__pyx_t_1) {
      __pyx_v_f = __pyx_k__L;
      goto __pyx_L15;
    }

    /* "numpy.pxd":262
 *                 elif t == NPY_LONG:        f = "l"
 *                 elif t == NPY_ULONG:       f = "L"
 *                 elif t == NPY_LONGLONG:    f = "q"             # <<<<<<<<<<<<<<
 *                 elif t == NPY_ULONGLONG:   f = "Q"
 *                 elif t == NPY_FLOAT:       f = "f"
 */
    __pyx_t_1 = (__pyx_v_t == NPY_LONGLONG);
    if (__pyx_t_1) {
      __pyx_v_f = __pyx_k__q;
      goto __pyx_L15;
    }

    /* "numpy.pxd":263
 *                 elif t == NPY_ULONG:       f = "L"
 *                 elif t == NPY_LONGLONG:    f = "q"
 *                 elif t == NPY_ULONGLONG:   f = "Q"             # <<<<<<<<<<<<<<
 *                 elif t == NPY_FLOAT:       f = "f"
 *                 elif t == NPY_DOUBLE:      f = "d"
 */
    __pyx_t_1 = (__pyx_v_t == NPY_ULONGLONG);
    if (__pyx_t_1) {
      __pyx_v_f = __pyx_k__Q;
      goto __pyx_L15;
    }

    /* "numpy.pxd":264
 *                 elif t == NPY_LONGLONG:    f = "q"
 *                 elif t == NPY_ULONGLONG:   f = "Q"
 *                 elif t == NPY_FLOAT:       f = "f"             # <<<<<<<<<<<<<<
 *                 elif t == NPY_DOUBLE:      f = "d"
 *                 elif t == NPY_LONGDOUBLE:  f = "g"
 */
    __pyx_t_1 = (__pyx_v_t == NPY_FLOAT);
    if (__pyx_t_1) {
      __pyx_v_f = __pyx_k__f;
      goto __pyx_L15;
    }

    /* "numpy.pxd":265
 *                 elif t == NPY_ULONGLONG:   f = "Q"
 *                 elif t == NPY_FLOAT:       f = "f"
 *                 elif t == NPY_DOUBLE:      f = "d"             # <<<<<<<<<<<<<<
 *                 elif t == NPY_LONGDOUBLE:  f = "g"
 *                 elif t == NPY_CFLOAT:      f = "Zf"
 */
    __pyx_t_1 = (__pyx_v_t == NPY_DOUBLE);
    if (__pyx_t_1) {
      __pyx_v_f = __pyx_k__d;
      goto __pyx_L15;
    }

    /* "numpy.pxd":266
 *                 elif t == NPY_FLOAT:       f = "f"
 *                 elif t == NPY_DOUBLE:      f = "d"
 *                 elif t == NPY_LONGDOUBLE:  f = "g"             # <<<<<<<<<<<<<<
 *                 elif t == NPY_CFLOAT:      f = "Zf"
 *                 elif t == NPY_CDOUBLE:     f = "Zd"
 */
    __pyx_t_1 = (__pyx_v_t == NPY_LONGDOUBLE);
    if (__pyx_t_1) {
      __pyx_v_f = __pyx_k__g;
      goto __pyx_L15;
    }

    /* "numpy.pxd":267
 *                 elif t == NPY_DOUBLE:      f = "d"
 *                 elif t == NPY_LONGDOUBLE:  f = "g"
 *                 elif t == NPY_CFLOAT:      f = "Zf"             # <<<<<<<<<<<<<<
 *                 elif t == NPY_CDOUBLE:     f = "Zd"
 *                 elif t == NPY_CLONGDOUBLE: f = "Zg"
 */
    __pyx_t_1 = (__pyx_v_t == NPY_CFLOAT);
    if (__pyx_t_1) {
      __pyx_v_f = __pyx_k__Zf;
      goto __pyx_L15;
    }

    /* "numpy.pxd":268
 *                 elif t == NPY_LONGDOUBLE:  f = "g"
 *                 elif t == NPY_CFLOAT:      f = "Zf"
 *                 elif t == NPY_CDOUBLE:     f = "Zd"             # <<<<<<<<<<<<<<
 *                 elif t == NPY_CLONGDOUBLE: f = "Zg"
 *                 elif t == NPY_OBJECT:      f = "O"
 */
    __pyx_t_1 = (__pyx_v_t == NPY_CDOUBLE);
    if (__pyx_t_1) {
      __pyx_v_f = __pyx_k__Zd;
      goto __pyx_L15;
    }

    /* "numpy.pxd":269
 *                 elif t == NPY_CFLOAT:      f = "Zf"
 *                 elif t == NPY_CDOUBLE:     f = "Zd"
 *                 elif t == NPY_CLONGDOUBLE: f = "Zg"             # <<<<<<<<<<<<<<
 *                 elif t == NPY_OBJECT:      f = "O"
 *                 else:
 */
    __pyx_t_1 = (__pyx_v_t == NPY_CLONGDOUBLE);
    if (__pyx_t_1) {
      __pyx_v_f = __pyx_k__Zg;
      goto __pyx_L15;
    }

    /* "numpy.pxd":270
 *                 elif t == NPY_CDOUBLE:     f = "Zd"
 *                 elif t == NPY_CLONGDOUBLE: f = "Zg"
 *                 elif t == NPY_OBJECT:      f = "O"             # <<<<<<<<<<<<<<
 *                 else:
 *                     raise ValueError(u"unknown dtype code in numpy.pxd (%d)" % t)
 */
    __pyx_t_1 = (__pyx_v_t == NPY_OBJECT);
    if (__pyx_t_1) {
      __pyx_v_f = __pyx_k__O;
      goto __pyx_L15;
    }
    /*else*/ {

      /* "numpy.pxd":272
 *                 elif t == NPY_OBJECT:      f = "O"
 *                 else:
 *                     raise ValueError(u"unknown dtype code in numpy.pxd (%d)" % t)             # <<<<<<<<<<<<<<
 *                 info.format = f
 *                 return
 */
      __pyx_t_4 = PyInt_FromLong(__pyx_v_t); if (unlikely(!__pyx_t_4)) {__pyx_filename = __pyx_f[1]; __pyx_lineno = 272; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
      __Pyx_GOTREF(__pyx_t_4);
      __pyx_t_8 = PyNumber_Remainder(((PyObject *)__pyx_kp_u_7), __pyx_t_4); if (unlikely(!__pyx_t_8)) {__pyx_filename = __pyx_f[1]; __pyx_lineno = 272; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
      __Pyx_GOTREF(((PyObject *)__pyx_t_8));
      __Pyx_DECREF(__pyx_t_4); __pyx_t_4 = 0;
      __pyx_t_4 = PyTuple_New(1); if (unlikely(!__pyx_t_4)) {__pyx_filename = __pyx_f[1]; __pyx_lineno = 272; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
      __Pyx_GOTREF(((PyObject *)__pyx_t_4));
      PyTuple_SET_ITEM(__pyx_t_4, 0, ((PyObject *)__pyx_t_8));
      __Pyx_GIVEREF(((PyObject *)__pyx_t_8));
      __pyx_t_8 = 0;
      __pyx_t_8 = PyObject_Call(__pyx_builtin_ValueError, ((PyObject *)__pyx_t_4), NULL); if (unlikely(!__pyx_t_8)) {__pyx_filename = __pyx_f[1]; __pyx_lineno = 272; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
      __Pyx_GOTREF(__pyx_t_8);
      __Pyx_DECREF(((PyObject *)__pyx_t_4)); __pyx_t_4 = 0;
      __Pyx_Raise(__pyx_t_8, 0, 0, 0);
      __Pyx_DECREF(__pyx_t_8); __pyx_t_8 = 0;
      {__pyx_filename = __pyx_f[1]; __pyx_lineno = 272; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
    }
    __pyx_L15:;

    /* "numpy.pxd":273
 *                 else:
 *                     raise ValueError(u"unknown dtype code in numpy.pxd (%d)" % t)
 *                 info.format = f             # <<<<<<<<<<<<<<
 *                 return
 *             else:
 */
    __pyx_v_info->format = __pyx_v_f;

    /* "numpy.pxd":274
 *                     raise ValueError(u"unknown dtype code in numpy.pxd (%d)" % t)
 *                 info.format = f
 *                 return             # <<<<<<<<<<<<<<
 *             else:
 *                 info.format = <char*>stdlib.malloc(_buffer_format_string_len)
 */
    __pyx_r = 0;
    goto __pyx_L0;
    goto __pyx_L13;
  }
  /*else*/ {

    /* "numpy.pxd":276
 *                 return
 *             else:
 *                 info.format = <char*>stdlib.malloc(_buffer_format_string_len)             # <<<<<<<<<<<<<<
 *                 info.format[0] = '^' # Native data types, manual alignment
 *                 offset = 0
 */
    __pyx_v_info->format = ((char *)malloc(255));

    /* "numpy.pxd":277
 *             else:
 *                 info.format = <char*>stdlib.malloc(_buffer_format_string_len)
 *                 info.format[0] = '^' # Native data types, manual alignment             # <<<<<<<<<<<<<<
 *                 offset = 0
 *                 f = _util_dtypestring(descr, info.format + 1,
 */
    (__pyx_v_info->format[0]) = '^';

    /* "numpy.pxd":278
 *                 info.format = <char*>stdlib.malloc(_buffer_format_string_len)
 *                 info.format[0] = '^' # Native data types, manual alignment
 *                 offset = 0             # <<<<<<<<<<<<<<
 *                 f = _util_dtypestring(descr, info.format + 1,
 *                                       info.format + _buffer_format_string_len,
 */
    __pyx_v_offset = 0;

    /* "numpy.pxd":281
 *                 f = _util_dtypestring(descr, info.format + 1,
 *                                       info.format + _buffer_format_string_len,
 *                                       &offset)             # <<<<<<<<<<<<<<
 *                 f[0] = 0 # Terminate format string
 * 
 */
    __pyx_t_9 = __pyx_f_5numpy__util_dtypestring(__pyx_v_descr, (__pyx_v_info->format + 1), (__pyx_v_info->format + 255), (&__pyx_v_offset)); if (unlikely(__pyx_t_9 == NULL)) {__pyx_filename = __pyx_f[1]; __pyx_lineno = 279; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
    __pyx_v_f = __pyx_t_9;

    /* "numpy.pxd":282
 *                                       info.format + _buffer_format_string_len,
 *                                       &offset)
 *                 f[0] = 0 # Terminate format string             # <<<<<<<<<<<<<<
 * 
 *         def __releasebuffer__(ndarray self, Py_buffer* info):
 */
    (__pyx_v_f[0]) = 0;
  }
  __pyx_L13:;

  __pyx_r = 0;
  goto __pyx_L0;
  __pyx_L1_error:;
  __Pyx_XDECREF(__pyx_t_4);
  __Pyx_XDECREF(__pyx_t_8);
  __Pyx_AddTraceback("numpy.ndarray.__getbuffer__", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __pyx_r = -1;
  if (__pyx_v_info != NULL && __pyx_v_info->obj != NULL) {
    __Pyx_GOTREF(__pyx_v_info->obj);
    __Pyx_DECREF(__pyx_v_info->obj); __pyx_v_info->obj = NULL;
  }
  goto __pyx_L2;
  __pyx_L0:;
  if (__pyx_v_info != NULL && __pyx_v_info->obj == Py_None) {
    __Pyx_GOTREF(Py_None);
    __Pyx_DECREF(Py_None); __pyx_v_info->obj = NULL;
  }
  __pyx_L2:;
  __Pyx_XDECREF((PyObject *)__pyx_v_descr);
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

/* "numpy.pxd":284
 *                 f[0] = 0 # Terminate format string
 * 
 *         def __releasebuffer__(ndarray self, Py_buffer* info):             # <<<<<<<<<<<<<<
 *             if PyArray_HASFIELDS(self):
 *                 stdlib.free(info.format)
 */

static CYTHON_UNUSED void __pyx_pf_5numpy_7ndarray_1__releasebuffer__(PyObject *__pyx_v_self, Py_buffer *__pyx_v_info); /*proto*/
static CYTHON_UNUSED void __pyx_pf_5numpy_7ndarray_1__releasebuffer__(PyObject *__pyx_v_self, Py_buffer *__pyx_v_info) {
  __Pyx_RefNannyDeclarations
  int __pyx_t_1;
  __Pyx_RefNannySetupContext("__releasebuffer__");

  /* "numpy.pxd":285
 * 
 *         def __releasebuffer__(ndarray self, Py_buffer* info):
 *             if PyArray_HASFIELDS(self):             # <<<<<<<<<<<<<<
 *                 stdlib.free(info.format)
 *             if sizeof(npy_intp) != sizeof(Py_ssize_t):
 */
  __pyx_t_1 = PyArray_HASFIELDS(((PyArrayObject *)__pyx_v_self));
  if (__pyx_t_1) {

    /* "numpy.pxd":286
 *         def __releasebuffer__(ndarray self, Py_buffer* info):
 *             if PyArray_HASFIELDS(self):
 *                 stdlib.free(info.format)             # <<<<<<<<<<<<<<
 *             if sizeof(npy_intp) != sizeof(Py_ssize_t):
 *                 stdlib.free(info.strides)
 */
    free(__pyx_v_info->format);
    goto __pyx_L5;
  }
  __pyx_L5:;

  /* "numpy.pxd":287
 *             if PyArray_HASFIELDS(self):
 *                 stdlib.free(info.format)
 *             if sizeof(npy_intp) != sizeof(Py_ssize_t):             # <<<<<<<<<<<<<<
 *                 stdlib.free(info.strides)
 *                 # info.shape was stored after info.strides in the same block
 */
  __pyx_t_1 = ((sizeof(npy_intp)) != (sizeof(Py_ssize_t)));
  if (__pyx_t_1) {

    /* "numpy.pxd":288
 *                 stdlib.free(info.format)
 *             if sizeof(npy_intp) != sizeof(Py_ssize_t):
 *                 stdlib.free(info.strides)             # <<<<<<<<<<<<<<
 *                 # info.shape was stored after info.strides in the same block
 * 
 */
    free(__pyx_v_info->strides);
    goto __pyx_L6;
  }
  __pyx_L6:;

  __Pyx_RefNannyFinishContext();
}

/* "numpy.pxd":764
 * ctypedef npy_cdouble     complex_t
 * 
 * cdef inline object PyArray_MultiIterNew1(a):             # <<<<<<<<<<<<<<
 *     return PyArray_MultiIterNew(1, <void*>a)
 * 
 */

static CYTHON_INLINE PyObject *__pyx_f_5numpy_PyArray_MultiIterNew1(PyObject *__pyx_v_a) {
  PyObject *__pyx_r = NULL;
  __Pyx_RefNannyDeclarations
  PyObject *__pyx_t_1 = NULL;
  int __pyx_lineno = 0;
  const char *__pyx_filename = NULL;
  int __pyx_clineno = 0;
  __Pyx_RefNannySetupContext("PyArray_MultiIterNew1");

  /* "numpy.pxd":765
 * 
 * cdef inline object PyArray_MultiIterNew1(a):
 *     return PyArray_MultiIterNew(1, <void*>a)             # <<<<<<<<<<<<<<
 * 
 * cdef inline object PyArray_MultiIterNew2(a, b):
 */
  __Pyx_XDECREF(__pyx_r);
  __pyx_t_1 = PyArray_MultiIterNew(1, ((void *)__pyx_v_a)); if (unlikely(!__pyx_t_1)) {__pyx_filename = __pyx_f[1]; __pyx_lineno = 765; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(__pyx_t_1);
  __pyx_r = __pyx_t_1;
  __pyx_t_1 = 0;
  goto __pyx_L0;

  __pyx_r = Py_None; __Pyx_INCREF(Py_None);
  goto __pyx_L0;
  __pyx_L1_error:;
  __Pyx_XDECREF(__pyx_t_1);
  __Pyx_AddTraceback("numpy.PyArray_MultiIterNew1", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __pyx_r = 0;
  __pyx_L0:;
  __Pyx_XGIVEREF(__pyx_r);
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

/* "numpy.pxd":767
 *     return PyArray_MultiIterNew(1, <void*>a)
 * 
 * cdef inline object PyArray_MultiIterNew2(a, b):             # <<<<<<<<<<<<<<
 *     return PyArray_MultiIterNew(2, <void*>a, <void*>b)
 * 
 */

static CYTHON_INLINE PyObject *__pyx_f_5numpy_PyArray_MultiIterNew2(PyObject *__pyx_v_a, PyObject *__pyx_v_b) {
  PyObject *__pyx_r = NULL;
  __Pyx_RefNannyDeclarations
  PyObject *__pyx_t_1 = NULL;
  int __pyx_lineno = 0;
  const char *__pyx_filename = NULL;
  int __pyx_clineno = 0;
  __Pyx_RefNannySetupContext("PyArray_MultiIterNew2");

  /* "numpy.pxd":768
 * 
 * cdef inline object PyArray_MultiIterNew2(a, b):
 *     return PyArray_MultiIterNew(2, <void*>a, <void*>b)             # <<<<<<<<<<<<<<
 * 
 * cdef inline object PyArray_MultiIterNew3(a, b, c):
 */
  __Pyx_XDECREF(__pyx_r);
  __pyx_t_1 = PyArray_MultiIterNew(2, ((void *)__pyx_v_a), ((void *)__pyx_v_b)); if (unlikely(!__pyx_t_1)) {__pyx_filename = __pyx_f[1]; __pyx_lineno = 768; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(__pyx_t_1);
  __pyx_r = __pyx_t_1;
  __pyx_t_1 = 0;
  goto __pyx_L0;

  __pyx_r = Py_None; __Pyx_INCREF(Py_None);
  goto __pyx_L0;
  __pyx_L1_error:;
  __Pyx_XDECREF(__pyx_t_1);
  __Pyx_AddTraceback("numpy.PyArray_MultiIterNew2", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __pyx_r = 0;
  __pyx_L0:;
  __Pyx_XGIVEREF(__pyx_r);
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

/* "numpy.pxd":770
 *     return PyArray_MultiIterNew(2, <void*>a, <void*>b)
 * 
 * cdef inline object PyArray_MultiIterNew3(a, b, c):             # <<<<<<<<<<<<<<
 *     return PyArray_MultiIterNew(3, <void*>a, <void*>b, <void*> c)
 * 
 */

static CYTHON_INLINE PyObject *__pyx_f_5numpy_PyArray_MultiIterNew3(PyObject *__pyx_v_a, PyObject *__pyx_v_b, PyObject *__pyx_v_c) {
  PyObject *__pyx_r = NULL;
  __Pyx_RefNannyDeclarations
  PyObject *__pyx_t_1 = NULL;
  int __pyx_lineno = 0;
  const char *__pyx_filename = NULL;
  int __pyx_clineno = 0;
  __Pyx_RefNannySetupContext("PyArray_MultiIterNew3");

  /* "numpy.pxd":771
 * 
 * cdef inline object PyArray_MultiIterNew3(a, b, c):
 *     return PyArray_MultiIterNew(3, <void*>a, <void*>b, <void*> c)             # <<<<<<<<<<<<<<
 * 
 * cdef inline object PyArray_MultiIterNew4(a, b, c, d):
 */
  __Pyx_XDECREF(__pyx_r);
  __pyx_t_1 = PyArray_MultiIterNew(3, ((void *)__pyx_v_a), ((void *)__pyx_v_b), ((void *)__pyx_v_c)); if (unlikely(!__pyx_t_1)) {__pyx_filename = __pyx_f[1]; __pyx_lineno = 771; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(__pyx_t_1);
  __pyx_r = __pyx_t_1;
  __pyx_t_1 = 0;
  goto __pyx_L0;

  __pyx_r = Py_None; __Pyx_INCREF(Py_None);
  goto __pyx_L0;
  __pyx_L1_error:;
  __Pyx_XDECREF(__pyx_t_1);
  __Pyx_AddTraceback("numpy.PyArray_MultiIterNew3", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __pyx_r = 0;
  __pyx_L0:;
  __Pyx_XGIVEREF(__pyx_r);
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

/* "numpy.pxd":773
 *     return PyArray_MultiIterNew(3, <void*>a, <void*>b, <void*> c)
 * 
 * cdef inline object PyArray_MultiIterNew4(a, b, c, d):             # <<<<<<<<<<<<<<
 *     return PyArray_MultiIterNew(4, <void*>a, <void*>b, <void*>c, <void*> d)
 * 
 */

static CYTHON_INLINE PyObject *__pyx_f_5numpy_PyArray_MultiIterNew4(PyObject *__pyx_v_a, PyObject *__pyx_v_b, PyObject *__pyx_v_c, PyObject *__pyx_v_d) {
  PyObject *__pyx_r = NULL;
  __Pyx_RefNannyDeclarations
  PyObject *__pyx_t_1 = NULL;
  int __pyx_lineno = 0;
  const char *__pyx_filename = NULL;
  int __pyx_clineno = 0;
  __Pyx_RefNannySetupContext("PyArray_MultiIterNew4");

  /* "numpy.pxd":774
 * 
 * cdef inline object PyArray_MultiIterNew4(a, b, c, d):
 *     return PyArray_MultiIterNew(4, <void*>a, <void*>b, <void*>c, <void*> d)             # <<<<<<<<<<<<<<
 * 
 * cdef inline object PyArray_MultiIterNew5(a, b, c, d, e):
 */
  __Pyx_XDECREF(__pyx_r);
  __pyx_t_1 = PyArray_MultiIterNew(4, ((void *)__pyx_v_a), ((void *)__pyx_v_b), ((void *)__pyx_v_c), ((void *)__pyx_v_d)); if (unlikely(!__pyx_t_1)) {__pyx_filename = __pyx_f[1]; __pyx_lineno = 774; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(__pyx_t_1);
  __pyx_r = __pyx_t_1;
  __pyx_t_1 = 0;
  goto __pyx_L0;

  __pyx_r = Py_None; __Pyx_INCREF(Py_None);
  goto __pyx_L0;
  __pyx_L1_error:;
  __Pyx_XDECREF(__pyx_t_1);
  __Pyx_AddTraceback("numpy.PyArray_MultiIterNew4", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __pyx_r = 0;
  __pyx_L0:;
  __Pyx_XGIVEREF(__pyx_r);
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

/* "numpy.pxd":776
 *     return PyArray_MultiIterNew(4, <void*>a, <void*>b, <void*>c, <void*> d)
 * 
 * cdef inline object PyArray_MultiIterNew5(a, b, c, d, e):             # <<<<<<<<<<<<<<
 *     return PyArray_MultiIterNew(5, <void*>a, <void*>b, <void*>c, <void*> d, <void*> e)
 * 
 */

static CYTHON_INLINE PyObject *__pyx_f_5numpy_PyArray_MultiIterNew5(PyObject *__pyx_v_a, PyObject *__pyx_v_b, PyObject *__pyx_v_c, PyObject *__pyx_v_d, PyObject *__pyx_v_e) {
  PyObject *__pyx_r = NULL;
  __Pyx_RefNannyDeclarations
  PyObject *__pyx_t_1 = NULL;
  int __pyx_lineno = 0;
  const char *__pyx_filename = NULL;
  int __pyx_clineno = 0;
  __Pyx_RefNannySetupContext("PyArray_MultiIterNew5");

  /* "numpy.pxd":777
 * 
 * cdef inline object PyArray_MultiIterNew5(a, b, c, d, e):
 *     return PyArray_MultiIterNew(5, <void*>a, <void*>b, <void*>c, <void*> d, <void*> e)             # <<<<<<<<<<<<<<
 * 
 * cdef inline char* _util_dtypestring(dtype descr, char* f, char* end, int* offset) except NULL:
 */
  __Pyx_XDECREF(__pyx_r);
  __pyx_t_1 = PyArray_MultiIterNew(5, ((void *)__pyx_v_a), ((void *)__pyx_v_b), ((void *)__pyx_v_c), ((void *)__pyx_v_d), ((void *)__pyx_v_e)); if (unlikely(!__pyx_t_1)) {__pyx_filename = __pyx_f[1]; __pyx_lineno = 777; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(__pyx_t_1);
  __pyx_r = __pyx_t_1;
  __pyx_t_1 = 0;
  goto __pyx_L0;

  __pyx_r = Py_None; __Pyx_INCREF(Py_None);
  goto __pyx_L0;
  __pyx_L1_error:;
  __Pyx_XDECREF(__pyx_t_1);
  __Pyx_AddTraceback("numpy.PyArray_MultiIterNew5", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __pyx_r = 0;
  __pyx_L0:;
  __Pyx_XGIVEREF(__pyx_r);
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

/* "numpy.pxd":779
 *     return PyArray_MultiIterNew(5, <void*>a, <void*>b, <void*>c, <void*> d, <void*> e)
 * 
 * cdef inline char* _util_dtypestring(dtype descr, char* f, char* end, int* offset) except NULL:             # <<<<<<<<<<<<<<
 *     # Recursive utility function used in __getbuffer__ to get format
 *     # string. The new location in the format string is returned.
 */

static CYTHON_INLINE char *__pyx_f_5numpy__util_dtypestring(PyArray_Descr *__pyx_v_descr, char *__pyx_v_f, char *__pyx_v_end, int *__pyx_v_offset) {
  PyArray_Descr *__pyx_v_child = 0;
  int __pyx_v_endian_detector;
  int __pyx_v_little_endian;
  PyObject *__pyx_v_fields = 0;
  PyObject *__pyx_v_childname = NULL;
  PyObject *__pyx_v_new_offset = NULL;
  PyObject *__pyx_v_t = NULL;
  char *__pyx_r;
  __Pyx_RefNannyDeclarations
  PyObject *__pyx_t_1 = NULL;
  Py_ssize_t __pyx_t_2;
  PyObject *__pyx_t_3 = NULL;
  PyObject *__pyx_t_4 = NULL;
  PyObject *__pyx_t_5 = NULL;
  int __pyx_t_6;
  int __pyx_t_7;
  int __pyx_t_8;
  int __pyx_t_9;
  long __pyx_t_10;
  char *__pyx_t_11;
  int __pyx_lineno = 0;
  const char *__pyx_filename = NULL;
  int __pyx_clineno = 0;
  __Pyx_RefNannySetupContext("_util_dtypestring");

  /* "numpy.pxd":786
 *     cdef int delta_offset
 *     cdef tuple i
 *     cdef int endian_detector = 1             # <<<<<<<<<<<<<<
 *     cdef bint little_endian = ((<char*>&endian_detector)[0] != 0)
 *     cdef tuple fields
 */
  __pyx_v_endian_detector = 1;

  /* "numpy.pxd":787
 *     cdef tuple i
 *     cdef int endian_detector = 1
 *     cdef bint little_endian = ((<char*>&endian_detector)[0] != 0)             # <<<<<<<<<<<<<<
 *     cdef tuple fields
 * 
 */
  __pyx_v_little_endian = ((((char *)(&__pyx_v_endian_detector))[0]) != 0);

  /* "numpy.pxd":790
 *     cdef tuple fields
 * 
 *     for childname in descr.names:             # <<<<<<<<<<<<<<
 *         fields = descr.fields[childname]
 *         child, new_offset = fields
 */
  if (unlikely(((PyObject *)__pyx_v_descr->names) == Py_None)) {
    PyErr_SetString(PyExc_TypeError, "'NoneType' object is not iterable"); {__pyx_filename = __pyx_f[1]; __pyx_lineno = 790; __pyx_clineno = __LINE__; goto __pyx_L1_error;} 
  }
  __pyx_t_1 = ((PyObject *)__pyx_v_descr->names); __Pyx_INCREF(__pyx_t_1); __pyx_t_2 = 0;
  for (;;) {
    if (__pyx_t_2 >= PyTuple_GET_SIZE(__pyx_t_1)) break;
    __pyx_t_3 = PyTuple_GET_ITEM(__pyx_t_1, __pyx_t_2); __Pyx_INCREF(__pyx_t_3); __pyx_t_2++;
    __Pyx_XDECREF(__pyx_v_childname);
    __pyx_v_childname = __pyx_t_3;
    __pyx_t_3 = 0;

    /* "numpy.pxd":791
 * 
 *     for childname in descr.names:
 *         fields = descr.fields[childname]             # <<<<<<<<<<<<<<
 *         child, new_offset = fields
 * 
 */
    __pyx_t_3 = PyObject_GetItem(__pyx_v_descr->fields, __pyx_v_childname); if (!__pyx_t_3) {__pyx_filename = __pyx_f[1]; __pyx_lineno = 791; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
    __Pyx_GOTREF(__pyx_t_3);
    if (!(likely(PyTuple_CheckExact(__pyx_t_3))||((__pyx_t_3) == Py_None)||(PyErr_Format(PyExc_TypeError, "Expected tuple, got %.200s", Py_TYPE(__pyx_t_3)->tp_name), 0))) {__pyx_filename = __pyx_f[1]; __pyx_lineno = 791; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
    __Pyx_XDECREF(((PyObject *)__pyx_v_fields));
    __pyx_v_fields = ((PyObject*)__pyx_t_3);
    __pyx_t_3 = 0;

    /* "numpy.pxd":792
 *     for childname in descr.names:
 *         fields = descr.fields[childname]
 *         child, new_offset = fields             # <<<<<<<<<<<<<<
 * 
 *         if (end - f) - (new_offset - offset[0]) < 15:
 */
    if (likely(PyTuple_CheckExact(((PyObject *)__pyx_v_fields)))) {
      PyObject* sequence = ((PyObject *)__pyx_v_fields);
      if (unlikely(PyTuple_GET_SIZE(sequence) != 2)) {
        if (PyTuple_GET_SIZE(sequence) > 2) __Pyx_RaiseTooManyValuesError(2);
        else __Pyx_RaiseNeedMoreValuesError(PyTuple_GET_SIZE(sequence));
        {__pyx_filename = __pyx_f[1]; __pyx_lineno = 792; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
      }
      __pyx_t_3 = PyTuple_GET_ITEM(sequence, 0); 
      __pyx_t_4 = PyTuple_GET_ITEM(sequence, 1); 
      __Pyx_INCREF(__pyx_t_3);
      __Pyx_INCREF(__pyx_t_4);
    } else {
      __Pyx_UnpackTupleError(((PyObject *)__pyx_v_fields), 2);
      {__pyx_filename = __pyx_f[1]; __pyx_lineno = 792; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
    }
    if (!(likely(((__pyx_t_3) == Py_None) || likely(__Pyx_TypeTest(__pyx_t_3, __pyx_ptype_5numpy_dtype))))) {__pyx_filename = __pyx_f[1]; __pyx_lineno = 792; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
    __Pyx_XDECREF(((PyObject *)__pyx_v_child));
    __pyx_v_child = ((PyArray_Descr *)__pyx_t_3);
    __pyx_t_3 = 0;
    __Pyx_XDECREF(__pyx_v_new_offset);
    __pyx_v_new_offset = __pyx_t_4;
    __pyx_t_4 = 0;

    /* "numpy.pxd":794
 *         child, new_offset = fields
 * 
 *         if (end - f) - (new_offset - offset[0]) < 15:             # <<<<<<<<<<<<<<
 *             raise RuntimeError(u"Format string allocated too short, see comment in numpy.pxd")
 * 
 */
    __pyx_t_4 = PyInt_FromLong((__pyx_v_end - __pyx_v_f)); if (unlikely(!__pyx_t_4)) {__pyx_filename = __pyx_f[1]; __pyx_lineno = 794; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
    __Pyx_GOTREF(__pyx_t_4);
    __pyx_t_3 = PyInt_FromLong((__pyx_v_offset[0])); if (unlikely(!__pyx_t_3)) {__pyx_filename = __pyx_f[1]; __pyx_lineno = 794; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
    __Pyx_GOTREF(__pyx_t_3);
    __pyx_t_5 = PyNumber_Subtract(__pyx_v_new_offset, __pyx_t_3); if (unlikely(!__pyx_t_5)) {__pyx_filename = __pyx_f[1]; __pyx_lineno = 794; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
    __Pyx_GOTREF(__pyx_t_5);
    __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
    __pyx_t_3 = PyNumber_Subtract(__pyx_t_4, __pyx_t_5); if (unlikely(!__pyx_t_3)) {__pyx_filename = __pyx_f[1]; __pyx_lineno = 794; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
    __Pyx_GOTREF(__pyx_t_3);
    __Pyx_DECREF(__pyx_t_4); __pyx_t_4 = 0;
    __Pyx_DECREF(__pyx_t_5); __pyx_t_5 = 0;
    __pyx_t_5 = PyObject_RichCompare(__pyx_t_3, __pyx_int_15, Py_LT); if (unlikely(!__pyx_t_5)) {__pyx_filename = __pyx_f[1]; __pyx_lineno = 794; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
    __Pyx_GOTREF(__pyx_t_5);
    __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
    __pyx_t_6 = __Pyx_PyObject_IsTrue(__pyx_t_5); if (unlikely(__pyx_t_6 < 0)) {__pyx_filename = __pyx_f[1]; __pyx_lineno = 794; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
    __Pyx_DECREF(__pyx_t_5); __pyx_t_5 = 0;
    if (__pyx_t_6) {

      /* "numpy.pxd":795
 * 
 *         if (end - f) - (new_offset - offset[0]) < 15:
 *             raise RuntimeError(u"Format string allocated too short, see comment in numpy.pxd")             # <<<<<<<<<<<<<<
 * 
 *         if ((child.byteorder == '>' and little_endian) or
 */
      __pyx_t_5 = PyObject_Call(__pyx_builtin_RuntimeError, ((PyObject *)__pyx_k_tuple_9), NULL); if (unlikely(!__pyx_t_5)) {__pyx_filename = __pyx_f[1]; __pyx_lineno = 795; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
      __Pyx_GOTREF(__pyx_t_5);
      __Pyx_Raise(__pyx_t_5, 0, 0, 0);
      __Pyx_DECREF(__pyx_t_5); __pyx_t_5 = 0;
      {__pyx_filename = __pyx_f[1]; __pyx_lineno = 795; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
      goto __pyx_L5;
    }
    __pyx_L5:;

    /* "numpy.pxd":797
 *             raise RuntimeError(u"Format string allocated too short, see comment in numpy.pxd")
 * 
 *         if ((child.byteorder == '>' and little_endian) or             # <<<<<<<<<<<<<<
 *             (child.byteorder == '<' and not little_endian)):
 *             raise ValueError(u"Non-native byte order not supported")
 */
    __pyx_t_6 = (__pyx_v_child->byteorder == '>');
    if (__pyx_t_6) {
      __pyx_t_7 = __pyx_v_little_endian;
    } else {
      __pyx_t_7 = __pyx_t_6;
    }
    if (!__pyx_t_7) {

      /* "numpy.pxd":798
 * 
 *         if ((child.byteorder == '>' and little_endian) or
 *             (child.byteorder == '<' and not little_endian)):             # <<<<<<<<<<<<<<
 *             raise ValueError(u"Non-native byte order not supported")
 *             # One could encode it in the format string and have Cython
 */
      __pyx_t_6 = (__pyx_v_child->byteorder == '<');
      if (__pyx_t_6) {
        __pyx_t_8 = (!__pyx_v_little_endian);
        __pyx_t_9 = __pyx_t_8;
      } else {
        __pyx_t_9 = __pyx_t_6;
      }
      __pyx_t_6 = __pyx_t_9;
    } else {
      __pyx_t_6 = __pyx_t_7;
    }
    if (__pyx_t_6) {

      /* "numpy.pxd":799
 *         if ((child.byteorder == '>' and little_endian) or
 *             (child.byteorder == '<' and not little_endian)):
 *             raise ValueError(u"Non-native byte order not supported")             # <<<<<<<<<<<<<<
 *             # One could encode it in the format string and have Cython
 *             # complain instead, BUT: < and > in format strings also imply
 */
      __pyx_t_5 = PyObject_Call(__pyx_builtin_ValueError, ((PyObject *)__pyx_k_tuple_10), NULL); if (unlikely(!__pyx_t_5)) {__pyx_filename = __pyx_f[1]; __pyx_lineno = 799; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
      __Pyx_GOTREF(__pyx_t_5);
      __Pyx_Raise(__pyx_t_5, 0, 0, 0);
      __Pyx_DECREF(__pyx_t_5); __pyx_t_5 = 0;
      {__pyx_filename = __pyx_f[1]; __pyx_lineno = 799; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
      goto __pyx_L6;
    }
    __pyx_L6:;

    /* "numpy.pxd":809
 * 
 *         # Output padding bytes
 *         while offset[0] < new_offset:             # <<<<<<<<<<<<<<
 *             f[0] = 120 # "x"; pad byte
 *             f += 1
 */
    while (1) {
      __pyx_t_5 = PyInt_FromLong((__pyx_v_offset[0])); if (unlikely(!__pyx_t_5)) {__pyx_filename = __pyx_f[1]; __pyx_lineno = 809; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
      __Pyx_GOTREF(__pyx_t_5);
      __pyx_t_3 = PyObject_RichCompare(__pyx_t_5, __pyx_v_new_offset, Py_LT); if (unlikely(!__pyx_t_3)) {__pyx_filename = __pyx_f[1]; __pyx_lineno = 809; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
      __Pyx_GOTREF(__pyx_t_3);
      __Pyx_DECREF(__pyx_t_5); __pyx_t_5 = 0;
      __pyx_t_6 = __Pyx_PyObject_IsTrue(__pyx_t_3); if (unlikely(__pyx_t_6 < 0)) {__pyx_filename = __pyx_f[1]; __pyx_lineno = 809; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
      __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
      if (!__pyx_t_6) break;

      /* "numpy.pxd":810
 *         # Output padding bytes
 *         while offset[0] < new_offset:
 *             f[0] = 120 # "x"; pad byte             # <<<<<<<<<<<<<<
 *             f += 1
 *             offset[0] += 1
 */
      (__pyx_v_f[0]) = 120;

      /* "numpy.pxd":811
 *         while offset[0] < new_offset:
 *             f[0] = 120 # "x"; pad byte
 *             f += 1             # <<<<<<<<<<<<<<
 *             offset[0] += 1
 * 
 */
      __pyx_v_f = (__pyx_v_f + 1);

      /* "numpy.pxd":812
 *             f[0] = 120 # "x"; pad byte
 *             f += 1
 *             offset[0] += 1             # <<<<<<<<<<<<<<
 * 
 *         offset[0] += child.itemsize
 */
      __pyx_t_10 = 0;
      (__pyx_v_offset[__pyx_t_10]) = ((__pyx_v_offset[__pyx_t_10]) + 1);
    }

    /* "numpy.pxd":814
 *             offset[0] += 1
 * 
 *         offset[0] += child.itemsize             # <<<<<<<<<<<<<<
 * 
 *         if not PyDataType_HASFIELDS(child):
 */
    __pyx_t_10 = 0;
    (__pyx_v_offset[__pyx_t_10]) = ((__pyx_v_offset[__pyx_t_10]) + __pyx_v_child->elsize);

    /* "numpy.pxd":816
 *         offset[0] += child.itemsize
 * 
 *         if not PyDataType_HASFIELDS(child):             # <<<<<<<<<<<<<<
 *             t = child.type_num
 *             if end - f < 5:
 */
    __pyx_t_6 = (!PyDataType_HASFIELDS(__pyx_v_child));
    if (__pyx_t_6) {

      /* "numpy.pxd":817
 * 
 *         if not PyDataType_HASFIELDS(child):
 *             t = child.type_num             # <<<<<<<<<<<<<<
 *             if end - f < 5:
 *                 raise RuntimeError(u"Format string allocated too short.")
 */
      __pyx_t_3 = PyInt_FromLong(__pyx_v_child->type_num); if (unlikely(!__pyx_t_3)) {__pyx_filename = __pyx_f[1]; __pyx_lineno = 817; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
      __Pyx_GOTREF(__pyx_t_3);
      __Pyx_XDECREF(__pyx_v_t);
      __pyx_v_t = __pyx_t_3;
      __pyx_t_3 = 0;

      /* "numpy.pxd":818
 *         if not PyDataType_HASFIELDS(child):
 *             t = child.type_num
 *             if end - f < 5:             # <<<<<<<<<<<<<<
 *                 raise RuntimeError(u"Format string allocated too short.")
 * 
 */
      __pyx_t_6 = ((__pyx_v_end - __pyx_v_f) < 5);
      if (__pyx_t_6) {

        /* "numpy.pxd":819
 *             t = child.type_num
 *             if end - f < 5:
 *                 raise RuntimeError(u"Format string allocated too short.")             # <<<<<<<<<<<<<<
 * 
 *             # Until ticket #99 is fixed, use integers to avoid warnings
 */
        __pyx_t_3 = PyObject_Call(__pyx_builtin_RuntimeError, ((PyObject *)__pyx_k_tuple_12), NULL); if (unlikely(!__pyx_t_3)) {__pyx_filename = __pyx_f[1]; __pyx_lineno = 819; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
        __Pyx_GOTREF(__pyx_t_3);
        __Pyx_Raise(__pyx_t_3, 0, 0, 0);
        __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
        {__pyx_filename = __pyx_f[1]; __pyx_lineno = 819; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
        goto __pyx_L10;
      }
      __pyx_L10:;

      /* "numpy.pxd":822
 * 
 *             # Until ticket #99 is fixed, use integers to avoid warnings
 *             if   t == NPY_BYTE:        f[0] =  98 #"b"             # <<<<<<<<<<<<<<
 *             elif t == NPY_UBYTE:       f[0] =  66 #"B"
 *             elif t == NPY_SHORT:       f[0] = 104 #"h"
 */
      __pyx_t_3 = PyInt_FromLong(NPY_BYTE); if (unlikely(!__pyx_t_3)) {__pyx_filename = __pyx_f[1]; __pyx_lineno = 822; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
      __Pyx_GOTREF(__pyx_t_3);
      __pyx_t_5 = PyObject_RichCompare(__pyx_v_t, __pyx_t_3, Py_EQ); if (unlikely(!__pyx_t_5)) {__pyx_filename = __pyx_f[1]; __pyx_lineno = 822; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
      __Pyx_GOTREF(__pyx_t_5);
      __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
      __pyx_t_6 = __Pyx_PyObject_IsTrue(__pyx_t_5); if (unlikely(__pyx_t_6 < 0)) {__pyx_filename = __pyx_f[1]; __pyx_lineno = 822; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
      __Pyx_DECREF(__pyx_t_5); __pyx_t_5 = 0;
      if (__pyx_t_6) {
        (__pyx_v_f[0]) = 98;
        goto __pyx_L11;
      }

      /* "numpy.pxd":823
 *             # Until ticket #99 is fixed, use integers to avoid warnings
 *             if   t == NPY_BYTE:        f[0] =  98 #"b"
 *             elif t == NPY_UBYTE:       f[0] =  66 #"B"             # <<<<<<<<<<<<<<
 *             elif t == NPY_SHORT:       f[0] = 104 #"h"
 *             elif t == NPY_USHORT:      f[0] =  72 #"H"
 */
      __pyx_t_5 = PyInt_FromLong(NPY_UBYTE); if (unlikely(!__pyx_t_5)) {__pyx_filename = __pyx_f[1]; __pyx_lineno = 823; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
      __Pyx_GOTREF(__pyx_t_5);
      __pyx_t_3 = PyObject_RichCompare(__pyx_v_t, __pyx_t_5, Py_EQ); if (unlikely(!__pyx_t_3)) {__pyx_filename = __pyx_f[1]; __pyx_lineno = 823; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
      __Pyx_GOTREF(__pyx_t_3);
      __Pyx_DECREF(__pyx_t_5); __pyx_t_5 = 0;
      __pyx_t_6 = __Pyx_PyObject_IsTrue(__pyx_t_3); if (unlikely(__pyx_t_6 < 0)) {__pyx_filename = __pyx_f[1]; __pyx_lineno = 823; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
      __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
      if (__pyx_t_6) {
        (__pyx_v_f[0]) = 66;
        goto __pyx_L11;
      }

      /* "numpy.pxd":824
 *             if   t == NPY_BYTE:        f[0] =  98 #"b"
 *             elif t == NPY_UBYTE:       f[0] =  66 #"B"
 *             elif t == NPY_SHORT:       f[0] = 104 #"h"             # <<<<<<<<<<<<<<
 *             elif t == NPY_USHORT:      f[0] =  72 #"H"
 *             elif t == NPY_INT:         f[0] = 105 #"i"
 */
      __pyx_t_3 = PyInt_FromLong(NPY_SHORT); if (unlikely(!__pyx_t_3)) {__pyx_filename = __pyx_f[1]; __pyx_lineno = 824; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
      __Pyx_GOTREF(__pyx_t_3);
      __pyx_t_5 = PyObject_RichCompare(__pyx_v_t, __pyx_t_3, Py_EQ); if (unlikely(!__pyx_t_5)) {__pyx_filename = __pyx_f[1]; __pyx_lineno = 824; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
      __Pyx_GOTREF(__pyx_t_5);
      __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
      __pyx_t_6 = __Pyx_PyObject_IsTrue(__pyx_t_5); if (unlikely(__pyx_t_6 < 0)) {__pyx_filename = __pyx_f[1]; __pyx_lineno = 824; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
      __Pyx_DECREF(__pyx_t_5); __pyx_t_5 = 0;
      if (__pyx_t_6) {
        (__pyx_v_f[0]) = 104;
        goto __pyx_L11;
      }

      /* "numpy.pxd":825
 *             elif t == NPY_UBYTE:       f[0] =  66 #"B"
 *             elif t == NPY_SHORT:       f[0] = 104 #"h"
 *             elif t == NPY_USHORT:      f[0] =  72 #"H"             # <<<<<<<<<<<<<<
 *             elif t == NPY_INT:         f[0] = 105 #"i"
 *             elif t == NPY_UINT:        f[0] =  73 #"I"
 */
      __pyx_t_5 = PyInt_FromLong(NPY_USHORT); if (unlikely(!__pyx_t_5)) {__pyx_filename = __pyx_f[1]; __pyx_lineno = 825; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
      __Pyx_GOTREF(__pyx_t_5);
      __pyx_t_3 = PyObject_RichCompare(__pyx_v_t, __pyx_t_5, Py_EQ); if (unlikely(!__pyx_t_3)) {__pyx_filename = __pyx_f[1]; __pyx_lineno = 825; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
      __Pyx_GOTREF(__pyx_t_3);
      __Pyx_DECREF(__pyx_t_5); __pyx_t_5 = 0;
      __pyx_t_6 = __Pyx_PyObject_IsTrue(__pyx_t_3); if (unlikely(__pyx_t_6 < 0)) {__pyx_filename = __pyx_f[1]; __pyx_lineno = 825; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
      __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
      if (__pyx_t_6) {
        (__pyx_v_f[0]) = 72;
        goto __pyx_L11;
      }

      /* "numpy.pxd":826
 *             elif t == NPY_SHORT:       f[0] = 104 #"h"
 *             elif t == NPY_USHORT:      f[0] =  72 #"H"
 *             elif t == NPY_INT:         f[0] = 105 #"i"             # <<<<<<<<<<<<<<
 *             elif t == NPY_UINT:        f[0] =  73 #"I"
 *             elif t == NPY_LONG:        f[0] = 108 #"l"
 */
      __pyx_t_3 = PyInt_FromLong(NPY_INT); if (unlikely(!__pyx_t_3)) {__pyx_filename = __pyx_f[1]; __pyx_lineno = 826; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
      __Pyx_GOTREF(__pyx_t_3);
      __pyx_t_5 = PyObject_RichCompare(__pyx_v_t, __pyx_t_3, Py_EQ); if (unlikely(!__pyx_t_5)) {__pyx_filename = __pyx_f[1]; __pyx_lineno = 826; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
      __Pyx_GOTREF(__pyx_t_5);
      __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
      __pyx_t_6 = __Pyx_PyObject_IsTrue(__pyx_t_5); if (unlikely(__pyx_t_6 < 0)) {__pyx_filename = __pyx_f[1]; __pyx_lineno = 826; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
      __Pyx_DECREF(__pyx_t_5); __pyx_t_5 = 0;
      if (__pyx_t_6) {
        (__pyx_v_f[0]) = 105;
        goto __pyx_L11;
      }

      /* "numpy.pxd":827
 *             elif t == NPY_USHORT:      f[0] =  72 #"H"
 *             elif t == NPY_INT:         f[0] = 105 #"i"
 *             elif t == NPY_UINT:        f[0] =  73 #"I"             # <<<<<<<<<<<<<<
 *             elif t == NPY_LONG:        f[0] = 108 #"l"
 *             elif t == NPY_ULONG:       f[0] = 76  #"L"
 */
      __pyx_t_5 = PyInt_FromLong(NPY_UINT); if (unlikely(!__pyx_t_5)) {__pyx_filename = __pyx_f[1]; __pyx_lineno = 827; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
      __Pyx_GOTREF(__pyx_t_5);
      __pyx_t_3 = PyObject_RichCompare(__pyx_v_t, __pyx_t_5, Py_EQ); if (unlikely(!__pyx_t_3)) {__pyx_filename = __pyx_f[1]; __pyx_lineno = 827; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
      __Pyx_GOTREF(__pyx_t_3);
      __Pyx_DECREF(__pyx_t_5); __pyx_t_5 = 0;
      __pyx_t_6 = __Pyx_PyObject_IsTrue(__pyx_t_3); if (unlikely(__pyx_t_6 < 0)) {__pyx_filename = __pyx_f[1]; __pyx_lineno = 827; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
      __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
      if (__pyx_t_6) {
        (__pyx_v_f[0]) = 73;
        goto __pyx_L11;
      }

      /* "numpy.pxd":828
 *             elif t == NPY_INT:         f[0] = 105 #"i"
 *             elif t == NPY_UINT:        f[0] =  73 #"I"
 *             elif t == NPY_LONG:        f[0] = 108 #"l"             # <<<<<<<<<<<<<<
 *             elif t == NPY_ULONG:       f[0] = 76  #"L"
 *             elif t == NPY_LONGLONG:    f[0] = 113 #"q"
 */
      __pyx_t_3 = PyInt_FromLong(NPY_LONG); if (unlikely(!__pyx_t_3)) {__pyx_filename = __pyx_f[1]; __pyx_lineno = 828; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
      __Pyx_GOTREF(__pyx_t_3);
      __pyx_t_5 = PyObject_RichCompare(__pyx_v_t, __pyx_t_3, Py_EQ); if (unlikely(!__pyx_t_5)) {__pyx_filename = __pyx_f[1]; __pyx_lineno = 828; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
      __Pyx_GOTREF(__pyx_t_5);
      __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
      __pyx_t_6 = __Pyx_PyObject_IsTrue(__pyx_t_5); if (unlikely(__pyx_t_6 < 0)) {__pyx_filename = __pyx_f[1]; __pyx_lineno = 828; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
      __Pyx_DECREF(__pyx_t_5); __pyx_t_5 = 0;
      if (__pyx_t_6) {
        (__pyx_v_f[0]) = 108;
        goto __pyx_L11;
      }

      /* "numpy.pxd":829
 *             elif t == NPY_UINT:        f[0] =  73 #"I"
 *             elif t == NPY_LONG:        f[0] = 108 #"l"
 *             elif t == NPY_ULONG:       f[0] = 76  #"L"             # <<<<<<<<<<<<<<
 *             elif t == NPY_LONGLONG:    f[0] = 113 #"q"
 *             elif t == NPY_ULONGLONG:   f[0] = 81  #"Q"
 */
      __pyx_t_5 = PyInt_FromLong(NPY_ULONG); if (unlikely(!__pyx_t_5)) {__pyx_filename = __pyx_f[1]; __pyx_lineno = 829; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
      __Pyx_GOTREF(__pyx_t_5);
      __pyx_t_3 = PyObject_RichCompare(__pyx_v_t, __pyx_t_5, Py_EQ); if (unlikely(!__pyx_t_3)) {__pyx_filename = __pyx_f[1]; __pyx_lineno = 829; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
      __Pyx_GOTREF(__pyx_t_3);
      __Pyx_DECREF(__pyx_t_5); __pyx_t_5 = 0;
      __pyx_t_6 = __Pyx_PyObject_IsTrue(__pyx_t_3); if (unlikely(__pyx_t_6 < 0)) {__pyx_filename = __pyx_f[1]; __pyx_lineno = 829; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
      __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
      if (__pyx_t_6) {
        (__pyx_v_f[0]) = 76;
        goto __pyx_L11;
      }

      /* "numpy.pxd":830
 *             elif t == NPY_LONG:        f[0] = 108 #"l"
 *             elif t == NPY_ULONG:       f[0] = 76  #"L"
 *             elif t == NPY_LONGLONG:    f[0] = 113 #"q"             # <<<<<<<<<<<<<<
 *             elif t == NPY_ULONGLONG:   f[0] = 81  #"Q"
 *             elif t == NPY_FLOAT:       f[0] = 102 #"f"
 */
      __pyx_t_3 = PyInt_FromLong(NPY_LONGLONG); if (unlikely(!__pyx_t_3)) {__pyx_filename = __pyx_f[1]; __pyx_lineno = 830; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
      __Pyx_GOTREF(__pyx_t_3);
      __pyx_t_5 = PyObject_RichCompare(__pyx_v_t, __pyx_t_3, Py_EQ); if (unlikely(!__pyx_t_5)) {__pyx_filename = __pyx_f[1]; __pyx_lineno = 830; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
      __Pyx_GOTREF(__pyx_t_5);
      __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
      __pyx_t_6 = __Pyx_PyObject_IsTrue(__pyx_t_5); if (unlikely(__pyx_t_6 < 0)) {__pyx_filename = __pyx_f[1]; __pyx_lineno = 830; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
      __Pyx_DECREF(__pyx_t_5); __pyx_t_5 = 0;
      if (__pyx_t_6) {
        (__pyx_v_f[0]) = 113;
        goto __pyx_L11;
      }

      /* "numpy.pxd":831
 *             elif t == NPY_ULONG:       f[0] = 76  #"L"
 *             elif t == NPY_LONGLONG:    f[0] = 113 #"q"
 *             elif t == NPY_ULONGLONG:   f[0] = 81  #"Q"             # <<<<<<<<<<<<<<
 *             elif t == NPY_FLOAT:       f[0] = 102 #"f"
 *             elif t == NPY_DOUBLE:      f[0] = 100 #"d"
 */
      __pyx_t_5 = PyInt_FromLong(NPY_ULONGLONG); if (unlikely(!__pyx_t_5)) {__pyx_filename = __pyx_f[1]; __pyx_lineno = 831; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
      __Pyx_GOTREF(__pyx_t_5);
      __pyx_t_3 = PyObject_RichCompare(__pyx_v_t, __pyx_t_5, Py_EQ); if (unlikely(!__pyx_t_3)) {__pyx_filename = __pyx_f[1]; __pyx_lineno = 831; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
      __Pyx_GOTREF(__pyx_t_3);
      __Pyx_DECREF(__pyx_t_5); __pyx_t_5 = 0;
      __pyx_t_6 = __Pyx_PyObject_IsTrue(__pyx_t_3); if (unlikely(__pyx_t_6 < 0)) {__pyx_filename = __pyx_f[1]; __pyx_lineno = 831; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
      __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
      if (__pyx_t_6) {
        (__pyx_v_f[0]) = 81;
        goto __pyx_L11;
      }

      /* "numpy.pxd":832
 *             elif t == NPY_LONGLONG:    f[0] = 113 #"q"
 *             elif t == NPY_ULONGLONG:   f[0] = 81  #"Q"
 *             elif t == NPY_FLOAT:       f[0] = 102 #"f"             # <<<<<<<<<<<<<<
 *             elif t == NPY_DOUBLE:      f[0] = 100 #"d"
 *             elif t == NPY_LONGDOUBLE:  f[0] = 103 #"g"
 */
      __pyx_t_3 = PyInt_FromLong(NPY_FLOAT); if (unlikely(!__pyx_t_3)) {__pyx_filename = __pyx_f[1]; __pyx_lineno = 832; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
      __Pyx_GOTREF(__pyx_t_3);
      __pyx_t_5 = PyObject_RichCompare(__pyx_v_t, __pyx_t_3, Py_EQ); if (unlikely(!__pyx_t_5)) {__pyx_filename = __pyx_f[1]; __pyx_lineno = 832; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
      __Pyx_GOTREF(__pyx_t_5);
      __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
      __pyx_t_6 = __Pyx_PyObject_IsTrue(__pyx_t_5); if (unlikely(__pyx_t_6 < 0)) {__pyx_filename = __pyx_f[1]; __pyx_lineno = 832; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
      __Pyx_DECREF(__pyx_t_5); __pyx_t_5 = 0;
      if (__pyx_t_6) {
        (__pyx_v_f[0]) = 102;
        goto __pyx_L11;
      }

      /* "numpy.pxd":833
 *             elif t == NPY_ULONGLONG:   f[0] = 81  #"Q"
 *             elif t == NPY_FLOAT:       f[0] = 102 #"f"
 *             elif t == NPY_DOUBLE:      f[0] = 100 #"d"             # <<<<<<<<<<<<<<
 *             elif t == NPY_LONGDOUBLE:  f[0] = 103 #"g"
 *             elif t == NPY_CFLOAT:      f[0] = 90; f[1] = 102; f += 1 # Zf
 */
      __pyx_t_5 = PyInt_FromLong(NPY_DOUBLE); if (unlikely(!__pyx_t_5)) {__pyx_filename = __pyx_f[1]; __pyx_lineno = 833; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
      __Pyx_GOTREF(__pyx_t_5);
      __pyx_t_3 = PyObject_RichCompare(__pyx_v_t, __pyx_t_5, Py_EQ); if (unlikely(!__pyx_t_3)) {__pyx_filename = __pyx_f[1]; __pyx_lineno = 833; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
      __Pyx_GOTREF(__pyx_t_3);
      __Pyx_DECREF(__pyx_t_5); __pyx_t_5 = 0;
      __pyx_t_6 = __Pyx_PyObject_IsTrue(__pyx_t_3); if (unlikely(__pyx_t_6 < 0)) {__pyx_filename = __pyx_f[1]; __pyx_lineno = 833; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
      __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
      if (__pyx_t_6) {
        (__pyx_v_f[0]) = 100;
        goto __pyx_L11;
      }

      /* "numpy.pxd":834
 *             elif t == NPY_FLOAT:       f[0] = 102 #"f"
 *             elif t == NPY_DOUBLE:      f[0] = 100 #"d"
 *             elif t == NPY_LONGDOUBLE:  f[0] = 103 #"g"             # <<<<<<<<<<<<<<
 *             elif t == NPY_CFLOAT:      f[0] = 90; f[1] = 102; f += 1 # Zf
 *             elif t == NPY_CDOUBLE:     f[0] = 90; f[1] = 100; f += 1 # Zd
 */
      __pyx_t_3 = PyInt_FromLong(NPY_LONGDOUBLE); if (unlikely(!__pyx_t_3)) {__pyx_filename = __pyx_f[1]; __pyx_lineno = 834; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
      __Pyx_GOTREF(__pyx_t_3);
      __pyx_t_5 = PyObject_RichCompare(__pyx_v_t, __pyx_t_3, Py_EQ); if (unlikely(!__pyx_t_5)) {__pyx_filename = __pyx_f[1]; __pyx_lineno = 834; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
      __Pyx_GOTREF(__pyx_t_5);
      __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
      __pyx_t_6 = __Pyx_PyObject_IsTrue(__pyx_t_5); if (unlikely(__pyx_t_6 < 0)) {__pyx_filename = __pyx_f[1]; __pyx_lineno = 834; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
      __Pyx_DECREF(__pyx_t_5); __pyx_t_5 = 0;
      if (__pyx_t_6) {
        (__pyx_v_f[0]) = 103;
        goto __pyx_L11;
      }

      /* "numpy.pxd":835
 *             elif t == NPY_DOUBLE:      f[0] = 100 #"d"
 *             elif t == NPY_LONGDOUBLE:  f[0] = 103 #"g"
 *             elif t == NPY_CFLOAT:      f[0] = 90; f[1] = 102; f += 1 # Zf             # <<<<<<<<<<<<<<
 *             elif t == NPY_CDOUBLE:     f[0] = 90; f[1] = 100; f += 1 # Zd
 *             elif t == NPY_CLONGDOUBLE: f[0] = 90; f[1] = 103; f += 1 # Zg
 */
      __pyx_t_5 = PyInt_FromLong(NPY_CFLOAT); if (unlikely(!__pyx_t_5)) {__pyx_filename = __pyx_f[1]; __pyx_lineno = 835; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
      __Pyx_GOTREF(__pyx_t_5);
      __pyx_t_3 = PyObject_RichCompare(__pyx_v_t, __pyx_t_5, Py_EQ); if (unlikely(!__pyx_t_3)) {__pyx_filename = __pyx_f[1]; __pyx_lineno = 835; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
      __Pyx_GOTREF(__pyx_t_3);
      __Pyx_DECREF(__pyx_t_5); __pyx_t_5 = 0;
      __pyx_t_6 = __Pyx_PyObject_IsTrue(__pyx_t_3); if (unlikely(__pyx_t_6 < 0)) {__pyx_filename = __pyx_f[1]; __pyx_lineno = 835; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
      __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
      if (__pyx_t_6) {
        (__pyx_v_f[0]) = 90;
        (__pyx_v_f[1]) = 102;
        __pyx_v_f = (__pyx_v_f + 1);
        goto __pyx_L11;
      }

      /* "numpy.pxd":836
 *             elif t == NPY_LONGDOUBLE:  f[0] = 103 #"g"
 *             elif t == NPY_CFLOAT:      f[0] = 90; f[1] = 102; f += 1 # Zf
 *             elif t == NPY_CDOUBLE:     f[0] = 90; f[1] = 100; f += 1 # Zd             # <<<<<<<<<<<<<<
 *             elif t == NPY_CLONGDOUBLE: f[0] = 90; f[1] = 103; f += 1 # Zg
 *             elif t == NPY_OBJECT:      f[0] = 79 #"O"
 */
      __pyx_t_3 = PyInt_FromLong(NPY_CDOUBLE); if (unlikely(!__pyx_t_3)) {__pyx_filename = __pyx_f[1]; __pyx_lineno = 836; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
      __Pyx_GOTREF(__pyx_t_3);
      __pyx_t_5 = PyObject_RichCompare(__pyx_v_t, __pyx_t_3, Py_EQ); if (unlikely(!__pyx_t_5)) {__pyx_filename = __pyx_f[1]; __pyx_lineno = 836; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
      __Pyx_GOTREF(__pyx_t_5);
      __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
      __pyx_t_6 = __Pyx_PyObject_IsTrue(__pyx_t_5); if (unlikely(__pyx_t_6 < 0)) {__pyx_filename = __pyx_f[1]; __pyx_lineno = 836; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
      __Pyx_DECREF(__pyx_t_5); __pyx_t_5 = 0;
      if (__pyx_t_6) {
        (__pyx_v_f[0]) = 90;
        (__pyx_v_f[1]) = 100;
        __pyx_v_f = (__pyx_v_f + 1);
        goto __pyx_L11;
      }

      /* "numpy.pxd":837
 *             elif t == NPY_CFLOAT:      f[0] = 90; f[1] = 102; f += 1 # Zf
 *             elif t == NPY_CDOUBLE:     f[0] = 90; f[1] = 100; f += 1 # Zd
 *             elif t == NPY_CLONGDOUBLE: f[0] = 90; f[1] = 103; f += 1 # Zg             # <<<<<<<<<<<<<<
 *             elif t == NPY_OBJECT:      f[0] = 79 #"O"
 *             else:
 */
      __pyx_t_5 = PyInt_FromLong(NPY_CLONGDOUBLE); if (unlikely(!__pyx_t_5)) {__pyx_filename = __pyx_f[1]; __pyx_lineno = 837; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
      __Pyx_GOTREF(__pyx_t_5);
      __pyx_t_3 = PyObject_RichCompare(__pyx_v_t, __pyx_t_5, Py_EQ); if (unlikely(!__pyx_t_3)) {__pyx_filename = __pyx_f[1]; __pyx_lineno = 837; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
      __Pyx_GOTREF(__pyx_t_3);
      __Pyx_DECREF(__pyx_t_5); __pyx_t_5 = 0;
      __pyx_t_6 = __Pyx_PyObject_IsTrue(__pyx_t_3); if (unlikely(__pyx_t_6 < 0)) {__pyx_filename = __pyx_f[1]; __pyx_lineno = 837; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
      __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
      if (__pyx_t_6) {
        (__pyx_v_f[0]) = 90;
        (__pyx_v_f[1]) = 103;
        __pyx_v_f = (__pyx_v_f + 1);
        goto __pyx_L11;
      }

      /* "numpy.pxd":838
 *             elif t == NPY_CDOUBLE:     f[0] = 90; f[1] = 100; f += 1 # Zd
 *             elif t == NPY_CLONGDOUBLE: f[0] = 90; f[1] = 103; f += 1 # Zg
 *             elif t == NPY_OBJECT:      f[0] = 79 #"O"             # <<<<<<<<<<<<<<
 *             else:
 *                 raise ValueError(u"unknown dtype code in numpy.pxd (%d)" % t)
 */
      __pyx_t_3 = PyInt_FromLong(NPY_OBJECT); if (unlikely(!__pyx_t_3)) {__pyx_filename = __pyx_f[1]; __pyx_lineno = 838; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
      __Pyx_GOTREF(__pyx_t_3);
      __pyx_t_5 = PyObject_RichCompare(__pyx_v_t, __pyx_t_3, Py_EQ); if (unlikely(!__pyx_t_5)) {__pyx_filename = __pyx_f[1]; __pyx_lineno = 838; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
      __Pyx_GOTREF(__pyx_t_5);
      __Pyx_DECREF(__pyx_t_3); __pyx_t_3 = 0;
      __pyx_t_6 = __Pyx_PyObject_IsTrue(__pyx_t_5); if (unlikely(__pyx_t_6 < 0)) {__pyx_filename = __pyx_f[1]; __pyx_lineno = 838; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
      __Pyx_DECREF(__pyx_t_5); __pyx_t_5 = 0;
      if (__pyx_t_6) {
        (__pyx_v_f[0]) = 79;
        goto __pyx_L11;
      }
      /*else*/ {

        /* "numpy.pxd":840
 *             elif t == NPY_OBJECT:      f[0] = 79 #"O"
 *             else:
 *                 raise ValueError(u"unknown dtype code in numpy.pxd (%d)" % t)             # <<<<<<<<<<<<<<
 *             f += 1
 *         else:
 */
        __pyx_t_5 = PyNumber_Remainder(((PyObject *)__pyx_kp_u_7), __pyx_v_t); if (unlikely(!__pyx_t_5)) {__pyx_filename = __pyx_f[1]; __pyx_lineno = 840; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
        __Pyx_GOTREF(((PyObject *)__pyx_t_5));
        __pyx_t_3 = PyTuple_New(1); if (unlikely(!__pyx_t_3)) {__pyx_filename = __pyx_f[1]; __pyx_lineno = 840; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
        __Pyx_GOTREF(((PyObject *)__pyx_t_3));
        PyTuple_SET_ITEM(__pyx_t_3, 0, ((PyObject *)__pyx_t_5));
        __Pyx_GIVEREF(((PyObject *)__pyx_t_5));
        __pyx_t_5 = 0;
        __pyx_t_5 = PyObject_Call(__pyx_builtin_ValueError, ((PyObject *)__pyx_t_3), NULL); if (unlikely(!__pyx_t_5)) {__pyx_filename = __pyx_f[1]; __pyx_lineno = 840; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
        __Pyx_GOTREF(__pyx_t_5);
        __Pyx_DECREF(((PyObject *)__pyx_t_3)); __pyx_t_3 = 0;
        __Pyx_Raise(__pyx_t_5, 0, 0, 0);
        __Pyx_DECREF(__pyx_t_5); __pyx_t_5 = 0;
        {__pyx_filename = __pyx_f[1]; __pyx_lineno = 840; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
      }
      __pyx_L11:;

      /* "numpy.pxd":841
 *             else:
 *                 raise ValueError(u"unknown dtype code in numpy.pxd (%d)" % t)
 *             f += 1             # <<<<<<<<<<<<<<
 *         else:
 *             # Cython ignores struct boundary information ("T{...}"),
 */
      __pyx_v_f = (__pyx_v_f + 1);
      goto __pyx_L9;
    }
    /*else*/ {

      /* "numpy.pxd":845
 *             # Cython ignores struct boundary information ("T{...}"),
 *             # so don't output it
 *             f = _util_dtypestring(child, f, end, offset)             # <<<<<<<<<<<<<<
 *     return f
 * 
 */
      __pyx_t_11 = __pyx_f_5numpy__util_dtypestring(__pyx_v_child, __pyx_v_f, __pyx_v_end, __pyx_v_offset); if (unlikely(__pyx_t_11 == NULL)) {__pyx_filename = __pyx_f[1]; __pyx_lineno = 845; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
      __pyx_v_f = __pyx_t_11;
    }
    __pyx_L9:;
  }
  __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;

  /* "numpy.pxd":846
 *             # so don't output it
 *             f = _util_dtypestring(child, f, end, offset)
 *     return f             # <<<<<<<<<<<<<<
 * 
 * 
 */
  __pyx_r = __pyx_v_f;
  goto __pyx_L0;

  __pyx_r = 0;
  goto __pyx_L0;
  __pyx_L1_error:;
  __Pyx_XDECREF(__pyx_t_1);
  __Pyx_XDECREF(__pyx_t_3);
  __Pyx_XDECREF(__pyx_t_4);
  __Pyx_XDECREF(__pyx_t_5);
  __Pyx_AddTraceback("numpy._util_dtypestring", __pyx_clineno, __pyx_lineno, __pyx_filename);
  __pyx_r = NULL;
  __pyx_L0:;
  __Pyx_XDECREF((PyObject *)__pyx_v_child);
  __Pyx_XDECREF(__pyx_v_fields);
  __Pyx_XDECREF(__pyx_v_childname);
  __Pyx_XDECREF(__pyx_v_new_offset);
  __Pyx_XDECREF(__pyx_v_t);
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

/* "numpy.pxd":961
 * 
 * 
 * cdef inline void set_array_base(ndarray arr, object base):             # <<<<<<<<<<<<<<
 *      cdef PyObject* baseptr
 *      if base is None:
 */

static CYTHON_INLINE void __pyx_f_5numpy_set_array_base(PyArrayObject *__pyx_v_arr, PyObject *__pyx_v_base) {
  PyObject *__pyx_v_baseptr;
  __Pyx_RefNannyDeclarations
  int __pyx_t_1;
  __Pyx_RefNannySetupContext("set_array_base");

  /* "numpy.pxd":963
 * cdef inline void set_array_base(ndarray arr, object base):
 *      cdef PyObject* baseptr
 *      if base is None:             # <<<<<<<<<<<<<<
 *          baseptr = NULL
 *      else:
 */
  __pyx_t_1 = (__pyx_v_base == Py_None);
  if (__pyx_t_1) {

    /* "numpy.pxd":964
 *      cdef PyObject* baseptr
 *      if base is None:
 *          baseptr = NULL             # <<<<<<<<<<<<<<
 *      else:
 *          Py_INCREF(base) # important to do this before decref below!
 */
    __pyx_v_baseptr = NULL;
    goto __pyx_L3;
  }
  /*else*/ {

    /* "numpy.pxd":966
 *          baseptr = NULL
 *      else:
 *          Py_INCREF(base) # important to do this before decref below!             # <<<<<<<<<<<<<<
 *          baseptr = <PyObject*>base
 *      Py_XDECREF(arr.base)
 */
    Py_INCREF(__pyx_v_base);

    /* "numpy.pxd":967
 *      else:
 *          Py_INCREF(base) # important to do this before decref below!
 *          baseptr = <PyObject*>base             # <<<<<<<<<<<<<<
 *      Py_XDECREF(arr.base)
 *      arr.base = baseptr
 */
    __pyx_v_baseptr = ((PyObject *)__pyx_v_base);
  }
  __pyx_L3:;

  /* "numpy.pxd":968
 *          Py_INCREF(base) # important to do this before decref below!
 *          baseptr = <PyObject*>base
 *      Py_XDECREF(arr.base)             # <<<<<<<<<<<<<<
 *      arr.base = baseptr
 * 
 */
  Py_XDECREF(__pyx_v_arr->base);

  /* "numpy.pxd":969
 *          baseptr = <PyObject*>base
 *      Py_XDECREF(arr.base)
 *      arr.base = baseptr             # <<<<<<<<<<<<<<
 * 
 * cdef inline object get_array_base(ndarray arr):
 */
  __pyx_v_arr->base = __pyx_v_baseptr;

  __Pyx_RefNannyFinishContext();
}

/* "numpy.pxd":971
 *      arr.base = baseptr
 * 
 * cdef inline object get_array_base(ndarray arr):             # <<<<<<<<<<<<<<
 *     if arr.base is NULL:
 *         return None
 */

static CYTHON_INLINE PyObject *__pyx_f_5numpy_get_array_base(PyArrayObject *__pyx_v_arr) {
  PyObject *__pyx_r = NULL;
  __Pyx_RefNannyDeclarations
  int __pyx_t_1;
  __Pyx_RefNannySetupContext("get_array_base");

  /* "numpy.pxd":972
 * 
 * cdef inline object get_array_base(ndarray arr):
 *     if arr.base is NULL:             # <<<<<<<<<<<<<<
 *         return None
 *     else:
 */
  __pyx_t_1 = (__pyx_v_arr->base == NULL);
  if (__pyx_t_1) {

    /* "numpy.pxd":973
 * cdef inline object get_array_base(ndarray arr):
 *     if arr.base is NULL:
 *         return None             # <<<<<<<<<<<<<<
 *     else:
 *         return <object>arr.base
 */
    __Pyx_XDECREF(__pyx_r);
    __Pyx_INCREF(Py_None);
    __pyx_r = Py_None;
    goto __pyx_L0;
    goto __pyx_L3;
  }
  /*else*/ {

    /* "numpy.pxd":975
 *         return None
 *     else:
 *         return <object>arr.base             # <<<<<<<<<<<<<<
 */
    __Pyx_XDECREF(__pyx_r);
    __Pyx_INCREF(((PyObject *)__pyx_v_arr->base));
    __pyx_r = ((PyObject *)__pyx_v_arr->base);
    goto __pyx_L0;
  }
  __pyx_L3:;

  __pyx_r = Py_None; __Pyx_INCREF(Py_None);
  __pyx_L0:;
  __Pyx_XGIVEREF(__pyx_r);
  __Pyx_RefNannyFinishContext();
  return __pyx_r;
}

static PyMethodDef __pyx_methods[] = {
  {__Pyx_NAMESTR("realSinusoid_"), (PyCFunction)__pyx_pf_5atom__realSinusoid_, METH_VARARGS|METH_KEYWORDS, __Pyx_DOCSTR(0)},
  {__Pyx_NAMESTR("gabor_"), (PyCFunction)__pyx_pf_5atom__1gabor_, METH_VARARGS|METH_KEYWORDS, __Pyx_DOCSTR(__pyx_doc_5atom__1gabor_)},
  {__Pyx_NAMESTR("gaborFM_"), (PyCFunction)__pyx_pf_5atom__2gaborFM_, METH_VARARGS|METH_KEYWORDS, __Pyx_DOCSTR(__pyx_doc_5atom__2gaborFM_)},
  {__Pyx_NAMESTR("hann_"), (PyCFunction)__pyx_pf_5atom__3hann_, METH_VARARGS|METH_KEYWORDS, __Pyx_DOCSTR(__pyx_doc_5atom__3hann_)},
  {__Pyx_NAMESTR("hannFM_"), (PyCFunction)__pyx_pf_5atom__4hannFM_, METH_VARARGS|METH_KEYWORDS, __Pyx_DOCSTR(__pyx_doc_5atom__4hannFM_)},
  {__Pyx_NAMESTR("blackman_"), (PyCFunction)__pyx_pf_5atom__5blackman_, METH_VARARGS|METH_KEYWORDS, __Pyx_DOCSTR(__pyx_doc_5atom__5blackman_)},
  {__Pyx_NAMESTR("blackmanFM_"), (PyCFunction)__pyx_pf_5atom__6blackmanFM_, METH_VARARGS|METH_KEYWORDS, __Pyx_DOCSTR(__pyx_doc_5atom__6blackmanFM_)},
  {__Pyx_NAMESTR("gamma_"), (PyCFunction)__pyx_pf_5atom__7gamma_, METH_VARARGS|METH_KEYWORDS, __Pyx_DOCSTR(0)},
  {__Pyx_NAMESTR("gammaFM_"), (PyCFunction)__pyx_pf_5atom__8gammaFM_, METH_VARARGS|METH_KEYWORDS, __Pyx_DOCSTR(0)},
  {__Pyx_NAMESTR("damped_"), (PyCFunction)__pyx_pf_5atom__9damped_, METH_VARARGS|METH_KEYWORDS, __Pyx_DOCSTR(0)},
  {__Pyx_NAMESTR("dampedFM_"), (PyCFunction)__pyx_pf_5atom__10dampedFM_, METH_VARARGS|METH_KEYWORDS, __Pyx_DOCSTR(0)},
  {__Pyx_NAMESTR("fof_"), (PyCFunction)__pyx_pf_5atom__11fof_, METH_VARARGS|METH_KEYWORDS, __Pyx_DOCSTR(0)},
  {__Pyx_NAMESTR("fofFM_"), (PyCFunction)__pyx_pf_5atom__12fofFM_, METH_VARARGS|METH_KEYWORDS, __Pyx_DOCSTR(0)},
  {0, 0, 0, 0}
};

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef __pyx_moduledef = {
    PyModuleDef_HEAD_INIT,
    __Pyx_NAMESTR("atom_"),
    0, /* m_doc */
    -1, /* m_size */
    __pyx_methods /* m_methods */,
    NULL, /* m_reload */
    NULL, /* m_traverse */
    NULL, /* m_clear */
    NULL /* m_free */
};
#endif

static __Pyx_StringTabEntry __pyx_string_tab[] = {
  {&__pyx_kp_u_1, __pyx_k_1, sizeof(__pyx_k_1), 0, 1, 0, 0},
  {&__pyx_kp_u_11, __pyx_k_11, sizeof(__pyx_k_11), 0, 1, 0, 0},
  {&__pyx_kp_u_3, __pyx_k_3, sizeof(__pyx_k_3), 0, 1, 0, 0},
  {&__pyx_kp_u_5, __pyx_k_5, sizeof(__pyx_k_5), 0, 1, 0, 0},
  {&__pyx_kp_u_7, __pyx_k_7, sizeof(__pyx_k_7), 0, 1, 0, 0},
  {&__pyx_kp_u_8, __pyx_k_8, sizeof(__pyx_k_8), 0, 1, 0, 0},
  {&__pyx_n_s__N, __pyx_k__N, sizeof(__pyx_k__N), 0, 0, 1, 1},
  {&__pyx_n_s__RuntimeError, __pyx_k__RuntimeError, sizeof(__pyx_k__RuntimeError), 0, 0, 1, 1},
  {&__pyx_n_s__ValueError, __pyx_k__ValueError, sizeof(__pyx_k__ValueError), 0, 0, 1, 1},
  {&__pyx_n_s____main__, __pyx_k____main__, sizeof(__pyx_k____main__), 0, 0, 1, 1},
  {&__pyx_n_s____test__, __pyx_k____test__, sizeof(__pyx_k____test__), 0, 0, 1, 1},
  {&__pyx_n_s__arange, __pyx_k__arange, sizeof(__pyx_k__arange), 0, 0, 1, 1},
  {&__pyx_n_s__bandwidth, __pyx_k__bandwidth, sizeof(__pyx_k__bandwidth), 0, 0, 1, 1},
  {&__pyx_n_s__chirp, __pyx_k__chirp, sizeof(__pyx_k__chirp), 0, 0, 1, 1},
  {&__pyx_n_s__cos, __pyx_k__cos, sizeof(__pyx_k__cos), 0, 0, 1, 1},
  {&__pyx_n_s__damp, __pyx_k__damp, sizeof(__pyx_k__damp), 0, 0, 1, 1},
  {&__pyx_n_s__decay_n, __pyx_k__decay_n, sizeof(__pyx_k__decay_n), 0, 0, 1, 1},
  {&__pyx_n_s__depth, __pyx_k__depth, sizeof(__pyx_k__depth), 0, 0, 1, 1},
  {&__pyx_n_s__dtype, __pyx_k__dtype, sizeof(__pyx_k__dtype), 0, 0, 1, 1},
  {&__pyx_n_s__max, __pyx_k__max, sizeof(__pyx_k__max), 0, 0, 1, 1},
  {&__pyx_n_s__np, __pyx_k__np, sizeof(__pyx_k__np), 0, 0, 1, 1},
  {&__pyx_n_s__numpy, __pyx_k__numpy, sizeof(__pyx_k__numpy), 0, 0, 1, 1},
  {&__pyx_n_s__omega, __pyx_k__omega, sizeof(__pyx_k__omega), 0, 0, 1, 1},
  {&__pyx_n_s__omega_m, __pyx_k__omega_m, sizeof(__pyx_k__omega_m), 0, 0, 1, 1},
  {&__pyx_n_s__order, __pyx_k__order, sizeof(__pyx_k__order), 0, 0, 1, 1},
  {&__pyx_n_s__phi, __pyx_k__phi, sizeof(__pyx_k__phi), 0, 0, 1, 1},
  {&__pyx_n_s__phi_m, __pyx_k__phi_m, sizeof(__pyx_k__phi_m), 0, 0, 1, 1},
  {&__pyx_n_s__range, __pyx_k__range, sizeof(__pyx_k__range), 0, 0, 1, 1},
  {&__pyx_n_s__rise_n, __pyx_k__rise_n, sizeof(__pyx_k__rise_n), 0, 0, 1, 1},
  {&__pyx_n_s__zeros, __pyx_k__zeros, sizeof(__pyx_k__zeros), 0, 0, 1, 1},
  {0, 0, 0, 0, 0, 0, 0}
};
static int __Pyx_InitCachedBuiltins(void) {
  __pyx_builtin_range = __Pyx_GetName(__pyx_b, __pyx_n_s__range); if (!__pyx_builtin_range) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 73; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __pyx_builtin_max = __Pyx_GetName(__pyx_b, __pyx_n_s__max); if (!__pyx_builtin_max) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 247; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __pyx_builtin_ValueError = __Pyx_GetName(__pyx_b, __pyx_n_s__ValueError); if (!__pyx_builtin_ValueError) {__pyx_filename = __pyx_f[1]; __pyx_lineno = 211; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __pyx_builtin_RuntimeError = __Pyx_GetName(__pyx_b, __pyx_n_s__RuntimeError); if (!__pyx_builtin_RuntimeError) {__pyx_filename = __pyx_f[1]; __pyx_lineno = 795; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  return 0;
  __pyx_L1_error:;
  return -1;
}

static int __Pyx_InitCachedConstants(void) {
  __Pyx_RefNannyDeclarations
  __Pyx_RefNannySetupContext("__Pyx_InitCachedConstants");

  /* "numpy.pxd":211
 *             if ((flags & pybuf.PyBUF_C_CONTIGUOUS == pybuf.PyBUF_C_CONTIGUOUS)
 *                 and not PyArray_CHKFLAGS(self, NPY_C_CONTIGUOUS)):
 *                 raise ValueError(u"ndarray is not C contiguous")             # <<<<<<<<<<<<<<
 * 
 *             if ((flags & pybuf.PyBUF_F_CONTIGUOUS == pybuf.PyBUF_F_CONTIGUOUS)
 */
  __pyx_k_tuple_2 = PyTuple_New(1); if (unlikely(!__pyx_k_tuple_2)) {__pyx_filename = __pyx_f[1]; __pyx_lineno = 211; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(((PyObject *)__pyx_k_tuple_2));
  __Pyx_INCREF(((PyObject *)__pyx_kp_u_1));
  PyTuple_SET_ITEM(__pyx_k_tuple_2, 0, ((PyObject *)__pyx_kp_u_1));
  __Pyx_GIVEREF(((PyObject *)__pyx_kp_u_1));
  __Pyx_GIVEREF(((PyObject *)__pyx_k_tuple_2));

  /* "numpy.pxd":215
 *             if ((flags & pybuf.PyBUF_F_CONTIGUOUS == pybuf.PyBUF_F_CONTIGUOUS)
 *                 and not PyArray_CHKFLAGS(self, NPY_F_CONTIGUOUS)):
 *                 raise ValueError(u"ndarray is not Fortran contiguous")             # <<<<<<<<<<<<<<
 * 
 *             info.buf = PyArray_DATA(self)
 */
  __pyx_k_tuple_4 = PyTuple_New(1); if (unlikely(!__pyx_k_tuple_4)) {__pyx_filename = __pyx_f[1]; __pyx_lineno = 215; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(((PyObject *)__pyx_k_tuple_4));
  __Pyx_INCREF(((PyObject *)__pyx_kp_u_3));
  PyTuple_SET_ITEM(__pyx_k_tuple_4, 0, ((PyObject *)__pyx_kp_u_3));
  __Pyx_GIVEREF(((PyObject *)__pyx_kp_u_3));
  __Pyx_GIVEREF(((PyObject *)__pyx_k_tuple_4));

  /* "numpy.pxd":253
 *                 if ((descr.byteorder == '>' and little_endian) or
 *                     (descr.byteorder == '<' and not little_endian)):
 *                     raise ValueError(u"Non-native byte order not supported")             # <<<<<<<<<<<<<<
 *                 if   t == NPY_BYTE:        f = "b"
 *                 elif t == NPY_UBYTE:       f = "B"
 */
  __pyx_k_tuple_6 = PyTuple_New(1); if (unlikely(!__pyx_k_tuple_6)) {__pyx_filename = __pyx_f[1]; __pyx_lineno = 253; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(((PyObject *)__pyx_k_tuple_6));
  __Pyx_INCREF(((PyObject *)__pyx_kp_u_5));
  PyTuple_SET_ITEM(__pyx_k_tuple_6, 0, ((PyObject *)__pyx_kp_u_5));
  __Pyx_GIVEREF(((PyObject *)__pyx_kp_u_5));
  __Pyx_GIVEREF(((PyObject *)__pyx_k_tuple_6));

  /* "numpy.pxd":795
 * 
 *         if (end - f) - (new_offset - offset[0]) < 15:
 *             raise RuntimeError(u"Format string allocated too short, see comment in numpy.pxd")             # <<<<<<<<<<<<<<
 * 
 *         if ((child.byteorder == '>' and little_endian) or
 */
  __pyx_k_tuple_9 = PyTuple_New(1); if (unlikely(!__pyx_k_tuple_9)) {__pyx_filename = __pyx_f[1]; __pyx_lineno = 795; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(((PyObject *)__pyx_k_tuple_9));
  __Pyx_INCREF(((PyObject *)__pyx_kp_u_8));
  PyTuple_SET_ITEM(__pyx_k_tuple_9, 0, ((PyObject *)__pyx_kp_u_8));
  __Pyx_GIVEREF(((PyObject *)__pyx_kp_u_8));
  __Pyx_GIVEREF(((PyObject *)__pyx_k_tuple_9));

  /* "numpy.pxd":799
 *         if ((child.byteorder == '>' and little_endian) or
 *             (child.byteorder == '<' and not little_endian)):
 *             raise ValueError(u"Non-native byte order not supported")             # <<<<<<<<<<<<<<
 *             # One could encode it in the format string and have Cython
 *             # complain instead, BUT: < and > in format strings also imply
 */
  __pyx_k_tuple_10 = PyTuple_New(1); if (unlikely(!__pyx_k_tuple_10)) {__pyx_filename = __pyx_f[1]; __pyx_lineno = 799; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(((PyObject *)__pyx_k_tuple_10));
  __Pyx_INCREF(((PyObject *)__pyx_kp_u_5));
  PyTuple_SET_ITEM(__pyx_k_tuple_10, 0, ((PyObject *)__pyx_kp_u_5));
  __Pyx_GIVEREF(((PyObject *)__pyx_kp_u_5));
  __Pyx_GIVEREF(((PyObject *)__pyx_k_tuple_10));

  /* "numpy.pxd":819
 *             t = child.type_num
 *             if end - f < 5:
 *                 raise RuntimeError(u"Format string allocated too short.")             # <<<<<<<<<<<<<<
 * 
 *             # Until ticket #99 is fixed, use integers to avoid warnings
 */
  __pyx_k_tuple_12 = PyTuple_New(1); if (unlikely(!__pyx_k_tuple_12)) {__pyx_filename = __pyx_f[1]; __pyx_lineno = 819; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(((PyObject *)__pyx_k_tuple_12));
  __Pyx_INCREF(((PyObject *)__pyx_kp_u_11));
  PyTuple_SET_ITEM(__pyx_k_tuple_12, 0, ((PyObject *)__pyx_kp_u_11));
  __Pyx_GIVEREF(((PyObject *)__pyx_kp_u_11));
  __Pyx_GIVEREF(((PyObject *)__pyx_k_tuple_12));
  __Pyx_RefNannyFinishContext();
  return 0;
  __pyx_L1_error:;
  __Pyx_RefNannyFinishContext();
  return -1;
}

static int __Pyx_InitGlobals(void) {
  if (__Pyx_InitStrings(__pyx_string_tab) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1; __pyx_clineno = __LINE__; goto __pyx_L1_error;};
  __pyx_int_15 = PyInt_FromLong(15); if (unlikely(!__pyx_int_15)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1; __pyx_clineno = __LINE__; goto __pyx_L1_error;};
  return 0;
  __pyx_L1_error:;
  return -1;
}

#if PY_MAJOR_VERSION < 3
PyMODINIT_FUNC initatom_(void); /*proto*/
PyMODINIT_FUNC initatom_(void)
#else
PyMODINIT_FUNC PyInit_atom_(void); /*proto*/
PyMODINIT_FUNC PyInit_atom_(void)
#endif
{
  PyObject *__pyx_t_1 = NULL;
  __Pyx_RefNannyDeclarations
  #if CYTHON_REFNANNY
  __Pyx_RefNanny = __Pyx_RefNannyImportAPI("refnanny");
  if (!__Pyx_RefNanny) {
      PyErr_Clear();
      __Pyx_RefNanny = __Pyx_RefNannyImportAPI("Cython.Runtime.refnanny");
      if (!__Pyx_RefNanny)
          Py_FatalError("failed to import 'refnanny' module");
  }
  #endif
  __Pyx_RefNannySetupContext("PyMODINIT_FUNC PyInit_atom_(void)");
  if ( __Pyx_check_binary_version() < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __pyx_empty_tuple = PyTuple_New(0); if (unlikely(!__pyx_empty_tuple)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __pyx_empty_bytes = PyBytes_FromStringAndSize("", 0); if (unlikely(!__pyx_empty_bytes)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  #ifdef __pyx_binding_PyCFunctionType_USED
  if (__pyx_binding_PyCFunctionType_init() < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  #endif
  /*--- Library function declarations ---*/
  /*--- Threads initialization code ---*/
  #if defined(__PYX_FORCE_INIT_THREADS) && __PYX_FORCE_INIT_THREADS
  #ifdef WITH_THREAD /* Python build with threading support? */
  PyEval_InitThreads();
  #endif
  #endif
  /*--- Module creation code ---*/
  #if PY_MAJOR_VERSION < 3
  __pyx_m = Py_InitModule4(__Pyx_NAMESTR("atom_"), __pyx_methods, 0, 0, PYTHON_API_VERSION);
  #else
  __pyx_m = PyModule_Create(&__pyx_moduledef);
  #endif
  if (!__pyx_m) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1; __pyx_clineno = __LINE__; goto __pyx_L1_error;};
  #if PY_MAJOR_VERSION < 3
  Py_INCREF(__pyx_m);
  #endif
  __pyx_b = PyImport_AddModule(__Pyx_NAMESTR(__Pyx_BUILTIN_MODULE_NAME));
  if (!__pyx_b) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1; __pyx_clineno = __LINE__; goto __pyx_L1_error;};
  if (__Pyx_SetAttrString(__pyx_m, "__builtins__", __pyx_b) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1; __pyx_clineno = __LINE__; goto __pyx_L1_error;};
  /*--- Initialize various global constants etc. ---*/
  if (unlikely(__Pyx_InitGlobals() < 0)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  if (__pyx_module_is_main_atom_) {
    if (__Pyx_SetAttrString(__pyx_m, "__name__", __pyx_n_s____main__) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1; __pyx_clineno = __LINE__; goto __pyx_L1_error;};
  }
  /*--- Builtin init code ---*/
  if (unlikely(__Pyx_InitCachedBuiltins() < 0)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  /*--- Constants init code ---*/
  if (unlikely(__Pyx_InitCachedConstants() < 0)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  /*--- Global init code ---*/
  /*--- Variable export code ---*/
  /*--- Function export code ---*/
  /*--- Type init code ---*/
  /*--- Type import code ---*/
  __pyx_ptype_5numpy_dtype = __Pyx_ImportType("numpy", "dtype", sizeof(PyArray_Descr), 0); if (unlikely(!__pyx_ptype_5numpy_dtype)) {__pyx_filename = __pyx_f[1]; __pyx_lineno = 151; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __pyx_ptype_5numpy_flatiter = __Pyx_ImportType("numpy", "flatiter", sizeof(PyArrayIterObject), 0); if (unlikely(!__pyx_ptype_5numpy_flatiter)) {__pyx_filename = __pyx_f[1]; __pyx_lineno = 161; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __pyx_ptype_5numpy_broadcast = __Pyx_ImportType("numpy", "broadcast", sizeof(PyArrayMultiIterObject), 0); if (unlikely(!__pyx_ptype_5numpy_broadcast)) {__pyx_filename = __pyx_f[1]; __pyx_lineno = 165; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __pyx_ptype_5numpy_ndarray = __Pyx_ImportType("numpy", "ndarray", sizeof(PyArrayObject), 0); if (unlikely(!__pyx_ptype_5numpy_ndarray)) {__pyx_filename = __pyx_f[1]; __pyx_lineno = 174; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __pyx_ptype_5numpy_ufunc = __Pyx_ImportType("numpy", "ufunc", sizeof(PyUFuncObject), 0); if (unlikely(!__pyx_ptype_5numpy_ufunc)) {__pyx_filename = __pyx_f[1]; __pyx_lineno = 857; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  /*--- Variable import code ---*/
  /*--- Function import code ---*/
  /*--- Execution code ---*/

  /* "atom_.pyx":2
 * cimport cython
 * import numpy as np             # <<<<<<<<<<<<<<
 * cimport numpy as np
 * 
 */
  __pyx_t_1 = __Pyx_Import(((PyObject *)__pyx_n_s__numpy), 0, -1); if (unlikely(!__pyx_t_1)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(__pyx_t_1);
  if (PyObject_SetAttr(__pyx_m, __pyx_n_s__np, __pyx_t_1) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;

  /* "atom_.pyx":14
 *     double log(double)
 * 
 * cdef double pi = 3.1415926535897931             # <<<<<<<<<<<<<<
 * 
 * 
 */
  __pyx_v_5atom__pi = 3.1415926535897931;

  /* "atom_.pyx":1
 * cimport cython             # <<<<<<<<<<<<<<
 * import numpy as np
 * cimport numpy as np
 */
  __pyx_t_1 = PyDict_New(); if (unlikely(!__pyx_t_1)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_GOTREF(((PyObject *)__pyx_t_1));
  if (PyObject_SetAttr(__pyx_m, __pyx_n_s____test__, ((PyObject *)__pyx_t_1)) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
  __Pyx_DECREF(((PyObject *)__pyx_t_1)); __pyx_t_1 = 0;

  /* "numpy.pxd":971
 *      arr.base = baseptr
 * 
 * cdef inline object get_array_base(ndarray arr):             # <<<<<<<<<<<<<<
 *     if arr.base is NULL:
 *         return None
 */
  goto __pyx_L0;
  __pyx_L1_error:;
  __Pyx_XDECREF(__pyx_t_1);
  if (__pyx_m) {
    __Pyx_AddTraceback("init atom_", __pyx_clineno, __pyx_lineno, __pyx_filename);
    Py_DECREF(__pyx_m); __pyx_m = 0;
  } else if (!PyErr_Occurred()) {
    PyErr_SetString(PyExc_ImportError, "init atom_");
  }
  __pyx_L0:;
  __Pyx_RefNannyFinishContext();
  #if PY_MAJOR_VERSION < 3
  return;
  #else
  return __pyx_m;
  #endif
}

/* Runtime support code */

#if CYTHON_REFNANNY
static __Pyx_RefNannyAPIStruct *__Pyx_RefNannyImportAPI(const char *modname) {
    PyObject *m = NULL, *p = NULL;
    void *r = NULL;
    m = PyImport_ImportModule((char *)modname);
    if (!m) goto end;
    p = PyObject_GetAttrString(m, (char *)"RefNannyAPI");
    if (!p) goto end;
    r = PyLong_AsVoidPtr(p);
end:
    Py_XDECREF(p);
    Py_XDECREF(m);
    return (__Pyx_RefNannyAPIStruct *)r;
}
#endif /* CYTHON_REFNANNY */

static PyObject *__Pyx_GetName(PyObject *dict, PyObject *name) {
    PyObject *result;
    result = PyObject_GetAttr(dict, name);
    if (!result) {
        if (dict != __pyx_b) {
            PyErr_Clear();
            result = PyObject_GetAttr(__pyx_b, name);
        }
        if (!result) {
            PyErr_SetObject(PyExc_NameError, name);
        }
    }
    return result;
}

static CYTHON_INLINE int __Pyx_TypeTest(PyObject *obj, PyTypeObject *type) {
    if (unlikely(!type)) {
        PyErr_Format(PyExc_SystemError, "Missing type object");
        return 0;
    }
    if (likely(PyObject_TypeCheck(obj, type)))
        return 1;
    PyErr_Format(PyExc_TypeError, "Cannot convert %.200s to %.200s",
                 Py_TYPE(obj)->tp_name, type->tp_name);
    return 0;
}

static CYTHON_INLINE int __Pyx_IsLittleEndian(void) {
  unsigned int n = 1;
  return *(unsigned char*)(&n) != 0;
}

typedef struct {
  __Pyx_StructField root;
  __Pyx_BufFmt_StackElem* head;
  size_t fmt_offset;
  size_t new_count, enc_count;
  int is_complex;
  char enc_type;
  char new_packmode;
  char enc_packmode;
} __Pyx_BufFmt_Context;

static void __Pyx_BufFmt_Init(__Pyx_BufFmt_Context* ctx,
                              __Pyx_BufFmt_StackElem* stack,
                              __Pyx_TypeInfo* type) {
  stack[0].field = &ctx->root;
  stack[0].parent_offset = 0;
  ctx->root.type = type;
  ctx->root.name = "buffer dtype";
  ctx->root.offset = 0;
  ctx->head = stack;
  ctx->head->field = &ctx->root;
  ctx->fmt_offset = 0;
  ctx->head->parent_offset = 0;
  ctx->new_packmode = '@';
  ctx->enc_packmode = '@';
  ctx->new_count = 1;
  ctx->enc_count = 0;
  ctx->enc_type = 0;
  ctx->is_complex = 0;
  while (type->typegroup == 'S') {
    ++ctx->head;
    ctx->head->field = type->fields;
    ctx->head->parent_offset = 0;
    type = type->fields->type;
  }
}

static int __Pyx_BufFmt_ParseNumber(const char** ts) {
    int count;
    const char* t = *ts;
    if (*t < '0' || *t > '9') {
      return -1;
    } else {
        count = *t++ - '0';
        while (*t >= '0' && *t < '9') {
            count *= 10;
            count += *t++ - '0';
        }
    }
    *ts = t;
    return count;
}

static void __Pyx_BufFmt_RaiseUnexpectedChar(char ch) {
  PyErr_Format(PyExc_ValueError,
               "Unexpected format string character: '%c'", ch);
}

static const char* __Pyx_BufFmt_DescribeTypeChar(char ch, int is_complex) {
  switch (ch) {
    case 'b': return "'char'";
    case 'B': return "'unsigned char'";
    case 'h': return "'short'";
    case 'H': return "'unsigned short'";
    case 'i': return "'int'";
    case 'I': return "'unsigned int'";
    case 'l': return "'long'";
    case 'L': return "'unsigned long'";
    case 'q': return "'long long'";
    case 'Q': return "'unsigned long long'";
    case 'f': return (is_complex ? "'complex float'" : "'float'");
    case 'd': return (is_complex ? "'complex double'" : "'double'");
    case 'g': return (is_complex ? "'complex long double'" : "'long double'");
    case 'T': return "a struct";
    case 'O': return "Python object";
    case 'P': return "a pointer";
    case 0: return "end";
    default: return "unparseable format string";
  }
}

static size_t __Pyx_BufFmt_TypeCharToStandardSize(char ch, int is_complex) {
  switch (ch) {
    case '?': case 'c': case 'b': case 'B': return 1;
    case 'h': case 'H': return 2;
    case 'i': case 'I': case 'l': case 'L': return 4;
    case 'q': case 'Q': return 8;
    case 'f': return (is_complex ? 8 : 4);
    case 'd': return (is_complex ? 16 : 8);
    case 'g': {
      PyErr_SetString(PyExc_ValueError, "Python does not define a standard format string size for long double ('g')..");
      return 0;
    }
    case 'O': case 'P': return sizeof(void*);
    default:
      __Pyx_BufFmt_RaiseUnexpectedChar(ch);
      return 0;
    }
}

static size_t __Pyx_BufFmt_TypeCharToNativeSize(char ch, int is_complex) {
  switch (ch) {
    case 'c': case 'b': case 'B': return 1;
    case 'h': case 'H': return sizeof(short);
    case 'i': case 'I': return sizeof(int);
    case 'l': case 'L': return sizeof(long);
    #ifdef HAVE_LONG_LONG
    case 'q': case 'Q': return sizeof(PY_LONG_LONG);
    #endif
    case 'f': return sizeof(float) * (is_complex ? 2 : 1);
    case 'd': return sizeof(double) * (is_complex ? 2 : 1);
    case 'g': return sizeof(long double) * (is_complex ? 2 : 1);
    case 'O': case 'P': return sizeof(void*);
    default: {
      __Pyx_BufFmt_RaiseUnexpectedChar(ch);
      return 0;
    }
  }
}

typedef struct { char c; short x; } __Pyx_st_short;
typedef struct { char c; int x; } __Pyx_st_int;
typedef struct { char c; long x; } __Pyx_st_long;
typedef struct { char c; float x; } __Pyx_st_float;
typedef struct { char c; double x; } __Pyx_st_double;
typedef struct { char c; long double x; } __Pyx_st_longdouble;
typedef struct { char c; void *x; } __Pyx_st_void_p;
#ifdef HAVE_LONG_LONG
typedef struct { char c; PY_LONG_LONG x; } __Pyx_st_longlong;
#endif

static size_t __Pyx_BufFmt_TypeCharToAlignment(char ch, int is_complex) {
  switch (ch) {
    case '?': case 'c': case 'b': case 'B': return 1;
    case 'h': case 'H': return sizeof(__Pyx_st_short) - sizeof(short);
    case 'i': case 'I': return sizeof(__Pyx_st_int) - sizeof(int);
    case 'l': case 'L': return sizeof(__Pyx_st_long) - sizeof(long);
#ifdef HAVE_LONG_LONG
    case 'q': case 'Q': return sizeof(__Pyx_st_longlong) - sizeof(PY_LONG_LONG);
#endif
    case 'f': return sizeof(__Pyx_st_float) - sizeof(float);
    case 'd': return sizeof(__Pyx_st_double) - sizeof(double);
    case 'g': return sizeof(__Pyx_st_longdouble) - sizeof(long double);
    case 'P': case 'O': return sizeof(__Pyx_st_void_p) - sizeof(void*);
    default:
      __Pyx_BufFmt_RaiseUnexpectedChar(ch);
      return 0;
    }
}

static char __Pyx_BufFmt_TypeCharToGroup(char ch, int is_complex) {
  switch (ch) {
    case 'c': case 'b': case 'h': case 'i': case 'l': case 'q': return 'I';
    case 'B': case 'H': case 'I': case 'L': case 'Q': return 'U';
    case 'f': case 'd': case 'g': return (is_complex ? 'C' : 'R');
    case 'O': return 'O';
    case 'P': return 'P';
    default: {
      __Pyx_BufFmt_RaiseUnexpectedChar(ch);
      return 0;
    }
  }
}

static void __Pyx_BufFmt_RaiseExpected(__Pyx_BufFmt_Context* ctx) {
  if (ctx->head == NULL || ctx->head->field == &ctx->root) {
    const char* expected;
    const char* quote;
    if (ctx->head == NULL) {
      expected = "end";
      quote = "";
    } else {
      expected = ctx->head->field->type->name;
      quote = "'";
    }
    PyErr_Format(PyExc_ValueError,
                 "Buffer dtype mismatch, expected %s%s%s but got %s",
                 quote, expected, quote,
                 __Pyx_BufFmt_DescribeTypeChar(ctx->enc_type, ctx->is_complex));
  } else {
    __Pyx_StructField* field = ctx->head->field;
    __Pyx_StructField* parent = (ctx->head - 1)->field;
    PyErr_Format(PyExc_ValueError,
                 "Buffer dtype mismatch, expected '%s' but got %s in '%s.%s'",
                 field->type->name, __Pyx_BufFmt_DescribeTypeChar(ctx->enc_type, ctx->is_complex),
                 parent->type->name, field->name);
  }
}

static int __Pyx_BufFmt_ProcessTypeChunk(__Pyx_BufFmt_Context* ctx) {
  char group;
  size_t size, offset;
  if (ctx->enc_type == 0) return 0;
  group = __Pyx_BufFmt_TypeCharToGroup(ctx->enc_type, ctx->is_complex);
  do {
    __Pyx_StructField* field = ctx->head->field;
    __Pyx_TypeInfo* type = field->type;

    if (ctx->enc_packmode == '@' || ctx->enc_packmode == '^') {
      size = __Pyx_BufFmt_TypeCharToNativeSize(ctx->enc_type, ctx->is_complex);
    } else {
      size = __Pyx_BufFmt_TypeCharToStandardSize(ctx->enc_type, ctx->is_complex);
    }
    if (ctx->enc_packmode == '@') {
      size_t align_at = __Pyx_BufFmt_TypeCharToAlignment(ctx->enc_type, ctx->is_complex);
      size_t align_mod_offset;
      if (align_at == 0) return -1;
      align_mod_offset = ctx->fmt_offset % align_at;
      if (align_mod_offset > 0) ctx->fmt_offset += align_at - align_mod_offset;
    }

    if (type->size != size || type->typegroup != group) {
      if (type->typegroup == 'C' && type->fields != NULL) {
        /* special case -- treat as struct rather than complex number */
        size_t parent_offset = ctx->head->parent_offset + field->offset;
        ++ctx->head;
        ctx->head->field = type->fields;
        ctx->head->parent_offset = parent_offset;
        continue;
      }

      __Pyx_BufFmt_RaiseExpected(ctx);
      return -1;
    }

    offset = ctx->head->parent_offset + field->offset;
    if (ctx->fmt_offset != offset) {
      PyErr_Format(PyExc_ValueError,
                   "Buffer dtype mismatch; next field is at offset %"PY_FORMAT_SIZE_T"d but %"PY_FORMAT_SIZE_T"d expected",
                   (Py_ssize_t)ctx->fmt_offset, (Py_ssize_t)offset);
      return -1;
    }

    ctx->fmt_offset += size;

    --ctx->enc_count; /* Consume from buffer string */

    /* Done checking, move to next field, pushing or popping struct stack if needed */
    while (1) {
      if (field == &ctx->root) {
        ctx->head = NULL;
        if (ctx->enc_count != 0) {
          __Pyx_BufFmt_RaiseExpected(ctx);
          return -1;
        }
        break; /* breaks both loops as ctx->enc_count == 0 */
      }
      ctx->head->field = ++field;
      if (field->type == NULL) {
        --ctx->head;
        field = ctx->head->field;
        continue;
      } else if (field->type->typegroup == 'S') {
        size_t parent_offset = ctx->head->parent_offset + field->offset;
        if (field->type->fields->type == NULL) continue; /* empty struct */
        field = field->type->fields;
        ++ctx->head;
        ctx->head->field = field;
        ctx->head->parent_offset = parent_offset;
        break;
      } else {
        break;
      }
    }
  } while (ctx->enc_count);
  ctx->enc_type = 0;
  ctx->is_complex = 0;
  return 0;
}

static const char* __Pyx_BufFmt_CheckString(__Pyx_BufFmt_Context* ctx, const char* ts) {
  int got_Z = 0;
  while (1) {
    switch(*ts) {
      case 0:
        if (ctx->enc_type != 0 && ctx->head == NULL) {
          __Pyx_BufFmt_RaiseExpected(ctx);
          return NULL;
        }
        if (__Pyx_BufFmt_ProcessTypeChunk(ctx) == -1) return NULL;
        if (ctx->head != NULL) {
          __Pyx_BufFmt_RaiseExpected(ctx);
          return NULL;
        }
        return ts;
      case ' ':
      case 10:
      case 13:
        ++ts;
        break;
      case '<':
        if (!__Pyx_IsLittleEndian()) {
          PyErr_SetString(PyExc_ValueError, "Little-endian buffer not supported on big-endian compiler");
          return NULL;
        }
        ctx->new_packmode = '=';
        ++ts;
        break;
      case '>':
      case '!':
        if (__Pyx_IsLittleEndian()) {
          PyErr_SetString(PyExc_ValueError, "Big-endian buffer not supported on little-endian compiler");
          return NULL;
        }
        ctx->new_packmode = '=';
        ++ts;
        break;
      case '=':
      case '@':
      case '^':
        ctx->new_packmode = *ts++;
        break;
      case 'T': /* substruct */
        {
          const char* ts_after_sub;
          size_t i, struct_count = ctx->new_count;
          ctx->new_count = 1;
          ++ts;
          if (*ts != '{') {
            PyErr_SetString(PyExc_ValueError, "Buffer acquisition: Expected '{' after 'T'");
            return NULL;
          }
          ++ts;
          ts_after_sub = ts;
          for (i = 0; i != struct_count; ++i) {
            ts_after_sub = __Pyx_BufFmt_CheckString(ctx, ts);
            if (!ts_after_sub) return NULL;
          }
          ts = ts_after_sub;
        }
        break;
      case '}': /* end of substruct; either repeat or move on */
        ++ts;
        return ts;
      case 'x':
        if (__Pyx_BufFmt_ProcessTypeChunk(ctx) == -1) return NULL;
        ctx->fmt_offset += ctx->new_count;
        ctx->new_count = 1;
        ctx->enc_count = 0;
        ctx->enc_type = 0;
        ctx->enc_packmode = ctx->new_packmode;
        ++ts;
        break;
      case 'Z':
        got_Z = 1;
        ++ts;
        if (*ts != 'f' && *ts != 'd' && *ts != 'g') {
          __Pyx_BufFmt_RaiseUnexpectedChar('Z');
          return NULL;
        }        /* fall through */
      case 'c': case 'b': case 'B': case 'h': case 'H': case 'i': case 'I':
      case 'l': case 'L': case 'q': case 'Q':
      case 'f': case 'd': case 'g':
      case 'O':
        if (ctx->enc_type == *ts && got_Z == ctx->is_complex &&
            ctx->enc_packmode == ctx->new_packmode) {
          /* Continue pooling same type */
          ctx->enc_count += ctx->new_count;
        } else {
          /* New type */
          if (__Pyx_BufFmt_ProcessTypeChunk(ctx) == -1) return NULL;
          ctx->enc_count = ctx->new_count;
          ctx->enc_packmode = ctx->new_packmode;
          ctx->enc_type = *ts;
          ctx->is_complex = got_Z;
        }
        ++ts;
        ctx->new_count = 1;
        got_Z = 0;
        break;
      case ':':
        ++ts;
        while(*ts != ':') ++ts;
        ++ts;
        break;
      default:
        {
          int number = __Pyx_BufFmt_ParseNumber(&ts);
          if (number == -1) { /* First char was not a digit */
            PyErr_Format(PyExc_ValueError,
                         "Does not understand character buffer dtype format string ('%c')", *ts);
            return NULL;
          }
          ctx->new_count = (size_t)number; 
        }
    }
  }
}

static CYTHON_INLINE void __Pyx_ZeroBuffer(Py_buffer* buf) {
  buf->buf = NULL;
  buf->obj = NULL;
  buf->strides = __Pyx_zeros;
  buf->shape = __Pyx_zeros;
  buf->suboffsets = __Pyx_minusones;
}

static CYTHON_INLINE int __Pyx_GetBufferAndValidate(Py_buffer* buf, PyObject* obj, __Pyx_TypeInfo* dtype, int flags, int nd, int cast, __Pyx_BufFmt_StackElem* stack) {
  if (obj == Py_None || obj == NULL) {
    __Pyx_ZeroBuffer(buf);
    return 0;
  }
  buf->buf = NULL;
  if (__Pyx_GetBuffer(obj, buf, flags) == -1) goto fail;
  if (buf->ndim != nd) {
    PyErr_Format(PyExc_ValueError,
                 "Buffer has wrong number of dimensions (expected %d, got %d)",
                 nd, buf->ndim);
    goto fail;
  }
  if (!cast) {
    __Pyx_BufFmt_Context ctx;
    __Pyx_BufFmt_Init(&ctx, stack, dtype);
    if (!__Pyx_BufFmt_CheckString(&ctx, buf->format)) goto fail;
  }
  if ((unsigned)buf->itemsize != dtype->size) {
    PyErr_Format(PyExc_ValueError,
      "Item size of buffer (%"PY_FORMAT_SIZE_T"d byte%s) does not match size of '%s' (%"PY_FORMAT_SIZE_T"d byte%s)",
      buf->itemsize, (buf->itemsize > 1) ? "s" : "",
      dtype->name, (Py_ssize_t)dtype->size, (dtype->size > 1) ? "s" : "");
    goto fail;
  }
  if (buf->suboffsets == NULL) buf->suboffsets = __Pyx_minusones;
  return 0;
fail:;
  __Pyx_ZeroBuffer(buf);
  return -1;
}

static CYTHON_INLINE void __Pyx_SafeReleaseBuffer(Py_buffer* info) {
  if (info->buf == NULL) return;
  if (info->suboffsets == __Pyx_minusones) info->suboffsets = NULL;
  __Pyx_ReleaseBuffer(info);
}

static CYTHON_INLINE void __Pyx_ErrRestore(PyObject *type, PyObject *value, PyObject *tb) {
    PyObject *tmp_type, *tmp_value, *tmp_tb;
    PyThreadState *tstate = PyThreadState_GET();

    tmp_type = tstate->curexc_type;
    tmp_value = tstate->curexc_value;
    tmp_tb = tstate->curexc_traceback;
    tstate->curexc_type = type;
    tstate->curexc_value = value;
    tstate->curexc_traceback = tb;
    Py_XDECREF(tmp_type);
    Py_XDECREF(tmp_value);
    Py_XDECREF(tmp_tb);
}

static CYTHON_INLINE void __Pyx_ErrFetch(PyObject **type, PyObject **value, PyObject **tb) {
    PyThreadState *tstate = PyThreadState_GET();
    *type = tstate->curexc_type;
    *value = tstate->curexc_value;
    *tb = tstate->curexc_traceback;

    tstate->curexc_type = 0;
    tstate->curexc_value = 0;
    tstate->curexc_traceback = 0;
}


static void __Pyx_RaiseArgtupleInvalid(
    const char* func_name,
    int exact,
    Py_ssize_t num_min,
    Py_ssize_t num_max,
    Py_ssize_t num_found)
{
    Py_ssize_t num_expected;
    const char *more_or_less;

    if (num_found < num_min) {
        num_expected = num_min;
        more_or_less = "at least";
    } else {
        num_expected = num_max;
        more_or_less = "at most";
    }
    if (exact) {
        more_or_less = "exactly";
    }
    PyErr_Format(PyExc_TypeError,
                 "%s() takes %s %"PY_FORMAT_SIZE_T"d positional argument%s (%"PY_FORMAT_SIZE_T"d given)",
                 func_name, more_or_less, num_expected,
                 (num_expected == 1) ? "" : "s", num_found);
}

static void __Pyx_RaiseDoubleKeywordsError(
    const char* func_name,
    PyObject* kw_name)
{
    PyErr_Format(PyExc_TypeError,
        #if PY_MAJOR_VERSION >= 3
        "%s() got multiple values for keyword argument '%U'", func_name, kw_name);
        #else
        "%s() got multiple values for keyword argument '%s'", func_name,
        PyString_AS_STRING(kw_name));
        #endif
}

static int __Pyx_ParseOptionalKeywords(
    PyObject *kwds,
    PyObject **argnames[],
    PyObject *kwds2,
    PyObject *values[],
    Py_ssize_t num_pos_args,
    const char* function_name)
{
    PyObject *key = 0, *value = 0;
    Py_ssize_t pos = 0;
    PyObject*** name;
    PyObject*** first_kw_arg = argnames + num_pos_args;

    while (PyDict_Next(kwds, &pos, &key, &value)) {
        name = first_kw_arg;
        while (*name && (**name != key)) name++;
        if (*name) {
            values[name-argnames] = value;
        } else {
            #if PY_MAJOR_VERSION < 3
            if (unlikely(!PyString_CheckExact(key)) && unlikely(!PyString_Check(key))) {
            #else
            if (unlikely(!PyUnicode_CheckExact(key)) && unlikely(!PyUnicode_Check(key))) {
            #endif
                goto invalid_keyword_type;
            } else {
                for (name = first_kw_arg; *name; name++) {
                    #if PY_MAJOR_VERSION >= 3
                    if (PyUnicode_GET_SIZE(**name) == PyUnicode_GET_SIZE(key) &&
                        PyUnicode_Compare(**name, key) == 0) break;
                    #else
                    if (PyString_GET_SIZE(**name) == PyString_GET_SIZE(key) &&
                        _PyString_Eq(**name, key)) break;
                    #endif
                }
                if (*name) {
                    values[name-argnames] = value;
                } else {
                    /* unexpected keyword found */
                    for (name=argnames; name != first_kw_arg; name++) {
                        if (**name == key) goto arg_passed_twice;
                        #if PY_MAJOR_VERSION >= 3
                        if (PyUnicode_GET_SIZE(**name) == PyUnicode_GET_SIZE(key) &&
                            PyUnicode_Compare(**name, key) == 0) goto arg_passed_twice;
                        #else
                        if (PyString_GET_SIZE(**name) == PyString_GET_SIZE(key) &&
                            _PyString_Eq(**name, key)) goto arg_passed_twice;
                        #endif
                    }
                    if (kwds2) {
                        if (unlikely(PyDict_SetItem(kwds2, key, value))) goto bad;
                    } else {
                        goto invalid_keyword;
                    }
                }
            }
        }
    }
    return 0;
arg_passed_twice:
    __Pyx_RaiseDoubleKeywordsError(function_name, **name);
    goto bad;
invalid_keyword_type:
    PyErr_Format(PyExc_TypeError,
        "%s() keywords must be strings", function_name);
    goto bad;
invalid_keyword:
    PyErr_Format(PyExc_TypeError,
    #if PY_MAJOR_VERSION < 3
        "%s() got an unexpected keyword argument '%s'",
        function_name, PyString_AsString(key));
    #else
        "%s() got an unexpected keyword argument '%U'",
        function_name, key);
    #endif
bad:
    return -1;
}

#if PY_MAJOR_VERSION < 3
static void __Pyx_Raise(PyObject *type, PyObject *value, PyObject *tb, PyObject *cause) {
    /* cause is unused */
    Py_XINCREF(type);
    Py_XINCREF(value);
    Py_XINCREF(tb);
    /* First, check the traceback argument, replacing None with NULL. */
    if (tb == Py_None) {
        Py_DECREF(tb);
        tb = 0;
    }
    else if (tb != NULL && !PyTraceBack_Check(tb)) {
        PyErr_SetString(PyExc_TypeError,
            "raise: arg 3 must be a traceback or None");
        goto raise_error;
    }
    /* Next, replace a missing value with None */
    if (value == NULL) {
        value = Py_None;
        Py_INCREF(value);
    }
    #if PY_VERSION_HEX < 0x02050000
    if (!PyClass_Check(type))
    #else
    if (!PyType_Check(type))
    #endif
    {
        /* Raising an instance.  The value should be a dummy. */
        if (value != Py_None) {
            PyErr_SetString(PyExc_TypeError,
                "instance exception may not have a separate value");
            goto raise_error;
        }
        /* Normalize to raise <class>, <instance> */
        Py_DECREF(value);
        value = type;
        #if PY_VERSION_HEX < 0x02050000
            if (PyInstance_Check(type)) {
                type = (PyObject*) ((PyInstanceObject*)type)->in_class;
                Py_INCREF(type);
            }
            else {
                type = 0;
                PyErr_SetString(PyExc_TypeError,
                    "raise: exception must be an old-style class or instance");
                goto raise_error;
            }
        #else
            type = (PyObject*) Py_TYPE(type);
            Py_INCREF(type);
            if (!PyType_IsSubtype((PyTypeObject *)type, (PyTypeObject *)PyExc_BaseException)) {
                PyErr_SetString(PyExc_TypeError,
                    "raise: exception class must be a subclass of BaseException");
                goto raise_error;
            }
        #endif
    }

    __Pyx_ErrRestore(type, value, tb);
    return;
raise_error:
    Py_XDECREF(value);
    Py_XDECREF(type);
    Py_XDECREF(tb);
    return;
}

#else /* Python 3+ */

static void __Pyx_Raise(PyObject *type, PyObject *value, PyObject *tb, PyObject *cause) {
    if (tb == Py_None) {
        tb = 0;
    } else if (tb && !PyTraceBack_Check(tb)) {
        PyErr_SetString(PyExc_TypeError,
            "raise: arg 3 must be a traceback or None");
        goto bad;
    }
    if (value == Py_None)
        value = 0;

    if (PyExceptionInstance_Check(type)) {
        if (value) {
            PyErr_SetString(PyExc_TypeError,
                "instance exception may not have a separate value");
            goto bad;
        }
        value = type;
        type = (PyObject*) Py_TYPE(value);
    } else if (!PyExceptionClass_Check(type)) {
        PyErr_SetString(PyExc_TypeError,
            "raise: exception class must be a subclass of BaseException");
        goto bad;
    }

    if (cause) {
        PyObject *fixed_cause;
        if (PyExceptionClass_Check(cause)) {
            fixed_cause = PyObject_CallObject(cause, NULL);
            if (fixed_cause == NULL)
                goto bad;
        }
        else if (PyExceptionInstance_Check(cause)) {
            fixed_cause = cause;
            Py_INCREF(fixed_cause);
        }
        else {
            PyErr_SetString(PyExc_TypeError,
                            "exception causes must derive from "
                            "BaseException");
            goto bad;
        }
        if (!value) {
            value = PyObject_CallObject(type, NULL);
        }
        PyException_SetCause(value, fixed_cause);
    }

    PyErr_SetObject(type, value);

    if (tb) {
        PyThreadState *tstate = PyThreadState_GET();
        PyObject* tmp_tb = tstate->curexc_traceback;
        if (tb != tmp_tb) {
            Py_INCREF(tb);
            tstate->curexc_traceback = tb;
            Py_XDECREF(tmp_tb);
        }
    }

bad:
    return;
}
#endif

static CYTHON_INLINE void __Pyx_RaiseNeedMoreValuesError(Py_ssize_t index) {
    PyErr_Format(PyExc_ValueError,
                 "need more than %"PY_FORMAT_SIZE_T"d value%s to unpack",
                 index, (index == 1) ? "" : "s");
}

static CYTHON_INLINE void __Pyx_RaiseTooManyValuesError(Py_ssize_t expected) {
    PyErr_Format(PyExc_ValueError,
                 "too many values to unpack (expected %"PY_FORMAT_SIZE_T"d)", expected);
}

static CYTHON_INLINE void __Pyx_RaiseNoneNotIterableError(void) {
    PyErr_SetString(PyExc_TypeError, "'NoneType' object is not iterable");
}

static void __Pyx_UnpackTupleError(PyObject *t, Py_ssize_t index) {
    if (t == Py_None) {
      __Pyx_RaiseNoneNotIterableError();
    } else if (PyTuple_GET_SIZE(t) < index) {
      __Pyx_RaiseNeedMoreValuesError(PyTuple_GET_SIZE(t));
    } else {
      __Pyx_RaiseTooManyValuesError(index);
    }
}

#if PY_MAJOR_VERSION < 3
static int __Pyx_GetBuffer(PyObject *obj, Py_buffer *view, int flags) {
  #if PY_VERSION_HEX >= 0x02060000
  if (PyObject_CheckBuffer(obj)) return PyObject_GetBuffer(obj, view, flags);
  #endif
  if (PyObject_TypeCheck(obj, __pyx_ptype_5numpy_ndarray)) return __pyx_pf_5numpy_7ndarray___getbuffer__(obj, view, flags);
  else {
  PyErr_Format(PyExc_TypeError, "'%100s' does not have the buffer interface", Py_TYPE(obj)->tp_name);
  return -1;
    }
}

static void __Pyx_ReleaseBuffer(Py_buffer *view) {
  PyObject* obj = view->obj;
  if (obj) {
    #if PY_VERSION_HEX >= 0x02060000
    if (PyObject_CheckBuffer(obj)) {PyBuffer_Release(view); return;}
    #endif
    if (PyObject_TypeCheck(obj, __pyx_ptype_5numpy_ndarray)) __pyx_pf_5numpy_7ndarray_1__releasebuffer__(obj, view);
    Py_DECREF(obj);
    view->obj = NULL;
  }
}

#endif

static PyObject *__Pyx_Import(PyObject *name, PyObject *from_list, long level) {
    PyObject *py_import = 0;
    PyObject *empty_list = 0;
    PyObject *module = 0;
    PyObject *global_dict = 0;
    PyObject *empty_dict = 0;
    PyObject *list;
    py_import = __Pyx_GetAttrString(__pyx_b, "__import__");
    if (!py_import)
        goto bad;
    if (from_list)
        list = from_list;
    else {
        empty_list = PyList_New(0);
        if (!empty_list)
            goto bad;
        list = empty_list;
    }
    global_dict = PyModule_GetDict(__pyx_m);
    if (!global_dict)
        goto bad;
    empty_dict = PyDict_New();
    if (!empty_dict)
        goto bad;
    #if PY_VERSION_HEX >= 0x02050000
    {
        PyObject *py_level = PyInt_FromLong(level);
        if (!py_level)
            goto bad;
        module = PyObject_CallFunctionObjArgs(py_import,
            name, global_dict, empty_dict, list, py_level, NULL);
        Py_DECREF(py_level);
    }
    #else
    if (level>0) {
        PyErr_SetString(PyExc_RuntimeError, "Relative import is not supported for Python <=2.4.");
        goto bad;
    }
    module = PyObject_CallFunctionObjArgs(py_import,
        name, global_dict, empty_dict, list, NULL);
    #endif
bad:
    Py_XDECREF(empty_list);
    Py_XDECREF(py_import);
    Py_XDECREF(empty_dict);
    return module;
}

#if CYTHON_CCOMPLEX
  #ifdef __cplusplus
    static CYTHON_INLINE __pyx_t_float_complex __pyx_t_float_complex_from_parts(float x, float y) {
      return ::std::complex< float >(x, y);
    }
  #else
    static CYTHON_INLINE __pyx_t_float_complex __pyx_t_float_complex_from_parts(float x, float y) {
      return x + y*(__pyx_t_float_complex)_Complex_I;
    }
  #endif
#else
    static CYTHON_INLINE __pyx_t_float_complex __pyx_t_float_complex_from_parts(float x, float y) {
      __pyx_t_float_complex z;
      z.real = x;
      z.imag = y;
      return z;
    }
#endif

#if CYTHON_CCOMPLEX
#else
    static CYTHON_INLINE int __Pyx_c_eqf(__pyx_t_float_complex a, __pyx_t_float_complex b) {
       return (a.real == b.real) && (a.imag == b.imag);
    }
    static CYTHON_INLINE __pyx_t_float_complex __Pyx_c_sumf(__pyx_t_float_complex a, __pyx_t_float_complex b) {
        __pyx_t_float_complex z;
        z.real = a.real + b.real;
        z.imag = a.imag + b.imag;
        return z;
    }
    static CYTHON_INLINE __pyx_t_float_complex __Pyx_c_difff(__pyx_t_float_complex a, __pyx_t_float_complex b) {
        __pyx_t_float_complex z;
        z.real = a.real - b.real;
        z.imag = a.imag - b.imag;
        return z;
    }
    static CYTHON_INLINE __pyx_t_float_complex __Pyx_c_prodf(__pyx_t_float_complex a, __pyx_t_float_complex b) {
        __pyx_t_float_complex z;
        z.real = a.real * b.real - a.imag * b.imag;
        z.imag = a.real * b.imag + a.imag * b.real;
        return z;
    }
    static CYTHON_INLINE __pyx_t_float_complex __Pyx_c_quotf(__pyx_t_float_complex a, __pyx_t_float_complex b) {
        __pyx_t_float_complex z;
        float denom = b.real * b.real + b.imag * b.imag;
        z.real = (a.real * b.real + a.imag * b.imag) / denom;
        z.imag = (a.imag * b.real - a.real * b.imag) / denom;
        return z;
    }
    static CYTHON_INLINE __pyx_t_float_complex __Pyx_c_negf(__pyx_t_float_complex a) {
        __pyx_t_float_complex z;
        z.real = -a.real;
        z.imag = -a.imag;
        return z;
    }
    static CYTHON_INLINE int __Pyx_c_is_zerof(__pyx_t_float_complex a) {
       return (a.real == 0) && (a.imag == 0);
    }
    static CYTHON_INLINE __pyx_t_float_complex __Pyx_c_conjf(__pyx_t_float_complex a) {
        __pyx_t_float_complex z;
        z.real =  a.real;
        z.imag = -a.imag;
        return z;
    }
    #if 1
        static CYTHON_INLINE float __Pyx_c_absf(__pyx_t_float_complex z) {
          #if !defined(HAVE_HYPOT) || defined(_MSC_VER)
            return sqrtf(z.real*z.real + z.imag*z.imag);
          #else
            return hypotf(z.real, z.imag);
          #endif
        }
        static CYTHON_INLINE __pyx_t_float_complex __Pyx_c_powf(__pyx_t_float_complex a, __pyx_t_float_complex b) {
            __pyx_t_float_complex z;
            float r, lnr, theta, z_r, z_theta;
            if (b.imag == 0 && b.real == (int)b.real) {
                if (b.real < 0) {
                    float denom = a.real * a.real + a.imag * a.imag;
                    a.real = a.real / denom;
                    a.imag = -a.imag / denom;
                    b.real = -b.real;
                }
                switch ((int)b.real) {
                    case 0:
                        z.real = 1;
                        z.imag = 0;
                        return z;
                    case 1:
                        return a;
                    case 2:
                        z = __Pyx_c_prodf(a, a);
                        return __Pyx_c_prodf(a, a);
                    case 3:
                        z = __Pyx_c_prodf(a, a);
                        return __Pyx_c_prodf(z, a);
                    case 4:
                        z = __Pyx_c_prodf(a, a);
                        return __Pyx_c_prodf(z, z);
                }
            }
            if (a.imag == 0) {
                if (a.real == 0) {
                    return a;
                }
                r = a.real;
                theta = 0;
            } else {
                r = __Pyx_c_absf(a);
                theta = atan2f(a.imag, a.real);
            }
            lnr = logf(r);
            z_r = expf(lnr * b.real - theta * b.imag);
            z_theta = theta * b.real + lnr * b.imag;
            z.real = z_r * cosf(z_theta);
            z.imag = z_r * sinf(z_theta);
            return z;
        }
    #endif
#endif

#if CYTHON_CCOMPLEX
  #ifdef __cplusplus
    static CYTHON_INLINE __pyx_t_double_complex __pyx_t_double_complex_from_parts(double x, double y) {
      return ::std::complex< double >(x, y);
    }
  #else
    static CYTHON_INLINE __pyx_t_double_complex __pyx_t_double_complex_from_parts(double x, double y) {
      return x + y*(__pyx_t_double_complex)_Complex_I;
    }
  #endif
#else
    static CYTHON_INLINE __pyx_t_double_complex __pyx_t_double_complex_from_parts(double x, double y) {
      __pyx_t_double_complex z;
      z.real = x;
      z.imag = y;
      return z;
    }
#endif

#if CYTHON_CCOMPLEX
#else
    static CYTHON_INLINE int __Pyx_c_eq(__pyx_t_double_complex a, __pyx_t_double_complex b) {
       return (a.real == b.real) && (a.imag == b.imag);
    }
    static CYTHON_INLINE __pyx_t_double_complex __Pyx_c_sum(__pyx_t_double_complex a, __pyx_t_double_complex b) {
        __pyx_t_double_complex z;
        z.real = a.real + b.real;
        z.imag = a.imag + b.imag;
        return z;
    }
    static CYTHON_INLINE __pyx_t_double_complex __Pyx_c_diff(__pyx_t_double_complex a, __pyx_t_double_complex b) {
        __pyx_t_double_complex z;
        z.real = a.real - b.real;
        z.imag = a.imag - b.imag;
        return z;
    }
    static CYTHON_INLINE __pyx_t_double_complex __Pyx_c_prod(__pyx_t_double_complex a, __pyx_t_double_complex b) {
        __pyx_t_double_complex z;
        z.real = a.real * b.real - a.imag * b.imag;
        z.imag = a.real * b.imag + a.imag * b.real;
        return z;
    }
    static CYTHON_INLINE __pyx_t_double_complex __Pyx_c_quot(__pyx_t_double_complex a, __pyx_t_double_complex b) {
        __pyx_t_double_complex z;
        double denom = b.real * b.real + b.imag * b.imag;
        z.real = (a.real * b.real + a.imag * b.imag) / denom;
        z.imag = (a.imag * b.real - a.real * b.imag) / denom;
        return z;
    }
    static CYTHON_INLINE __pyx_t_double_complex __Pyx_c_neg(__pyx_t_double_complex a) {
        __pyx_t_double_complex z;
        z.real = -a.real;
        z.imag = -a.imag;
        return z;
    }
    static CYTHON_INLINE int __Pyx_c_is_zero(__pyx_t_double_complex a) {
       return (a.real == 0) && (a.imag == 0);
    }
    static CYTHON_INLINE __pyx_t_double_complex __Pyx_c_conj(__pyx_t_double_complex a) {
        __pyx_t_double_complex z;
        z.real =  a.real;
        z.imag = -a.imag;
        return z;
    }
    #if 1
        static CYTHON_INLINE double __Pyx_c_abs(__pyx_t_double_complex z) {
          #if !defined(HAVE_HYPOT) || defined(_MSC_VER)
            return sqrt(z.real*z.real + z.imag*z.imag);
          #else
            return hypot(z.real, z.imag);
          #endif
        }
        static CYTHON_INLINE __pyx_t_double_complex __Pyx_c_pow(__pyx_t_double_complex a, __pyx_t_double_complex b) {
            __pyx_t_double_complex z;
            double r, lnr, theta, z_r, z_theta;
            if (b.imag == 0 && b.real == (int)b.real) {
                if (b.real < 0) {
                    double denom = a.real * a.real + a.imag * a.imag;
                    a.real = a.real / denom;
                    a.imag = -a.imag / denom;
                    b.real = -b.real;
                }
                switch ((int)b.real) {
                    case 0:
                        z.real = 1;
                        z.imag = 0;
                        return z;
                    case 1:
                        return a;
                    case 2:
                        z = __Pyx_c_prod(a, a);
                        return __Pyx_c_prod(a, a);
                    case 3:
                        z = __Pyx_c_prod(a, a);
                        return __Pyx_c_prod(z, a);
                    case 4:
                        z = __Pyx_c_prod(a, a);
                        return __Pyx_c_prod(z, z);
                }
            }
            if (a.imag == 0) {
                if (a.real == 0) {
                    return a;
                }
                r = a.real;
                theta = 0;
            } else {
                r = __Pyx_c_abs(a);
                theta = atan2(a.imag, a.real);
            }
            lnr = log(r);
            z_r = exp(lnr * b.real - theta * b.imag);
            z_theta = theta * b.real + lnr * b.imag;
            z.real = z_r * cos(z_theta);
            z.imag = z_r * sin(z_theta);
            return z;
        }
    #endif
#endif

static CYTHON_INLINE unsigned char __Pyx_PyInt_AsUnsignedChar(PyObject* x) {
    const unsigned char neg_one = (unsigned char)-1, const_zero = 0;
    const int is_unsigned = neg_one > const_zero;
    if (sizeof(unsigned char) < sizeof(long)) {
        long val = __Pyx_PyInt_AsLong(x);
        if (unlikely(val != (long)(unsigned char)val)) {
            if (!unlikely(val == -1 && PyErr_Occurred())) {
                PyErr_SetString(PyExc_OverflowError,
                    (is_unsigned && unlikely(val < 0)) ?
                    "can't convert negative value to unsigned char" :
                    "value too large to convert to unsigned char");
            }
            return (unsigned char)-1;
        }
        return (unsigned char)val;
    }
    return (unsigned char)__Pyx_PyInt_AsUnsignedLong(x);
}

static CYTHON_INLINE unsigned short __Pyx_PyInt_AsUnsignedShort(PyObject* x) {
    const unsigned short neg_one = (unsigned short)-1, const_zero = 0;
    const int is_unsigned = neg_one > const_zero;
    if (sizeof(unsigned short) < sizeof(long)) {
        long val = __Pyx_PyInt_AsLong(x);
        if (unlikely(val != (long)(unsigned short)val)) {
            if (!unlikely(val == -1 && PyErr_Occurred())) {
                PyErr_SetString(PyExc_OverflowError,
                    (is_unsigned && unlikely(val < 0)) ?
                    "can't convert negative value to unsigned short" :
                    "value too large to convert to unsigned short");
            }
            return (unsigned short)-1;
        }
        return (unsigned short)val;
    }
    return (unsigned short)__Pyx_PyInt_AsUnsignedLong(x);
}

static CYTHON_INLINE unsigned int __Pyx_PyInt_AsUnsignedInt(PyObject* x) {
    const unsigned int neg_one = (unsigned int)-1, const_zero = 0;
    const int is_unsigned = neg_one > const_zero;
    if (sizeof(unsigned int) < sizeof(long)) {
        long val = __Pyx_PyInt_AsLong(x);
        if (unlikely(val != (long)(unsigned int)val)) {
            if (!unlikely(val == -1 && PyErr_Occurred())) {
                PyErr_SetString(PyExc_OverflowError,
                    (is_unsigned && unlikely(val < 0)) ?
                    "can't convert negative value to unsigned int" :
                    "value too large to convert to unsigned int");
            }
            return (unsigned int)-1;
        }
        return (unsigned int)val;
    }
    return (unsigned int)__Pyx_PyInt_AsUnsignedLong(x);
}

static CYTHON_INLINE char __Pyx_PyInt_AsChar(PyObject* x) {
    const char neg_one = (char)-1, const_zero = 0;
    const int is_unsigned = neg_one > const_zero;
    if (sizeof(char) < sizeof(long)) {
        long val = __Pyx_PyInt_AsLong(x);
        if (unlikely(val != (long)(char)val)) {
            if (!unlikely(val == -1 && PyErr_Occurred())) {
                PyErr_SetString(PyExc_OverflowError,
                    (is_unsigned && unlikely(val < 0)) ?
                    "can't convert negative value to char" :
                    "value too large to convert to char");
            }
            return (char)-1;
        }
        return (char)val;
    }
    return (char)__Pyx_PyInt_AsLong(x);
}

static CYTHON_INLINE short __Pyx_PyInt_AsShort(PyObject* x) {
    const short neg_one = (short)-1, const_zero = 0;
    const int is_unsigned = neg_one > const_zero;
    if (sizeof(short) < sizeof(long)) {
        long val = __Pyx_PyInt_AsLong(x);
        if (unlikely(val != (long)(short)val)) {
            if (!unlikely(val == -1 && PyErr_Occurred())) {
                PyErr_SetString(PyExc_OverflowError,
                    (is_unsigned && unlikely(val < 0)) ?
                    "can't convert negative value to short" :
                    "value too large to convert to short");
            }
            return (short)-1;
        }
        return (short)val;
    }
    return (short)__Pyx_PyInt_AsLong(x);
}

static CYTHON_INLINE int __Pyx_PyInt_AsInt(PyObject* x) {
    const int neg_one = (int)-1, const_zero = 0;
    const int is_unsigned = neg_one > const_zero;
    if (sizeof(int) < sizeof(long)) {
        long val = __Pyx_PyInt_AsLong(x);
        if (unlikely(val != (long)(int)val)) {
            if (!unlikely(val == -1 && PyErr_Occurred())) {
                PyErr_SetString(PyExc_OverflowError,
                    (is_unsigned && unlikely(val < 0)) ?
                    "can't convert negative value to int" :
                    "value too large to convert to int");
            }
            return (int)-1;
        }
        return (int)val;
    }
    return (int)__Pyx_PyInt_AsLong(x);
}

static CYTHON_INLINE signed char __Pyx_PyInt_AsSignedChar(PyObject* x) {
    const signed char neg_one = (signed char)-1, const_zero = 0;
    const int is_unsigned = neg_one > const_zero;
    if (sizeof(signed char) < sizeof(long)) {
        long val = __Pyx_PyInt_AsLong(x);
        if (unlikely(val != (long)(signed char)val)) {
            if (!unlikely(val == -1 && PyErr_Occurred())) {
                PyErr_SetString(PyExc_OverflowError,
                    (is_unsigned && unlikely(val < 0)) ?
                    "can't convert negative value to signed char" :
                    "value too large to convert to signed char");
            }
            return (signed char)-1;
        }
        return (signed char)val;
    }
    return (signed char)__Pyx_PyInt_AsSignedLong(x);
}

static CYTHON_INLINE signed short __Pyx_PyInt_AsSignedShort(PyObject* x) {
    const signed short neg_one = (signed short)-1, const_zero = 0;
    const int is_unsigned = neg_one > const_zero;
    if (sizeof(signed short) < sizeof(long)) {
        long val = __Pyx_PyInt_AsLong(x);
        if (unlikely(val != (long)(signed short)val)) {
            if (!unlikely(val == -1 && PyErr_Occurred())) {
                PyErr_SetString(PyExc_OverflowError,
                    (is_unsigned && unlikely(val < 0)) ?
                    "can't convert negative value to signed short" :
                    "value too large to convert to signed short");
            }
            return (signed short)-1;
        }
        return (signed short)val;
    }
    return (signed short)__Pyx_PyInt_AsSignedLong(x);
}

static CYTHON_INLINE signed int __Pyx_PyInt_AsSignedInt(PyObject* x) {
    const signed int neg_one = (signed int)-1, const_zero = 0;
    const int is_unsigned = neg_one > const_zero;
    if (sizeof(signed int) < sizeof(long)) {
        long val = __Pyx_PyInt_AsLong(x);
        if (unlikely(val != (long)(signed int)val)) {
            if (!unlikely(val == -1 && PyErr_Occurred())) {
                PyErr_SetString(PyExc_OverflowError,
                    (is_unsigned && unlikely(val < 0)) ?
                    "can't convert negative value to signed int" :
                    "value too large to convert to signed int");
            }
            return (signed int)-1;
        }
        return (signed int)val;
    }
    return (signed int)__Pyx_PyInt_AsSignedLong(x);
}

static CYTHON_INLINE int __Pyx_PyInt_AsLongDouble(PyObject* x) {
    const int neg_one = (int)-1, const_zero = 0;
    const int is_unsigned = neg_one > const_zero;
    if (sizeof(int) < sizeof(long)) {
        long val = __Pyx_PyInt_AsLong(x);
        if (unlikely(val != (long)(int)val)) {
            if (!unlikely(val == -1 && PyErr_Occurred())) {
                PyErr_SetString(PyExc_OverflowError,
                    (is_unsigned && unlikely(val < 0)) ?
                    "can't convert negative value to int" :
                    "value too large to convert to int");
            }
            return (int)-1;
        }
        return (int)val;
    }
    return (int)__Pyx_PyInt_AsLong(x);
}

static CYTHON_INLINE unsigned long __Pyx_PyInt_AsUnsignedLong(PyObject* x) {
    const unsigned long neg_one = (unsigned long)-1, const_zero = 0;
    const int is_unsigned = neg_one > const_zero;
#if PY_VERSION_HEX < 0x03000000
    if (likely(PyInt_Check(x))) {
        long val = PyInt_AS_LONG(x);
        if (is_unsigned && unlikely(val < 0)) {
            PyErr_SetString(PyExc_OverflowError,
                            "can't convert negative value to unsigned long");
            return (unsigned long)-1;
        }
        return (unsigned long)val;
    } else
#endif
    if (likely(PyLong_Check(x))) {
        if (is_unsigned) {
            if (unlikely(Py_SIZE(x) < 0)) {
                PyErr_SetString(PyExc_OverflowError,
                                "can't convert negative value to unsigned long");
                return (unsigned long)-1;
            }
            return (unsigned long)PyLong_AsUnsignedLong(x);
        } else {
            return (unsigned long)PyLong_AsLong(x);
        }
    } else {
        unsigned long val;
        PyObject *tmp = __Pyx_PyNumber_Int(x);
        if (!tmp) return (unsigned long)-1;
        val = __Pyx_PyInt_AsUnsignedLong(tmp);
        Py_DECREF(tmp);
        return val;
    }
}

static CYTHON_INLINE unsigned PY_LONG_LONG __Pyx_PyInt_AsUnsignedLongLong(PyObject* x) {
    const unsigned PY_LONG_LONG neg_one = (unsigned PY_LONG_LONG)-1, const_zero = 0;
    const int is_unsigned = neg_one > const_zero;
#if PY_VERSION_HEX < 0x03000000
    if (likely(PyInt_Check(x))) {
        long val = PyInt_AS_LONG(x);
        if (is_unsigned && unlikely(val < 0)) {
            PyErr_SetString(PyExc_OverflowError,
                            "can't convert negative value to unsigned PY_LONG_LONG");
            return (unsigned PY_LONG_LONG)-1;
        }
        return (unsigned PY_LONG_LONG)val;
    } else
#endif
    if (likely(PyLong_Check(x))) {
        if (is_unsigned) {
            if (unlikely(Py_SIZE(x) < 0)) {
                PyErr_SetString(PyExc_OverflowError,
                                "can't convert negative value to unsigned PY_LONG_LONG");
                return (unsigned PY_LONG_LONG)-1;
            }
            return (unsigned PY_LONG_LONG)PyLong_AsUnsignedLongLong(x);
        } else {
            return (unsigned PY_LONG_LONG)PyLong_AsLongLong(x);
        }
    } else {
        unsigned PY_LONG_LONG val;
        PyObject *tmp = __Pyx_PyNumber_Int(x);
        if (!tmp) return (unsigned PY_LONG_LONG)-1;
        val = __Pyx_PyInt_AsUnsignedLongLong(tmp);
        Py_DECREF(tmp);
        return val;
    }
}

static CYTHON_INLINE long __Pyx_PyInt_AsLong(PyObject* x) {
    const long neg_one = (long)-1, const_zero = 0;
    const int is_unsigned = neg_one > const_zero;
#if PY_VERSION_HEX < 0x03000000
    if (likely(PyInt_Check(x))) {
        long val = PyInt_AS_LONG(x);
        if (is_unsigned && unlikely(val < 0)) {
            PyErr_SetString(PyExc_OverflowError,
                            "can't convert negative value to long");
            return (long)-1;
        }
        return (long)val;
    } else
#endif
    if (likely(PyLong_Check(x))) {
        if (is_unsigned) {
            if (unlikely(Py_SIZE(x) < 0)) {
                PyErr_SetString(PyExc_OverflowError,
                                "can't convert negative value to long");
                return (long)-1;
            }
            return (long)PyLong_AsUnsignedLong(x);
        } else {
            return (long)PyLong_AsLong(x);
        }
    } else {
        long val;
        PyObject *tmp = __Pyx_PyNumber_Int(x);
        if (!tmp) return (long)-1;
        val = __Pyx_PyInt_AsLong(tmp);
        Py_DECREF(tmp);
        return val;
    }
}

static CYTHON_INLINE PY_LONG_LONG __Pyx_PyInt_AsLongLong(PyObject* x) {
    const PY_LONG_LONG neg_one = (PY_LONG_LONG)-1, const_zero = 0;
    const int is_unsigned = neg_one > const_zero;
#if PY_VERSION_HEX < 0x03000000
    if (likely(PyInt_Check(x))) {
        long val = PyInt_AS_LONG(x);
        if (is_unsigned && unlikely(val < 0)) {
            PyErr_SetString(PyExc_OverflowError,
                            "can't convert negative value to PY_LONG_LONG");
            return (PY_LONG_LONG)-1;
        }
        return (PY_LONG_LONG)val;
    } else
#endif
    if (likely(PyLong_Check(x))) {
        if (is_unsigned) {
            if (unlikely(Py_SIZE(x) < 0)) {
                PyErr_SetString(PyExc_OverflowError,
                                "can't convert negative value to PY_LONG_LONG");
                return (PY_LONG_LONG)-1;
            }
            return (PY_LONG_LONG)PyLong_AsUnsignedLongLong(x);
        } else {
            return (PY_LONG_LONG)PyLong_AsLongLong(x);
        }
    } else {
        PY_LONG_LONG val;
        PyObject *tmp = __Pyx_PyNumber_Int(x);
        if (!tmp) return (PY_LONG_LONG)-1;
        val = __Pyx_PyInt_AsLongLong(tmp);
        Py_DECREF(tmp);
        return val;
    }
}

static CYTHON_INLINE signed long __Pyx_PyInt_AsSignedLong(PyObject* x) {
    const signed long neg_one = (signed long)-1, const_zero = 0;
    const int is_unsigned = neg_one > const_zero;
#if PY_VERSION_HEX < 0x03000000
    if (likely(PyInt_Check(x))) {
        long val = PyInt_AS_LONG(x);
        if (is_unsigned && unlikely(val < 0)) {
            PyErr_SetString(PyExc_OverflowError,
                            "can't convert negative value to signed long");
            return (signed long)-1;
        }
        return (signed long)val;
    } else
#endif
    if (likely(PyLong_Check(x))) {
        if (is_unsigned) {
            if (unlikely(Py_SIZE(x) < 0)) {
                PyErr_SetString(PyExc_OverflowError,
                                "can't convert negative value to signed long");
                return (signed long)-1;
            }
            return (signed long)PyLong_AsUnsignedLong(x);
        } else {
            return (signed long)PyLong_AsLong(x);
        }
    } else {
        signed long val;
        PyObject *tmp = __Pyx_PyNumber_Int(x);
        if (!tmp) return (signed long)-1;
        val = __Pyx_PyInt_AsSignedLong(tmp);
        Py_DECREF(tmp);
        return val;
    }
}

static CYTHON_INLINE signed PY_LONG_LONG __Pyx_PyInt_AsSignedLongLong(PyObject* x) {
    const signed PY_LONG_LONG neg_one = (signed PY_LONG_LONG)-1, const_zero = 0;
    const int is_unsigned = neg_one > const_zero;
#if PY_VERSION_HEX < 0x03000000
    if (likely(PyInt_Check(x))) {
        long val = PyInt_AS_LONG(x);
        if (is_unsigned && unlikely(val < 0)) {
            PyErr_SetString(PyExc_OverflowError,
                            "can't convert negative value to signed PY_LONG_LONG");
            return (signed PY_LONG_LONG)-1;
        }
        return (signed PY_LONG_LONG)val;
    } else
#endif
    if (likely(PyLong_Check(x))) {
        if (is_unsigned) {
            if (unlikely(Py_SIZE(x) < 0)) {
                PyErr_SetString(PyExc_OverflowError,
                                "can't convert negative value to signed PY_LONG_LONG");
                return (signed PY_LONG_LONG)-1;
            }
            return (signed PY_LONG_LONG)PyLong_AsUnsignedLongLong(x);
        } else {
            return (signed PY_LONG_LONG)PyLong_AsLongLong(x);
        }
    } else {
        signed PY_LONG_LONG val;
        PyObject *tmp = __Pyx_PyNumber_Int(x);
        if (!tmp) return (signed PY_LONG_LONG)-1;
        val = __Pyx_PyInt_AsSignedLongLong(tmp);
        Py_DECREF(tmp);
        return val;
    }
}

static void __Pyx_WriteUnraisable(const char *name, int clineno,
                                  int lineno, const char *filename) {
    PyObject *old_exc, *old_val, *old_tb;
    PyObject *ctx;
    __Pyx_ErrFetch(&old_exc, &old_val, &old_tb);
    #if PY_MAJOR_VERSION < 3
    ctx = PyString_FromString(name);
    #else
    ctx = PyUnicode_FromString(name);
    #endif
    __Pyx_ErrRestore(old_exc, old_val, old_tb);
    if (!ctx) {
        PyErr_WriteUnraisable(Py_None);
    } else {
        PyErr_WriteUnraisable(ctx);
        Py_DECREF(ctx);
    }
}

static int __Pyx_check_binary_version(void) {
    char ctversion[4], rtversion[4];
    PyOS_snprintf(ctversion, 4, "%d.%d", PY_MAJOR_VERSION, PY_MINOR_VERSION);
    PyOS_snprintf(rtversion, 4, "%s", Py_GetVersion());
    if (ctversion[0] != rtversion[0] || ctversion[2] != rtversion[2]) {
        char message[200];
        PyOS_snprintf(message, sizeof(message),
                      "compiletime version %s of module '%.100s' "
                      "does not match runtime version %s",
                      ctversion, __Pyx_MODULE_NAME, rtversion);
        #if PY_VERSION_HEX < 0x02050000
        return PyErr_Warn(NULL, message);
        #else
        return PyErr_WarnEx(NULL, message, 1);
        #endif
    }
    return 0;
}

#ifndef __PYX_HAVE_RT_ImportType
#define __PYX_HAVE_RT_ImportType
static PyTypeObject *__Pyx_ImportType(const char *module_name, const char *class_name,
    size_t size, int strict)
{
    PyObject *py_module = 0;
    PyObject *result = 0;
    PyObject *py_name = 0;
    char warning[200];

    py_module = __Pyx_ImportModule(module_name);
    if (!py_module)
        goto bad;
    #if PY_MAJOR_VERSION < 3
    py_name = PyString_FromString(class_name);
    #else
    py_name = PyUnicode_FromString(class_name);
    #endif
    if (!py_name)
        goto bad;
    result = PyObject_GetAttr(py_module, py_name);
    Py_DECREF(py_name);
    py_name = 0;
    Py_DECREF(py_module);
    py_module = 0;
    if (!result)
        goto bad;
    if (!PyType_Check(result)) {
        PyErr_Format(PyExc_TypeError,
            "%s.%s is not a type object",
            module_name, class_name);
        goto bad;
    }
    if (!strict && ((PyTypeObject *)result)->tp_basicsize > (Py_ssize_t)size) {
        PyOS_snprintf(warning, sizeof(warning),
            "%s.%s size changed, may indicate binary incompatibility",
            module_name, class_name);
        #if PY_VERSION_HEX < 0x02050000
        if (PyErr_Warn(NULL, warning) < 0) goto bad;
        #else
        if (PyErr_WarnEx(NULL, warning, 0) < 0) goto bad;
        #endif
    }
    else if (((PyTypeObject *)result)->tp_basicsize != (Py_ssize_t)size) {
        PyErr_Format(PyExc_ValueError,
            "%s.%s has the wrong size, try recompiling",
            module_name, class_name);
        goto bad;
    }
    return (PyTypeObject *)result;
bad:
    Py_XDECREF(py_module);
    Py_XDECREF(result);
    return NULL;
}
#endif

#ifndef __PYX_HAVE_RT_ImportModule
#define __PYX_HAVE_RT_ImportModule
static PyObject *__Pyx_ImportModule(const char *name) {
    PyObject *py_name = 0;
    PyObject *py_module = 0;

    #if PY_MAJOR_VERSION < 3
    py_name = PyString_FromString(name);
    #else
    py_name = PyUnicode_FromString(name);
    #endif
    if (!py_name)
        goto bad;
    py_module = PyImport_Import(py_name);
    Py_DECREF(py_name);
    return py_module;
bad:
    Py_XDECREF(py_name);
    return 0;
}
#endif

#include "compile.h"
#include "frameobject.h"
#include "traceback.h"

static void __Pyx_AddTraceback(const char *funcname, int __pyx_clineno,
                               int __pyx_lineno, const char *__pyx_filename) {
    PyObject *py_srcfile = 0;
    PyObject *py_funcname = 0;
    PyObject *py_globals = 0;
    PyCodeObject *py_code = 0;
    PyFrameObject *py_frame = 0;

    #if PY_MAJOR_VERSION < 3
    py_srcfile = PyString_FromString(__pyx_filename);
    #else
    py_srcfile = PyUnicode_FromString(__pyx_filename);
    #endif
    if (!py_srcfile) goto bad;
    if (__pyx_clineno) {
        #if PY_MAJOR_VERSION < 3
        py_funcname = PyString_FromFormat( "%s (%s:%d)", funcname, __pyx_cfilenm, __pyx_clineno);
        #else
        py_funcname = PyUnicode_FromFormat( "%s (%s:%d)", funcname, __pyx_cfilenm, __pyx_clineno);
        #endif
    }
    else {
        #if PY_MAJOR_VERSION < 3
        py_funcname = PyString_FromString(funcname);
        #else
        py_funcname = PyUnicode_FromString(funcname);
        #endif
    }
    if (!py_funcname) goto bad;
    py_globals = PyModule_GetDict(__pyx_m);
    if (!py_globals) goto bad;
    py_code = PyCode_New(
        0,            /*int argcount,*/
        #if PY_MAJOR_VERSION >= 3
        0,            /*int kwonlyargcount,*/
        #endif
        0,            /*int nlocals,*/
        0,            /*int stacksize,*/
        0,            /*int flags,*/
        __pyx_empty_bytes, /*PyObject *code,*/
        __pyx_empty_tuple,  /*PyObject *consts,*/
        __pyx_empty_tuple,  /*PyObject *names,*/
        __pyx_empty_tuple,  /*PyObject *varnames,*/
        __pyx_empty_tuple,  /*PyObject *freevars,*/
        __pyx_empty_tuple,  /*PyObject *cellvars,*/
        py_srcfile,   /*PyObject *filename,*/
        py_funcname,  /*PyObject *name,*/
        __pyx_lineno,   /*int firstlineno,*/
        __pyx_empty_bytes  /*PyObject *lnotab*/
    );
    if (!py_code) goto bad;
    py_frame = PyFrame_New(
        PyThreadState_GET(), /*PyThreadState *tstate,*/
        py_code,             /*PyCodeObject *code,*/
        py_globals,          /*PyObject *globals,*/
        0                    /*PyObject *locals*/
    );
    if (!py_frame) goto bad;
    py_frame->f_lineno = __pyx_lineno;
    PyTraceBack_Here(py_frame);
bad:
    Py_XDECREF(py_srcfile);
    Py_XDECREF(py_funcname);
    Py_XDECREF(py_code);
    Py_XDECREF(py_frame);
}

static int __Pyx_InitStrings(__Pyx_StringTabEntry *t) {
    while (t->p) {
        #if PY_MAJOR_VERSION < 3
        if (t->is_unicode) {
            *t->p = PyUnicode_DecodeUTF8(t->s, t->n - 1, NULL);
        } else if (t->intern) {
            *t->p = PyString_InternFromString(t->s);
        } else {
            *t->p = PyString_FromStringAndSize(t->s, t->n - 1);
        }
        #else  /* Python 3+ has unicode identifiers */
        if (t->is_unicode | t->is_str) {
            if (t->intern) {
                *t->p = PyUnicode_InternFromString(t->s);
            } else if (t->encoding) {
                *t->p = PyUnicode_Decode(t->s, t->n - 1, t->encoding, NULL);
            } else {
                *t->p = PyUnicode_FromStringAndSize(t->s, t->n - 1);
            }
        } else {
            *t->p = PyBytes_FromStringAndSize(t->s, t->n - 1);
        }
        #endif
        if (!*t->p)
            return -1;
        ++t;
    }
    return 0;
}

/* Type Conversion Functions */

static CYTHON_INLINE int __Pyx_PyObject_IsTrue(PyObject* x) {
   int is_true = x == Py_True;
   if (is_true | (x == Py_False) | (x == Py_None)) return is_true;
   else return PyObject_IsTrue(x);
}

static CYTHON_INLINE PyObject* __Pyx_PyNumber_Int(PyObject* x) {
  PyNumberMethods *m;
  const char *name = NULL;
  PyObject *res = NULL;
#if PY_VERSION_HEX < 0x03000000
  if (PyInt_Check(x) || PyLong_Check(x))
#else
  if (PyLong_Check(x))
#endif
    return Py_INCREF(x), x;
  m = Py_TYPE(x)->tp_as_number;
#if PY_VERSION_HEX < 0x03000000
  if (m && m->nb_int) {
    name = "int";
    res = PyNumber_Int(x);
  }
  else if (m && m->nb_long) {
    name = "long";
    res = PyNumber_Long(x);
  }
#else
  if (m && m->nb_int) {
    name = "int";
    res = PyNumber_Long(x);
  }
#endif
  if (res) {
#if PY_VERSION_HEX < 0x03000000
    if (!PyInt_Check(res) && !PyLong_Check(res)) {
#else
    if (!PyLong_Check(res)) {
#endif
      PyErr_Format(PyExc_TypeError,
                   "__%s__ returned non-%s (type %.200s)",
                   name, name, Py_TYPE(res)->tp_name);
      Py_DECREF(res);
      return NULL;
    }
  }
  else if (!PyErr_Occurred()) {
    PyErr_SetString(PyExc_TypeError,
                    "an integer is required");
  }
  return res;
}

static CYTHON_INLINE Py_ssize_t __Pyx_PyIndex_AsSsize_t(PyObject* b) {
  Py_ssize_t ival;
  PyObject* x = PyNumber_Index(b);
  if (!x) return -1;
  ival = PyInt_AsSsize_t(x);
  Py_DECREF(x);
  return ival;
}

static CYTHON_INLINE PyObject * __Pyx_PyInt_FromSize_t(size_t ival) {
#if PY_VERSION_HEX < 0x02050000
   if (ival <= LONG_MAX)
       return PyInt_FromLong((long)ival);
   else {
       unsigned char *bytes = (unsigned char *) &ival;
       int one = 1; int little = (int)*(unsigned char*)&one;
       return _PyLong_FromByteArray(bytes, sizeof(size_t), little, 0);
   }
#else
   return PyInt_FromSize_t(ival);
#endif
}

static CYTHON_INLINE size_t __Pyx_PyInt_AsSize_t(PyObject* x) {
   unsigned PY_LONG_LONG val = __Pyx_PyInt_AsUnsignedLongLong(x);
   if (unlikely(val == (unsigned PY_LONG_LONG)-1 && PyErr_Occurred())) {
       return (size_t)-1;
   } else if (unlikely(val != (unsigned PY_LONG_LONG)(size_t)val)) {
       PyErr_SetString(PyExc_OverflowError,
                       "value too large to convert to size_t");
       return (size_t)-1;
   }
   return (size_t)val;
}


#endif /* Py_PYTHON_H */
