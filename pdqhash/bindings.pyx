# distutils: language = c++
# cython: language_level=3
# cython: language=c++

import numpy as np
cimport numpy as cnp

cnp.import_array()

cdef extern from "ThreatExchange/pdq/cpp/common/pdqhashtypes.h" namespace "facebook::pdq::hashing":
    cdef struct Hash256:
        unsigned short w[16]

cdef extern from "ThreatExchange/pdq/cpp/hashing/pdqhashing.h" namespace "facebook::pdq::hashing":
    void pdqHash256FromFloatLuma(
        float* fullBuffer1,
        float* fullBuffer2,
        int numRows,
        int numCols,
        float buffer64x64[64][64],
        float buffer16x64[16][64],
        float buffer16x16[16][16],
        Hash256& hash_value,
        int& quality
    )

cdef extern from "ThreatExchange/pdq/cpp/hashing/pdqhashing.h" namespace "facebook::pdq::hashing":
    void pdqFloat256FromFloatLuma(
        float* fullBuffer1,
        float* fullBuffer2,
        int numRows,
        int numCols,
        float buffer64x64[64][64],
        float buffer16x64[16][64],
        float output_buffer16x16[16][16],
        int& quality
    )

cdef extern from "ThreatExchange/pdq/cpp/hashing/pdqhashing.h" namespace "facebook::pdq::hashing":
    bint pdqDihedralHash256esFromFloatLuma(
        float* fullBuffer1,
        float* fullBuffer2,
        int numRows,
        int numCols,
        float buffer64x64[64][64],
        float buffer16x64[16][64],
        float buffer16x16[16][16],
        float buffer16x16Aux[16][16],
        Hash256* hashptrOriginal,
        Hash256* hashptrRotate90,
        Hash256* hashptrRotate180,
        Hash256* hashptrRotate270,
        Hash256* hashptrFlipX,
        Hash256* hashptrFlipY,
        Hash256* hashptrFlipPlus1,
        Hash256* hashptrFlipMinus1,
        int& quality
    )

cdef extern from "ThreatExchange/pdq/cpp/downscaling/downscaling.h" namespace "facebook::pdq::downscaling":
    int computeJaroszFilterWindowSize(int oldDimension, int newDimension)

cdef extern from "ThreatExchange/pdq/cpp/hashing/torben.h" namespace "facebook::pdq::hashing":
    float torben(float m[], int n)

cdef inline object hash_to_vector(unsigned short[16] w):
    cdef int k
    return np.array([ ((w[(k & 255) >> 4] >> (k & 15)) & 1) for k in range(256) ], dtype=np.uint8)[::-1]

def hash_to_vector_py(hash_words):
    cdef unsigned short w[16]
    if len(hash_words) != 16:
        raise ValueError("Expected 16 uint16 words")
    for i in range(16):
        w[i] = <unsigned short> int(hash_words[i])
    return hash_to_vector(w)

def compute(cnp.ndarray[cnp.uint8_t, ndim=3] image):
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be H x W x 3 uint8")

    cdef cnp.ndarray[cnp.float32_t, ndim=2] gray = (
        image[:, :, 0]*0.299 + image[:, :, 1]*0.587 + image[:, :, 2]*0.114
    ).astype(np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim=2] placeholder = np.zeros_like(gray)

    cdef Hash256 hash_value
    cdef int quality = 0     # initialize to silence "referenced before assignment"
    cdef int numRows = <int> gray.shape[0]
    cdef int numCols = <int> gray.shape[1]
    cdef float buffer64x64[64][64]
    cdef float buffer16x64[16][64]
    cdef float buffer16x16[16][16]
    cdef float* fullBuffer1 = &gray[0, 0]
    cdef float* fullBuffer2 = &placeholder[0, 0]

    pdqHash256FromFloatLuma(
        fullBuffer1,
        fullBuffer2,
        numRows,
        numCols,
        buffer64x64,
        buffer16x64,
        buffer16x16,
        hash_value,
        quality
    )

    return hash_to_vector(hash_value.w), quality

def compute_float(cnp.ndarray[cnp.uint8_t, ndim=3] image):
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be H x W x 3 uint8")

    cdef cnp.ndarray[cnp.float32_t, ndim=2] gray = (
        image[:, :, 0]*0.299 + image[:, :, 1]*0.587 + image[:, :, 2]*0.114
    ).astype(np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim=2] placeholder = np.zeros_like(gray)

    cdef int quality = 0
    cdef int numRows = <int> gray.shape[0]
    cdef int numCols = <int> gray.shape[1]
    cdef float buffer64x64[64][64]
    cdef float buffer16x64[16][64]
    cdef float buffer16x16[16][16]
    cdef float* fullBuffer1 = &gray[0, 0]
    cdef float* fullBuffer2 = &placeholder[0, 0]

    pdqFloat256FromFloatLuma(
        fullBuffer1,
        fullBuffer2,
        numRows,
        numCols,
        buffer64x64,
        buffer16x64,
        buffer16x16,
        quality
    )

    return np.array(buffer16x16, dtype=np.float32).ravel()[::-1], quality

def compute_dihedral(cnp.ndarray[cnp.uint8_t, ndim=3] image):
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be H x W x 3 uint8")

    cdef cnp.ndarray[cnp.float32_t, ndim=2] gray = (
        image[:, :, 0]*0.299 + image[:, :, 1]*0.587 + image[:, :, 2]*0.114
    ).astype(np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim=2] placeholder = np.zeros_like(gray)

    cdef Hash256 hashptrOriginal
    cdef Hash256 hashptrRotate90
    cdef Hash256 hashptrRotate180
    cdef Hash256 hashptrRotate270
    cdef Hash256 hashptrFlipX
    cdef Hash256 hashptrFlipY
    cdef Hash256 hashptrFlipPlus1
    cdef Hash256 hashptrFlipMinus1

    cdef int quality = 0
    cdef int numRows = <int> gray.shape[0]
    cdef int numCols = <int> gray.shape[1]
    cdef float buffer64x64[64][64]
    cdef float buffer16x64[16][64]
    cdef float buffer16x16[16][16]
    cdef float buffer16x16Aux[16][16]
    cdef float* fullBuffer1 = &gray[0, 0]
    cdef float* fullBuffer2 = &placeholder[0, 0]

    cdef bint ok = pdqDihedralHash256esFromFloatLuma(
        fullBuffer1,
        fullBuffer2,
        numRows,
        numCols,
        buffer64x64,
        buffer16x64,
        buffer16x16,
        buffer16x16Aux,
        &hashptrOriginal,
        &hashptrRotate90,
        &hashptrRotate180,
        &hashptrRotate270,
        &hashptrFlipX,
        &hashptrFlipY,
        &hashptrFlipPlus1,
        &hashptrFlipMinus1,
        quality
    )
    if not ok:
        raise RuntimeError("pdqDihedralHash256esFromFloatLuma returned false")

    return [
        hash_to_vector(hashptrOriginal.w),
        hash_to_vector(hashptrRotate90.w),
        hash_to_vector(hashptrRotate180.w),
        hash_to_vector(hashptrRotate270.w),
        hash_to_vector(hashptrFlipX.w),
        hash_to_vector(hashptrFlipY.w),
        hash_to_vector(hashptrFlipPlus1.w),
        hash_to_vector(hashptrFlipMinus1.w),
    ], quality
