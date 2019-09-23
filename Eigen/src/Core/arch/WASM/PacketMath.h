#ifndef EIGEN_PACKET_MATH_WASM_H
#define EIGEN_PACKET_MATH_WASM_H

namespace Eigen {
namespace internal {

#define EIGEN_CACHEFRIENDLY_PRODUCT_THRESHOLD 8
#define EIGEN_ARCH_DEFAULT_NUMBER_OF_REGISTERS 8

typedef __f32x4 Packet4f;
typedef __i32x4 Packet4i;
typedef __f64x2 Packet2d;

template<> struct is_arithmetic<__f32x4> {
    enum {
        value = true
    };
};

template<> struct is_arithmetic<__i32x4> {
    enum {
        value = true
    };
};

template<> struct is_arithmetic<__f64x2> {
    enum {
        value = true
    };
};

#define _EIGEN_DECLARE_CONST_Packet4f(NAME, X) \
const Packet4f p4f_##NAME = pset1<Packet4f>(X)

#define _EIGEN_DECLARE_CONST_Packet2d(NAME, X) \
const Packet2d p2d_##NAME = pset1<Packet2d>(X)

#define _EIGEN_DECLARE_CONST_Packet4f_FROM_INT(NAME, X) \
const Packet4f p4f_##NAME = pset1frombits<Packet4f>(X)

#define _EIGEN_DECLARE_CONST_Packet4i(NAME, X) \
const Packet4i p4i_##NAME = pset1<Packet4i>(X)

template<> struct packet_traits<float> : default_packet_traits {
    typedef Packet4f type;
    typedef Packet4f half;
    enum {
        Vectorizable = 1,
        AlignedOnScalar = 1,
        size = 4,
        HasHalfPacket = 0,

        HasConj = 0,
        HasSetLinear = 0,

        HasDiv = 1,
        HasSqrt = 0,
        HasFloor = 1
    };
};
template<> struct packet_traits<double> : default_packet_traits {
    typedef Packet2d type;
    typedef Packet2d half;
    enum {
        Vectorizable = 1,
        AlignedOnScalar = 1,
        size = 2,
        HasHalfPacket = 0,

        HasDiv = 1,
        HasExp = 1,
        HasSqrt = 0,

        HasFloor = 1,
        HasCeil = 0,
        HasConj = 0
    };
};

template<> struct packet_traits<int> : default_packet_traits {
    typedef Packet4i type;
    typedef Packet4i half;
    enum {
        Vectorizable = 1,
        AlignedOnScalar = 1,
        size = 4,

        HasMax = 0,
        HasMin = 0,
        HasDiv = 0,
        HasConj = 0,
        HasSetLinear = 0
    };
};

template<> struct unpacket_traits<Packet4f> {
    typedef float type;
    typedef Packet4f half;
    typedef Packet4i integer_packet;
    enum {
        size = 4,
        alignment = Aligned16,
        vectorizable = true,
        masked_load_available = false,
        masked_store_available = false
    };
};
template<> struct unpacket_traits<Packet2d> {
    typedef double type;
    typedef Packet2d half;
    enum {
        size = 2,
        alignment = Aligned16,
        vectorizable = true,
        masked_load_available = false,
        masked_store_available = false
    };
};
template<> struct unpacket_traits<Packet4i> {
    typedef int type;
    typedef Packet4i half;
    enum {
        size = 4,
        alignment = Aligned16,
        vectorizable = true,
        masked_load_available = false,
        masked_store_available = false
    };
};

template<> EIGEN_STRONG_INLINE float pfirst<Packet4f>(const Packet4f& a) { return wasm_f32x4_extract_lane(a, 0); }
template<> EIGEN_STRONG_INLINE double pfirst<Packet2d>(const Packet2d& a) { return wasm_f64x2_extract_lane(a, 0); }
template<> EIGEN_STRONG_INLINE int pfirst<Packet4i>(const Packet4i& a) { return wasm_i32x4_extract_lane(a, 0); }

template<> EIGEN_STRONG_INLINE Packet4f pset1<Packet4f>(const float &from) { return wasm_f32x4_splat(from); }
template<> EIGEN_STRONG_INLINE Packet2d pset1<Packet2d>(const double &from) { return wasm_f64x2_splat(from); }
template<> EIGEN_STRONG_INLINE Packet4i pset1<Packet4i>(const int32_t &from) { return wasm_i32x4_splat(from); }
template<> EIGEN_STRONG_INLINE Packet4f pset1frombits<Packet4f>(unsigned int from) { return wasm_f32x4_convert_i32x4(pset1<Packet4i>(from)); }

template<> EIGEN_STRONG_INLINE Packet4f padd<Packet4f>(const Packet4f &a, const Packet4f &b) { return wasm_f32x4_add(a, b); }
template<> EIGEN_STRONG_INLINE Packet2d padd<Packet2d>(const Packet2d &a, const Packet2d &b) { return wasm_f64x2_add(a, b); }
template<> EIGEN_STRONG_INLINE Packet4i padd<Packet4i>(const Packet4i &a, const Packet4i &b) { return wasm_i32x4_add(a, b); }

template<> EIGEN_STRONG_INLINE Packet4f psub<Packet4f>(const Packet4f &a, const Packet4f &b) { return wasm_f32x4_sub(a, b); }
template<> EIGEN_STRONG_INLINE Packet2d psub<Packet2d>(const Packet2d &a, const Packet2d &b) { return wasm_f64x2_sub(a, b); }
template<> EIGEN_STRONG_INLINE Packet4i psub<Packet4i>(const Packet4i &a, const Packet4i &b) { return wasm_i32x4_sub(a, b); }

template<> EIGEN_STRONG_INLINE Packet4f pnegate(const Packet4f &a) { return wasm_f32x4_neg(a); }
template<> EIGEN_STRONG_INLINE Packet2d pnegate(const Packet2d &a) { return wasm_f64x2_neg(a); }
template<> EIGEN_STRONG_INLINE Packet4i pnegate(const Packet4i &a) { return wasm_i32x4_neg(a); }

template<> EIGEN_STRONG_INLINE Packet4f pmul<Packet4f>(const Packet4f &a, const Packet4f &b) { return wasm_f32x4_mul(a, b); }
template<> EIGEN_STRONG_INLINE Packet2d pmul<Packet2d>(const Packet2d &a, const Packet2d &b) { return wasm_f64x2_mul(a, b); }
template<> EIGEN_STRONG_INLINE Packet4i pmul<Packet4i>(const Packet4i &a, const Packet4i &b) { return wasm_i32x4_mul(a, b); }

template<> EIGEN_STRONG_INLINE Packet4f pdiv<Packet4f>(const Packet4f &a, const Packet4f &b) { return wasm_f32x4_div(a, b); }
template<> EIGEN_STRONG_INLINE Packet2d pdiv<Packet2d>(const Packet2d &a, const Packet2d &b) { return wasm_f64x2_div(a, b); }

template<> EIGEN_STRONG_INLINE Packet4f pmin<Packet4f>(const Packet4f &a, const Packet4f &b) { return wasm_f32x4_min(a, b); }
template<> EIGEN_STRONG_INLINE Packet2d pmin<Packet2d>(const Packet2d &a, const Packet2d &b) { return wasm_f64x2_min(a, b); }

template<> EIGEN_STRONG_INLINE Packet4f pmax<Packet4f>(const Packet4f &a, const Packet4f &b) { return wasm_f32x4_max(a, b); }
template<> EIGEN_STRONG_INLINE Packet2d pmax<Packet2d>(const Packet2d &a, const Packet2d &b) { return wasm_f64x2_max(a, b); }

template<> EIGEN_STRONG_INLINE Packet4f pcmp_le(const Packet4f &a, const Packet4f &b) { return wasm_f32x4_le(a, b); }
template<> EIGEN_STRONG_INLINE Packet4f pcmp_lt(const Packet4f &a, const Packet4f &b) { return wasm_f32x4_lt(a, b); }

template<> EIGEN_STRONG_INLINE Packet4f pcmp_eq(const Packet4f &a, const Packet4f &b) { return wasm_f32x4_eq(a, b); }
template<> EIGEN_STRONG_INLINE Packet4i pcmp_eq(const Packet4i &a, const Packet4i &b) { return wasm_i32x4_eq(a, b); }
template<> EIGEN_STRONG_INLINE Packet2d pcmp_eq(const Packet2d &a, const Packet2d &b) { return wasm_f64x2_eq(a, b); }

template<> EIGEN_STRONG_INLINE Packet4f pand<Packet4f>(const Packet4f &a, const Packet4f &b) { return wasm_v128_and(a, b); }
template<> EIGEN_STRONG_INLINE Packet2d pand<Packet2d>(const Packet2d &a, const Packet2d &b) { return wasm_v128_and(a, b); }
template<> EIGEN_STRONG_INLINE Packet4i pand<Packet4i>(const Packet4i &a, const Packet4i &b) { return wasm_v128_and(a, b); }

template<> EIGEN_STRONG_INLINE Packet4f pandnot<Packet4f>(const Packet4f &a, const Packet4f &b) { return wasm_v128_and(a, wasm_v128_not(b)); }
template<> EIGEN_STRONG_INLINE Packet2d pandnot<Packet2d>(const Packet2d &a, const Packet2d &b) { return wasm_v128_and(a, wasm_v128_not(b)); }
template<> EIGEN_STRONG_INLINE Packet4i pandnot<Packet4i>(const Packet4i &a, const Packet4i &b) { return wasm_v128_and(a, wasm_v128_not(b)); }

template<> EIGEN_STRONG_INLINE Packet4f por<Packet4f>(const Packet4f &a, const Packet4f &b) { return wasm_v128_or(a, b); }
template<> EIGEN_STRONG_INLINE Packet2d por<Packet2d>(const Packet2d &a, const Packet2d &b) { return wasm_v128_or(a, b); }
template<> EIGEN_STRONG_INLINE Packet4i por<Packet4i>(const Packet4i &a, const Packet4i &b) { return wasm_v128_or(a, b); }

template<> EIGEN_STRONG_INLINE Packet4f pxor<Packet4f>(const Packet4f &a, const Packet4f &b) { return wasm_v128_xor(a, b); }
template<> EIGEN_STRONG_INLINE Packet2d pxor<Packet2d>(const Packet2d &a, const Packet2d &b) { return wasm_v128_xor(a, b); }
template<> EIGEN_STRONG_INLINE Packet4i pxor<Packet4i>(const Packet4i &a, const Packet4i &b) { return wasm_v128_xor(a, b); }

template<> EIGEN_STRONG_INLINE Packet4f pload<Packet4f>(const float *from) { EIGEN_DEBUG_ALIGNED_LOAD return wasm_v128_load(from); }
template<> EIGEN_STRONG_INLINE Packet2d pload<Packet2d>(const double *from) { EIGEN_DEBUG_ALIGNED_LOAD return wasm_v128_load(from); }
template<> EIGEN_STRONG_INLINE Packet4i pload<Packet4i>(const int *from) { EIGEN_DEBUG_ALIGNED_LOAD return wasm_v128_load(from); }

template<> EIGEN_STRONG_INLINE Packet4f ploadu<Packet4f>(const float* from) { EIGEN_DEBUG_UNALIGNED_LOAD return wasm_v128_load(from); }
template<> EIGEN_STRONG_INLINE Packet2d ploadu<Packet2d>(const double *from) { EIGEN_DEBUG_ALIGNED_LOAD return wasm_v128_load(from); }
template<> EIGEN_STRONG_INLINE Packet4i ploadu<Packet4i>(const int32_t* from) { EIGEN_DEBUG_UNALIGNED_LOAD return wasm_v128_load(from); }

template<> EIGEN_STRONG_INLINE void pstore<float>(float *to, const Packet4f &from) { EIGEN_DEBUG_ALIGNED_STORE wasm_v128_store(to, from); }
template<> EIGEN_STRONG_INLINE void pstore<double>(double *to, const Packet2d &from) { EIGEN_DEBUG_ALIGNED_STORE wasm_v128_store(to, from); }
template<> EIGEN_STRONG_INLINE void pstore<int>(int *to, const Packet4i &from) { EIGEN_DEBUG_ALIGNED_STORE wasm_v128_store(to, from); }

template<> EIGEN_STRONG_INLINE void pstoreu<double>(double* to, const Packet2d& from) { EIGEN_DEBUG_UNALIGNED_STORE wasm_v128_store(to, from); }
template<> EIGEN_STRONG_INLINE void pstoreu<float>(float* to, const Packet4f& from) { EIGEN_DEBUG_UNALIGNED_STORE wasm_v128_store(to, from); }
template<> EIGEN_STRONG_INLINE void pstoreu<int>(int* to, const Packet4i& from) { EIGEN_DEBUG_UNALIGNED_STORE wasm_v128_store(to, from); }

template<> EIGEN_STRONG_INLINE Packet4f pabs(const Packet4f &a) { return wasm_f32x4_abs(a); }
template<> EIGEN_STRONG_INLINE Packet2d pabs(const Packet2d &a) { return wasm_f64x2_abs(a); }

template<> EIGEN_STRONG_INLINE Packet4f pfrexp<Packet4f>(const Packet4f &a, Packet4f &exponent) { return pfrexp_float(a, exponent); }
template<> EIGEN_STRONG_INLINE Packet4f pldexp<Packet4f>(const Packet4f &a, const Packet4f &exponent) { return pldexp_float(a, exponent); }

// TODO - implement pldexp for double
template<> EIGEN_STRONG_INLINE Packet2d pldexp<Packet2d>(const Packet2d& a, const Packet2d& exponent) {
    return a;
}

template<int N> EIGEN_STRONG_INLINE Packet4i pshiftright(Packet4i a) { return wasm_i32x4_shr(a, N); }
template<int N> EIGEN_STRONG_INLINE Packet4i pshiftleft(Packet4i a) { return wasm_i32x4_shl(a, N); }

template<> EIGEN_STRONG_INLINE Packet4f pfloor<Packet4f>(const Packet4f& a) {
    return wasm_f32x4_make(
            floorf(wasm_f32x4_extract_lane(a, 0)),
            floorf(wasm_f32x4_extract_lane(a, 1)),
            floorf(wasm_f32x4_extract_lane(a, 2)),
            floorf(wasm_f32x4_extract_lane(a, 3))
    );
}

template<> EIGEN_STRONG_INLINE Packet2d pfloor<Packet2d>(const Packet2d& a) {
    return wasm_f64x2_make(
         floor(wasm_f64x2_extract_lane(a, 0)),
         floor(wasm_f64x2_extract_lane(a, 1))
    );
}

template<> EIGEN_STRONG_INLINE float predux<Packet4f>(const Packet4f& a) {
    return wasm_f32x4_extract_lane(a, 0) +
         wasm_f32x4_extract_lane(a, 1) +
         wasm_f32x4_extract_lane(a, 2) +
         wasm_f32x4_extract_lane(a, 3);
}

template<> EIGEN_STRONG_INLINE int predux<Packet4i>(const Packet4i& a) {
    return wasm_i32x4_extract_lane(a, 0) +
         wasm_i32x4_extract_lane(a, 1) +
         wasm_i32x4_extract_lane(a, 2) +
         wasm_i32x4_extract_lane(a, 3);
}

template<> EIGEN_STRONG_INLINE double predux<Packet2d>(const Packet2d& a) {
    return wasm_f64x2_extract_lane(a, 0) + wasm_f64x2_extract_lane(a, 1);
}

template<> EIGEN_STRONG_INLINE float predux_max<Packet4f>(const Packet4f& a) {
    float arr[4] = {
        wasm_f32x4_extract_lane(a, 0),
        wasm_f32x4_extract_lane(a, 1),
        wasm_f32x4_extract_lane(a, 2),
        wasm_f32x4_extract_lane(a, 3)
    };
    return *std::max_element(arr, arr + 4);
}
template<> EIGEN_STRONG_INLINE double predux_max<Packet2d>(const Packet2d& a) {
    double arr[2] = {
        wasm_f64x2_extract_lane(a, 0),
        wasm_f64x2_extract_lane(a, 1)
    };
    return *std::max_element(arr, arr + 2);
}
template<> EIGEN_STRONG_INLINE int predux_max<Packet4i>(const Packet4i& a) {
    int arr[4] = {
        wasm_i32x4_extract_lane(a, 0),
        wasm_i32x4_extract_lane(a, 1),
        wasm_i32x4_extract_lane(a, 2),
        wasm_i32x4_extract_lane(a, 3)
    };
    return *std::max_element(arr, arr + 4);
}

template<> EIGEN_STRONG_INLINE float predux_min<Packet4f>(const Packet4f& a) {
    float arr[4] = {
        wasm_f32x4_extract_lane(a, 0),
        wasm_f32x4_extract_lane(a, 1),
        wasm_f32x4_extract_lane(a, 2),
        wasm_f32x4_extract_lane(a, 3)
    };
    return *std::min_element(arr, arr + 4);
}
template<> EIGEN_STRONG_INLINE double predux_min<Packet2d>(const Packet2d& a) {
    double arr[2] = {
        wasm_f64x2_extract_lane(a, 0),
        wasm_f64x2_extract_lane(a, 1)
    };
    return *std::min_element(arr, arr + 2);
}
template<> EIGEN_STRONG_INLINE int predux_min<Packet4i>(const Packet4i& a) {
    int arr[4] = {
        wasm_i32x4_extract_lane(a, 0),
        wasm_i32x4_extract_lane(a, 1),
        wasm_i32x4_extract_lane(a, 2),
        wasm_i32x4_extract_lane(a, 3)
    };
    return *std::min_element(arr, arr + 4);
}

template<> EIGEN_STRONG_INLINE float predux_mul<Packet4f>(const Packet4f& a) {
    return wasm_f32x4_extract_lane(a, 0) * wasm_f32x4_extract_lane(a, 1) * wasm_f32x4_extract_lane(a, 2) * wasm_f32x4_extract_lane(a, 3);
}
template<> EIGEN_STRONG_INLINE double predux_mul<Packet2d>(const Packet2d& a) {
    return wasm_f64x2_extract_lane(a, 0) * wasm_f64x2_extract_lane(a, 1);
}
template<> EIGEN_STRONG_INLINE int predux_mul<Packet4i>(const Packet4i& a) {
    return wasm_i32x4_extract_lane(a, 0) * wasm_i32x4_extract_lane(a, 1) * wasm_i32x4_extract_lane(a, 2) * wasm_i32x4_extract_lane(a, 3);
}

EIGEN_DEVICE_FUNC inline void ptranspose(PacketBlock<Packet4f,4>& kernel) {
    // 4x4 matrix transposition. Diagonal remains same, then can do 6
    // swaps of cells

    // Swap 01 and 10
    float tmp = wasm_f32x4_extract_lane(kernel.packet[0], 1);
    wasm_f32x4_replace_lane(kernel.packet[0], 1, wasm_f32x4_extract_lane(kernel.packet[1], 0));
    wasm_f32x4_replace_lane(kernel.packet[1], 0, tmp);

    // Swap 02 and 20
    tmp = wasm_f32x4_extract_lane(kernel.packet[0], 2);
    wasm_f32x4_replace_lane(kernel.packet[0], 2, wasm_f32x4_extract_lane(kernel.packet[2], 0));
    wasm_f32x4_replace_lane(kernel.packet[2], 0, tmp);

    // Swap 03 and 30
    tmp = wasm_f32x4_extract_lane(kernel.packet[0], 3);
    wasm_f32x4_replace_lane(kernel.packet[0], 3, wasm_f32x4_extract_lane(kernel.packet[3], 0));
    wasm_f32x4_replace_lane(kernel.packet[3], 0, tmp);

    // Swap 04 and 40
    tmp = wasm_f32x4_extract_lane(kernel.packet[0], 4);
    wasm_f32x4_replace_lane(kernel.packet[0], 4, wasm_f32x4_extract_lane(kernel.packet[4], 0));
    wasm_f32x4_replace_lane(kernel.packet[4], 0, tmp);

    // Swap 12 and 21
    tmp = wasm_f32x4_extract_lane(kernel.packet[1], 2);
    wasm_f32x4_replace_lane(kernel.packet[1], 2, wasm_f32x4_extract_lane(kernel.packet[2], 1));
    wasm_f32x4_replace_lane(kernel.packet[2], 1, tmp);

    // Swap 13 and 31
    tmp = wasm_f32x4_extract_lane(kernel.packet[1], 3);
    wasm_f32x4_replace_lane(kernel.packet[1], 3, wasm_f32x4_extract_lane(kernel.packet[3], 1));
    wasm_f32x4_replace_lane(kernel.packet[3], 1, tmp);

    // Swap 23 and 32
    tmp = wasm_f32x4_extract_lane(kernel.packet[2], 3);
    wasm_f32x4_replace_lane(kernel.packet[2], 3, wasm_f32x4_extract_lane(kernel.packet[3], 2));
    wasm_f32x4_replace_lane(kernel.packet[3], 2, tmp);
}

EIGEN_DEVICE_FUNC inline void ptranspose(PacketBlock<Packet2d,2>& kernel) {
    // 2x2 matrix transpose is just swapping two values
    double tmp = wasm_f64x2_extract_lane(kernel.packet[0], 1);
    wasm_f64x2_replace_lane(kernel.packet[0], 1, wasm_f64x2_extract_lane(kernel.packet[1], 0));
    wasm_f64x2_replace_lane(kernel.packet[1], 0, tmp);
}

EIGEN_DEVICE_FUNC inline void ptranspose(PacketBlock<Packet4i,4>& kernel) {
    // 4x4 matrix transposition with integers same as floats

    // Swap 01 and 10
    int tmp = wasm_i32x4_extract_lane(kernel.packet[0], 1);
    wasm_i32x4_replace_lane(kernel.packet[0], 1, wasm_i32x4_extract_lane(kernel.packet[1], 0));
    wasm_i32x4_replace_lane(kernel.packet[1], 0, tmp);

    // Swap 02 and 20
    tmp = wasm_i32x4_extract_lane(kernel.packet[0], 2);
    wasm_i32x4_replace_lane(kernel.packet[0], 2, wasm_i32x4_extract_lane(kernel.packet[2], 0));
    wasm_i32x4_replace_lane(kernel.packet[2], 0, tmp);

    // Swap 03 and 30
    tmp = wasm_i32x4_extract_lane(kernel.packet[0], 3);
    wasm_i32x4_replace_lane(kernel.packet[0], 3, wasm_i32x4_extract_lane(kernel.packet[3], 0));
    wasm_i32x4_replace_lane(kernel.packet[3], 0, tmp);

    // Swap 04 and 40
    tmp = wasm_i32x4_extract_lane(kernel.packet[0], 4);
    wasm_i32x4_replace_lane(kernel.packet[0], 4, wasm_i32x4_extract_lane(kernel.packet[4], 0));
    wasm_i32x4_replace_lane(kernel.packet[4], 0, tmp);

    // Swap 12 and 21
    tmp = wasm_i32x4_extract_lane(kernel.packet[1], 2);
    wasm_i32x4_replace_lane(kernel.packet[1], 2, wasm_i32x4_extract_lane(kernel.packet[2], 1));
    wasm_i32x4_replace_lane(kernel.packet[2], 1, tmp);

    // Swap 13 and 31
    tmp = wasm_i32x4_extract_lane(kernel.packet[1], 3);
    wasm_i32x4_replace_lane(kernel.packet[1], 3, wasm_i32x4_extract_lane(kernel.packet[3], 1));
    wasm_i32x4_replace_lane(kernel.packet[3], 1, tmp);

    // Swap 23 and 32
    tmp = wasm_i32x4_extract_lane(kernel.packet[2], 3);
    wasm_i32x4_replace_lane(kernel.packet[2], 3, wasm_i32x4_extract_lane(kernel.packet[3], 2));
    wasm_i32x4_replace_lane(kernel.packet[3], 2, tmp);
}

}
}

#endif