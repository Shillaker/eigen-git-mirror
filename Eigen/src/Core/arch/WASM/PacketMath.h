#ifndef EIGEN_PACKET_MATH_WASM_H
#define EIGEN_PACKET_MATH_WASM_H

namespace Eigen {
    namespace internal {

#define EIGEN_CACHEFRIENDLY_PRODUCT_THRESHOLD 8
#define EIGEN_ARCH_DEFAULT_NUMBER_OF_REGISTERS 8

        typedef float __f32x4
        Packet4f;
        typedef int __i32x4
        Packet4i;
        typedef double __f64x2
        Packet2d;

        template<>
        struct is_arithmetic<float __f32x4> {
            enum {
                value = true
            };
        };
        template<>
        struct is_arithmetic<int __i32x4> {
            enum {
                value = true
            };
        };
        template<>
        struct is_arithmetic<double __f64x2> {
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

        template<>
        struct packet_traits<float> : default_packet_traits {
            typedef Packet4f type;
            typedef Packet4f half;
            enum {
                Vectorizable = 1,
                AlignedOnScalar = 1,
                size = 4,
                HasHalfPacket = 0,

                HasDiv = 1,
                HasExp = 1,
                HasSqrt = 1,
                HasRSqrt = 1,
                HasFloor = 1,
                HasLog = 1,
            };
        };
        template<>
        struct packet_traits<double> : default_packet_traits {
            typedef Packet2d type;
            typedef Packet2d half;
            enum {
                Vectorizable = 1,
                AlignedOnScalar = 1,
                size = 2,
                HasHalfPacket = 0,

                HasDiv = 1,
                HasExp = 1,
                HasSqrt = 1,
                HasRSqrt = 1,
                HasFloor = 1,
                HasLog = 1,
            };
        };

        template<>
        struct packet_traits<int> : default_packet_traits {
            typedef Packet4i type;
            typedef Packet4i half;
            enum {
                Vectorizable = 1,
                AlignedOnScalar = 1,
                size = 4,

                HasBlend = 1
            };
        };

        template<>
        struct unpacket_traits<Packet4f> {
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
        template<>
        struct unpacket_traits<Packet2d> {
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
        template<>
        struct unpacket_traits<Packet4i> {
            typedef int type;
            typedef Packet4i half;
            enum {
                size = 4,
                alignment = Aligned16,
                vectorizable = false,
                masked_load_available = false,
                masked_store_available = false
            };
        };

        template<> EIGEN_STRONG_INLINE Packet4f

        padd<Packet4f>(const Packet4f &a, const Packet4f &b) { return wasm_f32x4_add(a, b); }

        template<> EIGEN_STRONG_INLINE Packet2d

        padd<Packet2d>(const Packet2d &a, const Packet2d &b) { return wasm_f64x2_add(a, b); }

        template<> EIGEN_STRONG_INLINE Packet4i

        padd<Packet4i>(const Packet4i &a, const Packet4i &b) { return wasm_i32x4_add(a, b); }

        template<> EIGEN_STRONG_INLINE Packet4f

        psub<Packet4f>(const Packet4f &a, const Packet4f &b) { return wasm_f32x4_sub(a, b); }

        template<> EIGEN_STRONG_INLINE Packet2d

        psub<Packet2d>(const Packet2d &a, const Packet2d &b) { return wasm_f64x2_sub(a, b); }

        template<> EIGEN_STRONG_INLINE Packet4i

        psub<Packet4i>(const Packet4i &a, const Packet4i &b) { return wasm_i32x4_sub(a, b); }

        template<> EIGEN_STRONG_INLINE Packet4f

        pnegate(const Packet4f &a) { return wasm_f32x4_neg(a); }

        template<> EIGEN_STRONG_INLINE Packet2d

        pnegate(const Packet2d &a) { return wasm_f64x2_neg(a); }

        template<> EIGEN_STRONG_INLINE Packet4i

        pnegate(const Packet4i &a) { return wasm_i32x4_neg(a); }

        template<> EIGEN_STRONG_INLINE Packet4f

        pmul<Packet4f>(const Packet4f &a, const Packet4f &b) { return wasm_f32x4_mul(a, b); }

        template<> EIGEN_STRONG_INLINE Packet2d

        pmul<Packet2d>(const Packet2d &a, const Packet2d &b) { return wasm_f64x2_mul(a, b); }

        template<> EIGEN_STRONG_INLINE Packet4i

        pmul<Packet4i>(const Packet4i &a, const Packet4i &b) { return wasm_i32x4_mul(a, b); }

        template<> EIGEN_STRONG_INLINE Packet4f

        pdiv<Packet4f>(const Packet4f &a, const Packet4f &b) { return wasm_f32x4_div(a, b); }

        template<> EIGEN_STRONG_INLINE Packet2d

        pdiv<Packet2d>(const Packet2d &a, const Packet2d &b) { return wasm_f64x2_div(a, b); }

        template<> EIGEN_STRONG_INLINE Packet4f

        pmin<Packet4f>(const Packet4f &a, const Packet4f &b) { return wasm_f32x4_min(a, b); }

        template<> EIGEN_STRONG_INLINE Packet2d

        pmin<Packet2d>(const Packet2d &a, const Packet2d &b) { return wasm_f64x2_min(a, b); }

        template<> EIGEN_STRONG_INLINE Packet4f

        pmax<Packet4f>(const Packet4f &a, const Packet4f &b) { return wasm_f32x4_max(a, b); }

        template<> EIGEN_STRONG_INLINE Packet2d

        pmax<Packet2d>(const Packet2d &a, const Packet2d &b) { return wasm_f64x2_max(a, b); }

        template<> EIGEN_STRONG_INLINE Packet4f

        pcmp_le(const Packet4f &a, const Packet4f &b) { return wasm_f32x4_le(a, b); }

        template<> EIGEN_STRONG_INLINE Packet4f

        pcmp_lt(const Packet4f &a, const Packet4f &b) { return wasm_f32x4_lt(a, b); }

        template<> EIGEN_STRONG_INLINE Packet4f

        pcmp_eq(const Packet4f &a, const Packet4f &b) { return wasm_f32x4_eq(a, b); }

        template<> EIGEN_STRONG_INLINE Packet4i

        pcmp_eq(const Packet4i &a, const Packet4i &b) { return wasm_i32x4_eq(a, b); }

        template<> EIGEN_STRONG_INLINE Packet2d

        pcmp_eq(const Packet2d &a, const Packet2d &b) { return wasm_f64x2_eq(a, b); }

        template<> EIGEN_STRONG_INLINE Packet4i

        ptrue<Packet4i>(const Packet4i &a) { return wasm_i32x4_all_true(a); }

        template<> EIGEN_STRONG_INLINE Packet4f

        pand<Packet4f>(const Packet4f &a, const Packet4f &b) { return wasm_v128_and(a, b); }

        template<> EIGEN_STRONG_INLINE Packet2d

        pand<Packet2d>(const Packet2d &a, const Packet2d &b) { return wasm_v128_and(a, b); }

        template<> EIGEN_STRONG_INLINE Packet4i

        pand<Packet4i>(const Packet4i &a, const Packet4i &b) { return wasm_v128_and(a, b); }

        template<> EIGEN_STRONG_INLINE Packet4f

        por<Packet4f>(const Packet4f &a, const Packet4f &b) { return wasm_v128_or(a, b); }

        template<> EIGEN_STRONG_INLINE Packet2d

        por<Packet2d>(const Packet2d &a, const Packet2d &b) { return wasm_v128_or(a, b); }

        template<> EIGEN_STRONG_INLINE Packet4i

        por<Packet4i>(const Packet4i &a, const Packet4i &b) { return wasm_v128_or(a, b); }

        template<> EIGEN_STRONG_INLINE Packet4f

        pxor<Packet4f>(const Packet4f &a, const Packet4f &b) { return wasm_v128_xor(a, b); }

        template<> EIGEN_STRONG_INLINE Packet2d

        pxor<Packet2d>(const Packet2d &a, const Packet2d &b) { return wasm_v128_xor(a, b); }

        template<> EIGEN_STRONG_INLINE Packet4i

        pxor<Packet4i>(const Packet4i &a, const Packet4i &b) { return wasm_v128_xor(a, b); }

        template<> EIGEN_STRONG_INLINE Packet4f

        pload<Packet4f>(const float *from) {
            EIGEN_DEBUG_ALIGNED_LOAD
            return wasm_v128_load(from);
        }

        template<> EIGEN_STRONG_INLINE Packet2d

        pload<Packet2d>(const double *from) {
            EIGEN_DEBUG_ALIGNED_LOAD
            return wasm_v128_load(from);
        }

        template<> EIGEN_STRONG_INLINE Packet4i

        pload<Packet4i>(const int *from) {
            EIGEN_DEBUG_ALIGNED_LOAD
            return wasm_v128_load(from);
        }

        template<>
        EIGEN_STRONG_INLINE void
        pstore<float>(float *to, const Packet4f &from) {
            EIGEN_DEBUG_ALIGNED_STORE wasm_v128_store(to, from);
        }

        template<>
        EIGEN_STRONG_INLINE void
        pstore<double>(double *to, const Packet2d &from) {
            EIGEN_DEBUG_ALIGNED_STORE wasm_v128_store(to, from);
        }

        template<>
        EIGEN_STRONG_INLINE void pstore<int>(int *to, const Packet4i &from) {
            EIGEN_DEBUG_ALIGNED_STORE wasm_v128_store(to, from);
        }

        template<> EIGEN_STRONG_INLINE Packet4f

        pabs(const Packet4f &a) { return wasm_f32x4_abs(a); }

        template<> EIGEN_STRONG_INLINE Packet2d

        pabs(const Packet2d &a) { wasm_f64x2_abs(a); }
    }
}

#endif