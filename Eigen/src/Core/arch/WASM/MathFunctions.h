#ifndef EIGEN_MATH_FUNCTIONS_WASM_H
#define EIGEN_MATH_FUNCTIONS_WASM_H

namespace Eigen {

namespace internal {
template<> EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS EIGEN_UNUSED
Packet4f plog<Packet4f>(const Packet4f &_x) {
    return plog_float(_x);
}

template<> EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS EIGEN_UNUSED
Packet4f plog1p<Packet4f>(const Packet4f &_x) {
    return generic_plog1p(_x);
}

template<> EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS EIGEN_UNUSED
Packet4f pexpm1<Packet4f>(const Packet4f &_x) {
    return generic_expm1(_x);
}

template<> EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS EIGEN_UNUSED
Packet4f pexp<Packet4f>(const Packet4f &_x) {
    return pexp_float(_x);
}

template<> EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS EIGEN_UNUSED
Packet2d pexp<Packet2d>(const Packet2d &x) {
    return pexp_double(x);
}

template<> EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS EIGEN_UNUSED
Packet4f psin<Packet4f>(const Packet4f &_x) {
    return psin_float(_x);
}

template<> EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS EIGEN_UNUSED
Packet4f pcos<Packet4f>(const Packet4f &_x) {
    return pcos_float(_x);
}

template<> EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS EIGEN_UNUSED
Packet4f psqrt<Packet4f>(const Packet4f &x) {
    // Once sqrt is taken out of the unimplemented simd instructions
    // we can uncomment this
    // return wasm_f32x4_sqrt(x);

    return wasm_f32x4_make(
            sqrtf(wasm_f32x4_extract_lane(x, 0)),
            sqrtf(wasm_f32x4_extract_lane(x, 1)),
            sqrtf(wasm_f32x4_extract_lane(x, 2)),
            sqrtf(wasm_f32x4_extract_lane(x, 3))
    );
}

template<> EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS EIGEN_UNUSED
Packet2d psqrt<Packet2d>(const Packet2d &x) {
    return wasm_f64x2_make(
            sqrt(wasm_f64x2_extract_lane(x, 0)),
            sqrt(wasm_f64x2_extract_lane(x, 1))
    );
}

template<> EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS EIGEN_UNUSED
Packet4f prsqrt<Packet4f>(const Packet4f &x) {
    // Uncomment once sqrt available
    // return wasm_f32x4_div(pset1<Packet4f>(1.0f), wasm_f32x4_sqrt(x));
    return wasm_f32x4_make(
            1.0f/sqrtf(wasm_f32x4_extract_lane(x, 0)),
            1.0f/sqrtf(wasm_f32x4_extract_lane(x, 1)),
            1.0f/sqrtf(wasm_f32x4_extract_lane(x, 2)),
            1.0f/sqrtf(wasm_f32x4_extract_lane(x, 3))
    );
}

template<> EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS EIGEN_UNUSED
Packet2d prsqrt<Packet2d>(const Packet2d &x) {
    // Uncomment once sqrt available
    // return wasm_f64x2_div(pset1<Packet2d>(1.0), wasm_f64x2_sqrt(x));
    return wasm_f64x2_make(
            1.0/sqrt(wasm_f64x2_extract_lane(x, 0)),
            1.0/sqrt(wasm_f64x2_extract_lane(x, 1))
    );
}

}
}

#endif