#ifndef EIGEN_TYPE_CASTING_WASM_H
#define EIGEN_TYPE_CASTING_WASM_H

namespace Eigen {
namespace internal {

template <>
struct type_casting_traits<float, int> {
  enum {
    VectorizedCast = 1,
    SrcCoeffRatio = 1,
    TgtCoeffRatio = 1
  };
};

template <>
struct type_casting_traits<int, float> {
  enum {
    VectorizedCast = 1,
    SrcCoeffRatio = 1,
    TgtCoeffRatio = 1
  };
};

template<> EIGEN_STRONG_INLINE Packet4i pcast<Packet4f, Packet4i>(const Packet4f& a) {
  return wasm_i32x4_trunc_saturate_f32x4(a);
}

template<> EIGEN_STRONG_INLINE Packet4f pcast<Packet4i, Packet4f>(const Packet4i& a) {
  return wasm_f32x4_convert_i32x4(a);
}

}
}

#endif
