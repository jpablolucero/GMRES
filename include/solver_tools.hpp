#pragma once
#include <vector>
#include <concepts>
#include <type_traits>
#include <algorithm>
#include <numeric>

template <typename T>
concept HasUnsignedSizeType = requires
  {
    typename T::size_type;
    requires std::is_unsigned_v<typename T::size_type>;
  };

template <typename T>
concept HasSizeFunction = requires(T t)
  {
    { t.size() } -> std::same_as<typename T::size_type>;
  };

template <typename T>
concept ConstructibleFromInteger = requires(T, typename T::size_type x)
  {
    requires std::is_integral_v<typename T::size_type>;
    { T(x) };
  };

template <typename T>
concept HasFloatingPointValueType = requires
  {
    typename T::value_type;
    requires std::is_floating_point_v<typename T::value_type>;
  };

template <typename T>
concept HasIndexOperator = requires(T t, typename T::size_type i)
  {
    { t[i] } -> std::convertible_to<typename T::value_type>;
  };

template <typename T>
concept ValidVectorType
= HasUnsignedSizeType<T> &&
  HasSizeFunction<T> &&
  ConstructibleFromInteger<T> &&
  HasFloatingPointValueType<T> &&
  HasIndexOperator<T>;

template <typename InnerProduct, typename V>
concept ValidInnerProduct =
ValidVectorType<V> &&
requires(InnerProduct inner_product, const V& v1, const V& v2)
  {
    { inner_product(v1, v2) } -> std::convertible_to<typename V::value_type>;
  };

template <class M>
class IdentityPreconditioner
{
public:
  IdentityPreconditioner() {}
  IdentityPreconditioner(const M&) {}
  
  template<class V>
  const V& operator()(const V& ve) const
  {
    return ve;
  }
};

struct DefaultInnerProduct
{
  template <ValidVectorType V>
  typename V::value_type operator()(const V & v1, const V & v2) const
  {
    typename V::value_type result = 0.0;
    for (typename V::size_type i = 0; i < v1.size(); ++i)
      result += v1[i] * v2[i];
    return result;
  }
};
