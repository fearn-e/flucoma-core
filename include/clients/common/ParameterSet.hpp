#pragma once

#include "ParameterConstraints.hpp"
#include "ParameterTypes.hpp"
#include "TupleUtilities.hpp"
#include <tuple>
#include <functional>

namespace fluid {
namespace client {

/// Each parameter descriptor in the base client is a three-element tuple
/// Third element is flag indicating whether fixed (instantiation only) or not

template <typename, typename>
class ParameterDescriptorSet;

template <size_t... Os, typename... Ts>
class ParameterDescriptorSet<std::index_sequence<Os...>, std::tuple<Ts...>>
{
  template <bool B>
  struct FixedParam
  {
    template <typename T>
    using apply = std::is_same<Fixed<B>, typename std::tuple_element<2, T>::type>;
  };
  
  using IsFixed   = FixedParam<true>;
  using IsMutable = FixedParam<false>;
  
  struct IsRelational
  {
    template <typename T>
    using apply = std::is_base_of<impl::Relational, T>;
  };
  
  struct IsNonRelational
  {
    template <typename T>
    using apply = std::integral_constant<bool, !(std::is_base_of<impl::Relational,T>::value)>;
  };
  
  template<typename T>
  using DefaultValue = decltype(std::declval<T>().defaultValue);

public:
  
  template <typename T>
  using ValueType       = typename std::tuple_element<0, T>::type::type;

  using ValueTuple      = std::tuple<ValueType<Ts>...>;
  using ValueRefTuple   = std::tuple<ValueType<Ts>&...>;
  using DescriptorType  = std::tuple<Ts...>;

  template <size_t N>
  using ParamType = typename std::tuple_element<0,typename std::tuple_element<N, DescriptorType>::type>::type;
  
  using IndexList         = std::index_sequence_for<Ts...>;
  using FixedIndexList    = typename impl::FilterTupleIndices<IsFixed, DescriptorType, IndexList>::type;
  using MutableIndexList  = typename impl::FilterTupleIndices<IsMutable, DescriptorType, IndexList>::type;
  
  template<typename T, typename List>
  using RelationalList = typename impl::FilterTupleIndices<IsRelational,T,List>::type;

  template<typename T, typename List>
  using NonRelationalList = typename impl::FilterTupleIndices<IsNonRelational,T,List>::type;

    
  template<typename T>
  size_t NumOf() const
  {
    return typename impl::FilterTupleIndices<T,DescriptorType,IndexList>::size();    
  }
    
  static constexpr size_t NumFixedParams    = FixedIndexList::size();
  static constexpr size_t NumMutableParams  = MutableIndexList::size();

  constexpr ParameterDescriptorSet(const Ts &&... ts) : mDescriptors{std::make_tuple(ts...)} {}
  constexpr ParameterDescriptorSet(const std::tuple<Ts...>&& t): mDescriptors{t} {}

  constexpr size_t size() const noexcept { return sizeof...(Ts); }
  constexpr size_t count() const noexcept { return countImpl(IndexList()); }
  
  template <template <size_t N, typename T> class Func,typename...Args>
  void iterate(Args&&...args) const
  {
    iterateImpl<Func>(IndexList(),std::forward<Args>(args)...);
  }
  
  template <template <size_t N, typename T> class Func>
  void iterateFixed() const
  {
    iterateImpl<Func>(FixedIndexList());
  }
  
  template <template <size_t N, typename T> class Func>
  void iterateMutable() const
  {
    iterateImpl<Func>(MutableIndexList());
  }
    
  template <size_t N>
  constexpr auto& get() const
  {
    return std::get<0>(std::get<N>(mDescriptors));
  }

  constexpr const DescriptorType& descriptors() const { return mDescriptors; }
  
  
  template<size_t N>
  std::enable_if_t<isDetected<DefaultValue, ParamType<N>>::value,typename ParamType<N>::type>
  makeValue() const
  {
    return std::get<0>(std::get<N>(mDescriptors)).defaultValue;
  }

  template<size_t N>
  std::enable_if_t<!isDetected<DefaultValue, ParamType<N>>::value,typename ParamType<N>::type>
  makeValue() const
  {
    return typename ParamType<N>::type{};
  }

private:

  const DescriptorType mDescriptors;
  
  template <size_t... Is>
  constexpr size_t countImpl(std::index_sequence<Is...>) const noexcept
  {
    size_t count{0};
    std::initializer_list<int>{(count = count + std::get<0>(std::get<Is>(mDescriptors)).fixedSize, 0)...};
    return count;
  }
  
  template <template <size_t N, typename T> class Op, typename...Args,size_t... Is>
  void iterateImpl(std::index_sequence<Is...>,Args&&...args) const
  {
    (void)std::initializer_list<int>{(Op<Is, ParamType<Is>>()(std::get<0>(std::get<Is>(mDescriptors)),std::forward<Args>(args)...), 0)...};
  }
};

template <typename>
class ParameterSetView;

template <size_t...Os, typename... Ts>
class ParameterSetView<const ParameterDescriptorSet<std::index_sequence<Os...>, std::tuple<Ts...>>>
{
  enum ConstraintTypes
  {
    kAll,
    kNonRelational,
    kRelational,
  };
    
protected:

  using DescriptorSetType = ParameterDescriptorSet<std::index_sequence<Os...>, std::tuple<Ts...>>;
    
  template <size_t N>
  constexpr auto descriptor() const
  {
    return mDescriptors.get().template get<N>();
  }
    
  using DescriptorType      = typename DescriptorSetType::DescriptorType;
  using ValueTuple          = typename DescriptorSetType::ValueTuple;
  using ValueRefTuple       = typename DescriptorSetType::ValueRefTuple;
  using IndexList           = typename DescriptorSetType::IndexList;
  using FixedIndexList      = typename DescriptorSetType::FixedIndexList;
  using MutableIndexList    = typename DescriptorSetType::MutableIndexList;

  template<typename T, typename List>
  using RelationalList = typename DescriptorSetType::template RelationalList<T,List>;

  template<typename T, typename List>
  using NonRelationalList =  typename DescriptorSetType::template NonRelationalList<T,List>;

  template <size_t N>
  using ParamType = typename DescriptorSetType::template ParamType<N>;

public:

  constexpr ParameterSetView(const DescriptorSetType &d, ValueRefTuple t)
  : mDescriptors{std::cref(d)}
  , mParams{t}
  , mKeepConstrained(false)
  {}
  
  auto keepConstrained(bool keep)
  {
    std::array<Result, sizeof...(Ts)> results;
    
    if (keep && !mKeepConstrained)
      results = constrainParameterValues();
      
    mKeepConstrained = keep;
    return results;
  }
    
  std::array<Result, sizeof...(Ts)> constrainParameterValues()
  {
    return constrainParameterValuesImpl(IndexList());
  }

  template <template <size_t N, typename T> class Func, typename... Args>
  std::array<Result, sizeof...(Ts)> setParameterValues(bool reportage, Args &&... args)
  {
    return setParameterValuesImpl<Func>(IndexList(), reportage, std::forward<Args>(args)...);
  }

  template <template <size_t N, typename T> class Func, typename... Args>
  std::array<Result, sizeof...(Ts)> setFixedParameterValues(bool reportage, Args &&... args)
  {
    auto res =  setParameterValuesImpl<Func>(FixedIndexList(), reportage, std::forward<Args>(args)...);
    return constrainParameterValuesImpl(IndexList());
  }

  template <template <size_t N, typename T> class Func, typename... Args>
  std::array<Result, sizeof...(Ts)> setMutableParameterValues(bool reportage, Args &&... args)
  {
    return setParameterValuesImpl<Func>(MutableIndexList(), reportage, std::forward<Args>(args)...);
  }

  template <template <size_t N, typename T> class Func, typename... Args>
  void forEachParam(Args &&... args)
  {
    forEachParamImpl<Func>(IndexList(), std::forward<Args>(args)...);
  }

  template <typename T, template <size_t, typename> class Func, typename... Args>
  void forEachParamType(Args &&... args)
  {
    using Is = typename impl::FilterTupleIndices<IsParamType<T>, std::decay_t<DescriptorType>, IndexList>::type;
    forEachParamImpl<Func>(Is{}, std::forward<Args>(args)...);
  }

  void reset() { resetImpl(IndexList()); }

  template <size_t N>
  void set(typename ParamType<N>::type &&x, Result *reportage) noexcept
  {
    if (reportage) reportage->reset();
    auto &constraints   = constraint<N>();
    auto &param         = std::get<N>(mParams);
    const size_t offset = std::get<N>(std::make_tuple(Os...));
    param               = mKeepConstrained ? constrain<offset, N, kAll>(x, constraints, reportage) : x;
  }

  template <std::size_t N>
  auto &get() const
  {
    return std::get<N>(mParams);
  }
 
  template<size_t offset>
  auto subset()
  {
    return impl::RefTupleFrom<offset>(mParams);
  }
  
private:
  template <typename T>
  struct IsParamType
  {
    template <typename U>
    using apply = std::is_same<T, typename std::tuple_element<0, U>::type>;
  };

  template <size_t N, typename VTuple>
  auto &paramValue(VTuple &values)
  {
    return std::get<N>(values);
  }
  
  template <size_t N>
  constexpr auto& constraint() const
  {
    return std::get<1>(std::get<N>(mDescriptors.get().descriptors()));
  }

  template <size_t... Is>
  void resetImpl(std::index_sequence<Is...>)
  {
    std::initializer_list<int>{(std::get<Is>(mParams) = descriptor<Is>().defaultValue, 0)...};
  }

  template <template <size_t, typename> class Func, typename... Args, size_t... Is>
  void forEachParamImpl(std::index_sequence<Is...>, Args &&... args)
  {
    (void)std::initializer_list<int>{(Func<Is, ParamType<Is>>()(get<Is>(), std::forward<Args>(args)...), 0)...};
  }

  template <template <size_t, typename> class Func, typename... Args, size_t... Is>
  auto setParameterValuesImpl(std::index_sequence<Is...>, bool reportage, Args &&... args)
  {
    static std::array<Result, sizeof...(Ts)> results;

    std::initializer_list<int>{
        (set<Is>(Func<Is, ParamType<Is>>()(std::forward<Args>(args)...), reportage ? &results[Is] : nullptr),
         0)...};

    return results;
  }
    
  template <size_t Offset, size_t N, ConstraintTypes C, typename T, typename... Constraints>
  T constrain(T& thisParam, const std::tuple<Constraints...> &c, Result *r)
  {
    using CT  = std::tuple<Constraints...>;
    using Idx = std::index_sequence_for<Constraints...>;
    switch(C)
    {
      case kAll:            return constrainImpl<Offset, N>(thisParam, c, Idx(), r);
      case kNonRelational:  return constrainImpl<Offset, N>(thisParam, c, NonRelationalList<CT, Idx>(), r);
      case kRelational:     return constrainImpl<Offset, N>(thisParam, c, RelationalList<CT, Idx>(), r);
    }
  }
  
  template <size_t Offset, size_t N, typename T, typename Constraints, size_t... Is>
  T constrainImpl(T &thisParam, Constraints &c, std::index_sequence<Is...>, Result *r)
  {
    T res = thisParam;
    (void) std::initializer_list<int>{(std::get<Is>(c).template clamp<Offset, N>(res, mParams, mDescriptors.get(), r), 0)...};
    return res;
  }
    
  template <size_t... Is>
  auto constrainParameterValuesImpl(std::index_sequence<Is...>)
  {
    std::array<Result, sizeof...(Is)> results;
        
    (void)std::initializer_list<int>{(paramValue<Is>(mParams) = constrain<Os, Is, kNonRelational>(paramValue<Is>(mParams), constraint<Is>(), &std::get<Is>(results)), 0)...};
    (void)std::initializer_list<int>{(paramValue<Is>(mParams) = constrain<Os, Is, kRelational>(paramValue<Is>(mParams), constraint<Is>(), &std::get<Is>(results)), 0)...};
        
    return results;
  }
  
//  template<size_t N, typename F>
//  void addUpdateCallback(F&& f)
//  {
//    mUpdateCallbacks[N].emplace_back(std::move(f));
//  }
  
protected:
//  std::array<std::vector<std::function<void()>>,sizeof...(Ts)> mUpdateCallbacks;
  std::reference_wrapper<const DescriptorSetType> mDescriptors;
  bool          mKeepConstrained;
private:  
  ValueRefTuple mParams;
};

template <typename>
class ParameterSet
{};
  
template <size_t...Os, typename... Ts>
class ParameterSet<const ParameterDescriptorSet<std::index_sequence<Os...>, std::tuple<Ts...>>>
  : public ParameterSetView<const ParameterDescriptorSet<std::index_sequence<Os...>, std::tuple<Ts...>>>
{
  using DescriptorSetType = ParameterDescriptorSet<std::index_sequence<Os...>, std::tuple<Ts...>>;
  using ViewType = ParameterSetView<const DescriptorSetType>;
  using IndexList = typename DescriptorSetType::IndexList;
  using ValueTuple = typename DescriptorSetType::ValueTuple;

public:
  
  constexpr ParameterSet(const DescriptorSetType &d)
    : ViewType(d, createRefTuple(IndexList())), mParams{create(d, IndexList())}
  {}
  
  // Copy construct / assign
  
  ParameterSet(const ParameterSet& p)
    : ViewType(p.mDescriptors.get(), createRefTuple(IndexList())), mParams{p.mParams}
  {
    this->mKeepConstrained = p.mKeepConstrained; 
  }
  
  ParameterSet& operator =(const ParameterSet&p)
  {
    *(static_cast<ViewType*>(this)) = ViewType(p.mDescriptors.get(), createRefTuple(IndexList()));
    mParams = p.mParams;
    this->mKeepConstrained = p.mKeepConstrained;
    return *this;
  }
  
  // Move construct /assign

  ParameterSet(ParameterSet&&) noexcept = default;
  ParameterSet& operator =(ParameterSet&&) noexcept = default;
 
private:
  
  template <size_t... Is>
  constexpr auto create(const DescriptorSetType &d, std::index_sequence<Is...>) const
  {
    return std::make_tuple(d.template makeValue<Is>()...);
  }
  
  template <size_t... Is>
  constexpr auto createRefTuple(std::index_sequence<Is...>)
  {
    return std::tie(std::get<Is>(mParams)...);
  }
  
  ValueTuple mParams;
};
    
template <typename... Ts>
using ParamDescTypeFor = ParameterDescriptorSet<impl::zeroSequenceFor<Ts...>, std::tuple<Ts...>>;
    
template <typename... Args>
constexpr ParamDescTypeFor<Args...> defineParameters(Args &&... args)
{
  return {std::forward<Args>(args)...};
}

//define COMMA_SEP(r, token, i, e) BOOST_PP_COMMA_IF(i) token(e)


//define WRAP(token, ...) BOOST_PP_SEQ_FOR_EACH_I(COMMA_SEP, token, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))
//
//define FLUID_PARAM(x,type,...) decltype(type##Param(#x,__VA_ARGS__)) x = type##Param(#x,__VA_ARGS__);
//
//define FLUID_DEFINE_PARAMS(...) BOOST_PP_SEQ_FOR_EACH_I(FLUID_PARAM, _, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))

//Boilerplate macro for clients
#define FLUID_DECLARE_PARAMS(...) \
    using ParamDescType = std::add_const_t<decltype(defineParameters(__VA_ARGS__))>; \
    using ParamSetViewType = ParameterSetView<const ParamDescType>; \
    std::reference_wrapper<ParamSetViewType>   mParams; \
    void setParams(ParamSetViewType& p) { mParams = p; } \
    template<size_t N> auto& get() const { return mParams.get().template get<N>();} \
    static constexpr ParamDescType  getParameterDescriptors() { return defineParameters(__VA_ARGS__);}


auto constexpr NoParameters = defineParameters(); 

} // namespace client
} // namespace fluid
