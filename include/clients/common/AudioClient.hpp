#pragma once

#include <cstddef>
#include <type_traits>

namespace fluid  {
namespace client {

struct Audio {};
struct AudioIn: Audio {};
struct AudioOut: Audio {};

template <typename T>
constexpr bool isAudioIn = std::is_base_of<AudioIn, T>::value;

template <typename T>
constexpr bool isAudioOut = std::is_base_of<AudioOut, T>::value;

struct ControlOut{};

template <typename T>
constexpr bool isControlOut = std::is_base_of<ControlOut, T>::value;

}
}
