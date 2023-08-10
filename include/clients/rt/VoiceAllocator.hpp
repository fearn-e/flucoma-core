/*
Part of the Fluid Corpus Manipulation Project (http://www.flucoma.org/)
Copyright 2017-2019 University of Huddersfield.
Licensed under the BSD-3 License.
See license.md file in the project root for full license information.
This project has received funding from the European Research Council (ERC)
under the European Union’s Horizon 2020 research and innovation programme
(grant agreement No 725899).
*/

#pragma once

#include "../common/AudioClient.hpp"
#include "../common/FluidBaseClient.hpp"
#include "../common/FluidSource.hpp"
#include "../common/ParameterConstraints.hpp"
#include "../common/ParameterSet.hpp"
#include "../common/ParameterTypes.hpp"
// #include "../../algorithms/public/RunningStats.hpp"
#include "../../data/TensorTypes.hpp"
#include <Windows.h>

namespace fluid {
namespace client {
namespace voiceallocator {

template <typename T>
using HostVector = FluidTensorView<T, 1>;

constexpr auto VoiceAllocatorParams = defineParameters(LongParam(
    "history", "History Size", 2,
    Min(2))); // will be most probably a max num voice and all other params

class VoiceAllocatorClient : public FluidBaseClient,
                             public ControlIn,
                             ControlOut
{
public:
  using ParamDescType = decltype(VoiceAllocatorParams);

  using ParamSetViewType = ParameterSetView<ParamDescType>;
  std::reference_wrapper<ParamSetViewType> mParams;

  void setParams(ParamSetViewType& p) { mParams = p; }

  template <size_t N>
  auto& get() const
  {
    return mParams.get().template get<N>();
  }

  static constexpr auto& getParameterDescriptors()
  {
    return VoiceAllocatorParams;
  }

  VoiceAllocatorClient(ParamSetViewType& p, FluidContext&, int numVoices = 5)
      : mParams(p), mInputSize{0}, mSizeTracker{0}, mVoices(numVoices), mIncomingVoices(numVoices), mFreqRange{50}
  {
    controlChannelsIn(3);
    controlChannelsOut({3, -1});
    setInputLabels({"left", "middle", "right"});
    setOutputLabels({"lefto", "middleo", "righto"});
    for (int i = 0; i < numVoices; ++i) { mFreeVoices.push(i); }
  }

  template <typename T>
  void process(std::vector<HostVector<T>>& input,
               std::vector<HostVector<T>>& output, FluidContext&)
  {

    bool inputSizeChanged = mInputSize != input[0].size();
    bool sizeParamChanged = mSizeTracker.changed(get<0>());

    if (inputSizeChanged || sizeParamChanged)
    {
      mInputSize = input[0].size();
      //      mAlgorithm.init(get<0>(),mInputSize);
    }
    //algo start
    PrintTensor(input);

    mIncomingVoices.resize(0);
    //place non-zero frequency voices in mIncomingVoices
    for (int i = 0; i < mVoices.size(); ++i)
    {
        if (input[0].row(i) != 0 && input[1].row(i) != 0)
        {
            std::pair<double, double> temp;
            temp.first = input[0].row(i); temp.second = input[1].row(i);
            mIncomingVoices.push_back(temp);
        }
    }
    //debug code, output mIncomingVoices
    for (auto each : mIncomingVoices) {
        OutputDebugString(std::to_string(each.first).c_str());
        OutputDebugString(" ");
        OutputDebugString(std::to_string(each.second).c_str());
        OutputDebugString("\n");
    }
    output[2] <<= input[2];
    output[1] <<= input[1];
    output[0] <<= input[0];
    PrintTensor(output);
  }

  template <typename T>
  void PrintTensor(std::vector<HostVector<T>>& tensorToPrint)
  {
    std::string o;
    std::string temp;
    index       columns = 3;
    for (index j = 0; j < columns; ++j)
    {
      for (index i = 0; i < tensorToPrint[j].rows(); ++i)
      {
        o += " ";
        temp = std::to_string(tensorToPrint[j].row(i));
        temp.resize(4);
        o += temp;
      }
      o += ",";
    }
    o += "\n";
    OutputDebugString(o.c_str());
  }

  MessageResult<void> clear()
  {
    //    mAlgorithm.init(get<0>(),mInputSize);
    return {};
  }

  static auto getMessageDescriptors()
  {
    return defineMessages(makeMessage("clear", &VoiceAllocatorClient::clear));
  }

  index latency() { return 0; }

private:
  //  algorithm::RunningStats mAlgorithm;
  std::queue<int>                               mFreeVoices;
  std::deque<int>                               mActiveVoices;
  std::vector<std::tuple<float, float, bool>>   mVoices; //freq, mag, active
  std::vector<std::pair<double, double>>        mIncomingVoices; //freq, mag
  int                                           mFreqRange;
  index                                         mInputSize;
  ParameterTrackChanges<index>                  mSizeTracker;
};

} // namespace voiceallocator

using VoiceAllocatorClient =
    ClientWrapper<voiceallocator::VoiceAllocatorClient>;

} // namespace client
} // namespace fluid
