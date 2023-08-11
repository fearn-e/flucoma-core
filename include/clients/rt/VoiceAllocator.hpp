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
#include "../../algorithms/util/PartialTracking.hpp"
#include <Windows.h>
#include <Eigen/Core>
#include <cmath>

namespace fluid {
namespace client {
namespace voiceallocator {

enum VoiceAllocatorParamIndex {
    kMaxNumVoices,
    kBirthLowTreshold,
    kBirthHighTreshold,
    kMinTrackLen,
    kTrackMethod,
    kTrackMagRange,
    kTrackFreqRange,
    kTrackProb
};

template <typename T>
using HostVector = FluidTensorView<T, 1>;

constexpr auto VoiceAllocatorParams = defineParameters(
    LongParam("maxNumVoices", "Max Number of Voices", 5, Min(1), Max(256)),
    FloatParam("birthLowTreshold", "Track Birth Low Frequency Treshold", -24, Min(-144), Max(0)),
    FloatParam("birthHighTreshold", "Track Birth High Frequency Treshold", -60, Min(-144), Max(0)),
    LongParam("minTrackLen", "Minimum Track Length", 1, Min(1)),
    EnumParam("trackMethod", "Tracking Method", 0, "Greedy", "Hungarian"),
    FloatParam("trackMagRange", "Tracking Magnitude Range (dB)", 15., Min(1.), Max(200.)),
    FloatParam("trackFreqRange", "Tracking Frequency Range (Hz)", 50., Min(1.), Max(10000.)),
    FloatParam("trackProb", "Tracking Matching Probability", 0.5, Min(0.0), Max(1.0))
);

class VoiceAllocatorClient : public FluidBaseClient,
                             public ControlIn,
                             ControlOut
{
    template <typename T>
    using vector = rt::vector<T>;

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

  VoiceAllocatorClient(ParamSetViewType& p, FluidContext& c, int numVoices = 5)
      : mParams(p), mInputSize{ 0 }, mSizeTracker{ 0 }, mFreeVoices(c.allocator()), mActiveVoices(c.allocator()), mFreqRange{50}, mTracking(c.allocator())
  {
    controlChannelsIn(3);
    controlChannelsOut({3, -1});
    setInputLabels({"left", "middle", "right"});
    setOutputLabels({"lefto", "middleo", "righto"});
    for (int i = 0; i < numVoices; ++i) { mFreeVoices.push(i); }
    mTracking.init();
  }

  template <typename T>
  void process(std::vector<HostVector<T>>& input,
               std::vector<HostVector<T>>& output, FluidContext& c)
  {
    Allocator& alloc = c.allocator();
    bool inputSizeChanged = mInputSize != input[0].size();
    bool sizeParamChanged = mSizeTracker.changed(get<0>());

    if (inputSizeChanged || sizeParamChanged)
    {
      mInputSize = input[0].size();
      //      mAlgorithm.init(get<0>(),mInputSize);
    }
    //algo start
    PrintTensor(input);

    //place non-zero freq-mag pairs in incomingVoices
    rt::vector<algorithm::SinePeak> incomingVoices(0, alloc);
    for (int i = 0; i < mVoices.size(); ++i)
    {
        if (input[0].row(i) != 0 && input[1].row(i) != 0)
        {
            incomingVoices.push_back({input[0].row(i), 
                input[1].row(i), false});
        }
    }
    //debug code, output mIncomingVoices
    for (auto each : mIncomingVoices) {
        OutputDebugString(std::to_string(each.first).c_str());

    if (true) //change this to IF INPUT = TYPE MAGNITUDE, if dB skip
    {
        for (algorithm::SinePeak voice : incomingVoices)
        {
            voice.logMag = 20 * log10(std::max(voice.logMag, algorithm::epsilon));
        }
    }

    double maxAmp = -999;
    for (algorithm::SinePeak voice : incomingVoices)
    {
        if (voice.logMag > maxAmp) { maxAmp = voice.logMag; }
    }

    mTracking.processFrame(incomingVoices, maxAmp, get<kMinTrackLen>(), get<kBirthLowTreshold>(), get<kBirthHighTreshold>(), get<kTrackMethod>(), get<kTrackMagRange>(), get<kTrackFreqRange>(), get<kTrackProb>(), alloc);

    vector<algorithm::SinePeak> voices = mTracking.getActivePeaks(alloc);
        OutputDebugString(" ");
        OutputDebugString(std::to_string(each.second).c_str());
        OutputDebugString("\n");
    }

    for (int i = 0; i < voices.size(); ++i)
    {
        output[0].row(i) = voices[i].freq;
        output[1].row(i) = voices[i].logMag;
        output[2].row(i) = voices[i].assigned;
    }

    mTracking.prune();

    //output[2] <<= input[2];
    //output[1] <<= input[1];
    //output[0] <<= input[0];
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
  algorithm::PartialTracking                    mTracking;
  rt::queue<int>                                mFreeVoices;
  rt::deque<int>                                mActiveVoices;
  int                                           mFreqRange;
  index                                         mInputSize;
  ParameterTrackChanges<index>                  mSizeTracker;
};

} // namespace voiceallocator

using VoiceAllocatorClient =
    ClientWrapper<voiceallocator::VoiceAllocatorClient>;

} // namespace client
} // namespace fluid
