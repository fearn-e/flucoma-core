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
    LongParamRuntimeMax<Primary>("maxNumVoices", "Max Number of Voices", 5, Min(1), Max(256)),
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

  VoiceAllocatorClient(ParamSetViewType& p, FluidContext& c)
      : mParams(p), mInputSize{ 0 }, mSizeTracker{ 0 }, mFreeVoices(c.allocator()), mActiveVoices(c.allocator()), mVoiceIDAssignment(0, c.allocator()), mTracking(c.allocator())
  {
    controlChannelsIn(2);
    controlChannelsOut({3, get<kMaxNumVoices>(), get<kMaxNumVoices>().max()});
    setInputLabels({"frequencies", "magnitudes"});
    setOutputLabels({"frequencies", "magnitudes", "voice IDs"});
    init();
  }

  void init()
  {
      mMaxNumVoices = static_cast<index>(get<kMaxNumVoices>());
      controlChannelsOut({ 3, mMaxNumVoices });
      while (!mFreeVoices.empty())
      {
          mFreeVoices.pop();
      }
      while (!mActiveVoices.empty())
      {
          mActiveVoices.pop_back();
      }
      while (!mVoiceIDAssignment.empty())
      {
          mVoiceIDAssignment.pop_back();
      }
      for (index i = 0; i < mMaxNumVoices; ++i)
      {
          mFreeVoices.push(i);
          mVoiceIDAssignment.push_back(-1);
      }
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
      init();
    }
    //OutputDebugString("input:\n");
    //PrintTensor(input);

    //place non-zero freq-mag pairs in incomingVoices
    rt::vector<algorithm::SinePeak> incomingVoices(0, alloc);
    for (int i = 0; i < mMaxNumVoices; ++i)
    {
        if (input[0].row(i) != 0 && input[1].row(i) != 0)
        {
            incomingVoices.push_back({input[0].row(i), 
                input[1].row(i), false});
        }
    }

    if (true) //todo: change this to IF INPUT = TYPE MAGNITUDE, if dB skip
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

    vector<std::tuple<index, algorithm::SinePeak>> voices = mTracking.getActiveVoices(alloc);

    /*//debug console outputs
    OutputDebugString("Incoming Voices:\n");
    for (algorithm::SinePeak voice : incomingVoices)
    {
        OutputDebugString(std::to_string(voice.freq).c_str());
        OutputDebugString(" ");
        OutputDebugString(std::to_string(voice.logMag).c_str());
        OutputDebugString(" ");
        OutputDebugString(std::to_string(voice.assigned).c_str());
        OutputDebugString("\n");
    }
    OutputDebugString("Max Amp:\n");
    OutputDebugString(std::to_string(maxAmp).c_str());
    OutputDebugString("\n");
    OutputDebugString("Outgoing Voices:\n");
    for (algorithm::SinePeak voice : voices)
    {
        OutputDebugString(std::to_string(voice.freq).c_str());
        OutputDebugString(" ");
        OutputDebugString(std::to_string(voice.logMag).c_str());
        OutputDebugString(" ");
        OutputDebugString(std::to_string(voice.assigned).c_str());
        OutputDebugString("\n");
    }
    *///end of debug

    //voice ID assignment
    /*
    for (auto activeVoice : mActiveVoices)
    {
        //no note offs, so need to do note offs ourselves

        if (!mFreeVoices.empty())
        {
            index voiceIndex = mFreeVoices.front();
            mFreeVoices.pop();
            mActiveVoices.push_back(voiceIndex);

        }
        else
        {

        }
    }*/

    //todo: need to set output size to = kMaxNumVoices

    //clear output
    for (int i = 0; i < mMaxNumVoices; ++i)
    {
        output[0].row(i) = 0;
        output[1].row(i) = 0;
        output[2].row(i) = -1;
    }

    for (int i = 0; i < mMaxNumVoices; ++i)
    {
        output[0].row(i) = std::get<1>(voices[i]).freq;
        output[1].row(i) = std::get<1>(voices[i]).logMag;
        output[2].row(i) = std::get<0>(voices[i]);
    }

    mTracking.prune();

    //output[2] <<= input[2];
    //output[1] <<= input[1];
    //output[0] <<= input[0];
  }

  /* template <typename T>
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
  }*/

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
  index                                         mMaxNumVoices;
  vector<index>                                 mVoiceIDAssignment;
  rt::queue<int>                                mFreeVoices;
  rt::deque<int>                                mActiveVoices;
  index                                         mInputSize;
  ParameterTrackChanges<index>                  mSizeTracker;
};

} // namespace voiceallocator

using VoiceAllocatorClient =
    ClientWrapper<voiceallocator::VoiceAllocatorClient>;

} // namespace client
} // namespace fluid
