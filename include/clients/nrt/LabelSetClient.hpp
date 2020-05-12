#pragma once
#include "CommonResults.hpp"
#include "FluidSharedInstanceAdaptor.hpp"
#include "clients/common/SharedClientUtils.hpp"
#include "data/FluidDataSet.hpp"
#include <clients/common/FluidBaseClient.hpp>
#include <clients/common/MessageSet.hpp>
#include <clients/common/OfflineClient.hpp>
#include <clients/common/ParameterSet.hpp>
#include <clients/common/ParameterTypes.hpp>
#include <clients/common/Result.hpp>
#include <clients/common/FluidNRTClientWrapper.hpp>
#include <data/FluidTensor.hpp>
#include <data/TensorTypes.hpp>
#include <data/FluidFile.hpp>
#include <data/FluidIndex.hpp>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>

namespace fluid {
namespace client {

class LabelSetClient : public FluidBaseClient, OfflineIn, OfflineOut {
  enum { kName };

public:
  using string = std::string;
  using BufferPtr = std::shared_ptr<BufferAdaptor>;
  using LabelSet = FluidDataSet<string, string, 1>;

  template <typename T> Result process(FluidContext &) { return {}; }

  FLUID_DECLARE_PARAMS(StringParam<Fixed<true>>("name", "DataSet"));

  LabelSetClient(ParamSetViewType &p) : mParams(p), mLabelSet(1) {}


  // TODO: refactor with addPoint
  MessageResult<void> addLabel(string id, string label) {
    if(id.empty()) return EmptyIdError;
    if(label.empty()) return EmptyLabelError;
    FluidTensor<string, 1> point = {label};
    return mLabelSet.add(id, point) ? OKResult : DuplicateError;
  }

  MessageResult<string> getLabel(string id) const {
    if(id.empty()) return EmptyIdError;
    FluidTensor<string, 1> point(1);
    mLabelSet.get(id, point);
    return  point(0);
  }

  MessageResult<void> updateLabel(string id, string label) {
    if(id.empty()) return EmptyIdError;
    if(label.empty()) return EmptyLabelError;
    FluidTensor<string, 1> point = {label};
    return mLabelSet.update(id, point) ? OKResult : PointNotFoundError;
  }

  MessageResult<void> deleteLabel(string id) {
    return mLabelSet.remove(id) ? OKResult : PointNotFoundError;
  }

  MessageResult<index> size() {
    return mLabelSet.size();
  }

  MessageResult<void> clear() {
    mLabelSet = LabelSet(1);
    return OKResult;
  }

  MessageResult<void> write(string fileName) {
    auto file = FluidFile(fileName, "w");
    if(!file.valid()){return {Result::Status::kError, file.error()};}
    file.add("labels", mLabelSet.getData());
    file.add("ids", mLabelSet.getIds());
    file.add("rows", mLabelSet.size());
    return file.write()? OKResult:WriteError;
  }

 MessageResult<void> read(string fileName) {
   auto file = FluidFile(fileName, "r");
   if(!file.valid()){return {Result::Status::kError, file.error()};}
   if(!file.read()){return ReadError;}
   if(!file.checkKeys({"labels","ids","rows"})){
     return {Result::Status::kError, file.error()};
   }
   index  rows;
   file.get("rows", rows);
   FluidTensor<string, 1> ids(rows);
   FluidTensor<string, 2> labels(rows, 1);
   file.get("ids", ids, rows);
   file.get("labels", labels, rows, 1);
   mLabelSet = LabelSet(ids, labels);
   return OKResult;
 }

  FLUID_DECLARE_MESSAGES(
      makeMessage("addLabel", &LabelSetClient::addLabel),
      makeMessage("getLabel", &LabelSetClient::getLabel),
      makeMessage("deleteLabel", &LabelSetClient::deleteLabel),
      makeMessage("size", &LabelSetClient::size),
      makeMessage("clear", &LabelSetClient::clear),
      makeMessage("write", &LabelSetClient::write),
      makeMessage("read", &LabelSetClient::read)
  );

  const LabelSet getLabelSet() const { return mLabelSet; }
  void setLabelSet(LabelSet ls) {mLabelSet = ls; }

private:
  LabelSet mLabelSet{1};
};
using LabelSetClientRef = SharedClientRef<LabelSetClient>;
using NRTThreadedLabelSetClient =
    NRTThreadingAdaptor<typename LabelSetClientRef::SharedType>;

} // namespace client
} // namespace fluid
