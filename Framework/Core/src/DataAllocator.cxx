// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "Framework/CompilerBuiltins.h"
#include "Framework/TableBuilder.h"
#include "Framework/TableTreeHelpers.h"
#include "Framework/DataAllocator.h"
#include "Framework/MessageContext.h"
#include "Framework/ArrowContext.h"
#include "Framework/DataSpecUtils.h"
#include "Framework/DataProcessingHeader.h"
#include "Framework/FairMQResizableBuffer.h"
#include "Framework/DataProcessingContext.h"
#include "Framework/DeviceSpec.h"
#include "Framework/StreamContext.h"
#include "Framework/Signpost.h"
#include "Headers/DataHeader.h"
#include "Headers/DataHeaderHelpers.h"
#include "Headers/Stack.h"

#include <fairmq/Device.h>

#include <arrow/ipc/writer.h>
#include <arrow/type.h>
#include <arrow/io/memory.h>
#include <arrow/util/config.h>

#include <TClonesArray.h>

#include <utility>

O2_DECLARE_DYNAMIC_LOG(stream_context);
O2_DECLARE_DYNAMIC_LOG(parts);

namespace o2::framework
{

using DataHeader = o2::header::DataHeader;
using DataDescription = o2::header::DataDescription;
using DataProcessingHeader = o2::framework::DataProcessingHeader;

DataAllocator::DataAllocator(ServiceRegistryRef contextRegistry)
  : mRegistry{contextRegistry}
{
}

RouteIndex DataAllocator::matchDataHeader(const Output& spec, size_t timeslice)
{
  auto& allowedOutputRoutes = mRegistry.get<DeviceSpec const>().outputs;
  auto& stream = mRegistry.get<o2::framework::StreamContext>();
  // FIXME: we should take timeframeId into account as well.
  for (auto ri = 0; ri < allowedOutputRoutes.size(); ++ri) {
    auto& route = allowedOutputRoutes[ri];
    if (DataSpecUtils::match(route.matcher, spec.origin, spec.description, spec.subSpec) && ((timeslice % route.maxTimeslices) == route.timeslice)) {
      stream.routeCreated.at(ri) = true;
      auto sid = _o2_signpost_id_t{(int64_t)&stream};
      O2_SIGNPOST_EVENT_EMIT(stream_context, sid, "data_allocator", "Route %" PRIu64 " (%{public}s) created for timeslice %" PRIu64,
                             (uint64_t)ri, DataSpecUtils::describe(route.matcher).c_str(), (uint64_t)timeslice);
      return RouteIndex{ri};
    }
  }
  throw runtime_error_f(
    "Worker is not authorised to create message with "
    "origin(%s) description(%s) subSpec(%d)",
    spec.origin.as<std::string>().c_str(),
    spec.description.as<std::string>().c_str(),
    spec.subSpec);
}

DataChunk& DataAllocator::newChunk(const Output& spec, size_t size)
{
  auto& timingInfo = mRegistry.get<TimingInfo>();
  RouteIndex routeIndex = matchDataHeader(spec, timingInfo.timeslice);
  auto& context = mRegistry.get<MessageContext>();

  fair::mq::MessagePtr headerMessage = headerMessageFromOutput(spec, routeIndex,                     //
                                                               o2::header::gSerializationMethodNone, //
                                                               size                                  //
  );
  auto& co = context.add<MessageContext::ContainerRefObject<DataChunk>>(std::move(headerMessage), routeIndex, 0, size);
  return co;
}

void DataAllocator::adoptChunk(const Output& spec, char* buffer, size_t size, fair::mq::FreeFn* freefn, void* hint = nullptr)
{
  // Find a matching channel, create a new message for it and put it in the
  // queue to be sent at the end of the processing
  RouteIndex routeIndex = matchDataHeader(spec, mRegistry.get<TimingInfo>().timeslice);

  fair::mq::MessagePtr headerMessage = headerMessageFromOutput(spec, routeIndex,                     //
                                                               o2::header::gSerializationMethodNone, //
                                                               size                                  //
  );

  // FIXME: how do we want to use subchannels? time based parallelism?
  auto& context = mRegistry.get<MessageContext>();
  context.add<MessageContext::TrivialObject>(std::move(headerMessage), routeIndex, 0, buffer, size, freefn, hint);
}

fair::mq::MessagePtr DataAllocator::headerMessageFromOutput(Output const& spec,                     //
                                                            RouteIndex routeIndex,                  //
                                                            o2::header::SerializationMethod method, //
                                                            size_t payloadSize)                     //
{
  auto& timingInfo = mRegistry.get<TimingInfo>();
  DataHeader dh;
  dh.dataOrigin = spec.origin;
  dh.dataDescription = spec.description;
  dh.subSpecification = spec.subSpec;
  dh.payloadSize = payloadSize;
  dh.payloadSerializationMethod = method;
  dh.tfCounter = timingInfo.tfCounter;
  dh.firstTForbit = timingInfo.firstTForbit;
  dh.runNumber = timingInfo.runNumber;

  DataProcessingHeader dph{timingInfo.timeslice, 1, timingInfo.creation};
  static_cast<o2::header::BaseHeader&>(dph).flagsDerivedHeader |= timingInfo.keepAtEndOfStream ? DataProcessingHeader::KEEP_AT_EOS_FLAG : 0;
  auto& proxy = mRegistry.get<FairMQDeviceProxy>();
  auto* transport = proxy.getOutputTransport(routeIndex);

  auto channelAlloc = o2::pmr::getTransportAllocator(transport);
  return o2::pmr::getMessage(o2::header::Stack{channelAlloc, dh, dph, spec.metaHeader});
}

void DataAllocator::addPartToContext(RouteIndex routeIndex, fair::mq::MessagePtr&& payloadMessage, const Output& spec,
                                     o2::header::SerializationMethod serializationMethod)
{
  auto headerMessage = headerMessageFromOutput(spec, routeIndex, serializationMethod, 0);
  O2_SIGNPOST_ID_FROM_POINTER(pid, parts, headerMessage->GetData());
  // FIXME: this is kind of ugly, we know that we can change the content of the
  // header message because we have just created it, but the API declares it const
  const DataHeader* cdh = o2::header::get<DataHeader*>(headerMessage->GetData());
  auto* dh = const_cast<DataHeader*>(cdh);
  dh->payloadSize = payloadMessage->GetSize();
  O2_SIGNPOST_START(parts, pid, "parts", "addPartToContext %{public}s@%p %" PRIu64,
                    cdh ? fmt::format("{}/{}/{}", cdh->dataOrigin, cdh->dataDescription, cdh->subSpecification).c_str() : "unknown", headerMessage->GetData(), dh->payloadSize);
  auto& context = mRegistry.get<MessageContext>();
  // make_scoped creates the context object inside of a scope handler, since it goes out of
  // scope immediately, the created object is scheduled and can be directly sent if the context
  // is configured with the dispatcher callback
  context.make_scoped<MessageContext::TrivialObject>(std::move(headerMessage), std::move(payloadMessage), routeIndex);
}

void DataAllocator::adopt(const Output& spec, std::string* ptr)
{
  std::unique_ptr<std::string> payload(ptr);
  auto& timingInfo = mRegistry.get<TimingInfo>();
  RouteIndex routeIndex = matchDataHeader(spec, timingInfo.timeslice);
  // the correct payload size is set later when sending the
  // StringContext, see DataProcessor::doSend
  auto header = headerMessageFromOutput(spec, routeIndex, o2::header::gSerializationMethodNone, 0);
  const DataHeader* cdh = o2::header::get<DataHeader*>(header->GetData());
  O2_SIGNPOST_ID_FROM_POINTER(pid, parts, header->GetData());
  O2_SIGNPOST_START(parts, pid, "parts", "addPartToContext %{public}s@%p %" PRIu64,
                    cdh ? fmt::format("{}/{}/{}", cdh->dataOrigin, cdh->dataDescription, cdh->subSpecification).c_str() : "unknown", header->GetData(), cdh->payloadSize);
  mRegistry.get<StringContext>().addString(std::move(header), std::move(payload), routeIndex);
  assert(payload.get() == nullptr);
}

void doWriteTable(std::shared_ptr<FairMQResizableBuffer> b, arrow::Table* table)
{
  auto mock = std::make_shared<arrow::io::MockOutputStream>();
  int64_t expectedSize = 0;
  auto mockWriter = arrow::ipc::MakeStreamWriter(mock.get(), table->schema());
  arrow::Status outStatus;
  if (O2_BUILTIN_LIKELY(table->num_rows() != 0)) {
    outStatus = mockWriter.ValueOrDie()->WriteTable(*table);
  } else {
    std::vector<std::shared_ptr<arrow::Array>> columns;
    columns.resize(table->columns().size());
    for (size_t ci = 0; ci < table->columns().size(); ci++) {
      columns[ci] = table->column(ci)->chunk(0);
    }
    auto batch = arrow::RecordBatch::Make(table->schema(), 0, columns);
    outStatus = mockWriter.ValueOrDie()->WriteRecordBatch(*batch);
  }

  expectedSize = mock->Tell().ValueOrDie();
  auto reserve = b->Reserve(expectedSize);
  if (reserve.ok() == false) {
    throw std::runtime_error("Unable to reserve memory for table");
  }

  auto stream = std::make_shared<FairMQOutputStream>(b);
  auto outBatch = arrow::ipc::MakeStreamWriter(stream.get(), table->schema());
  if (outBatch.ok() == false) {
    throw ::std::runtime_error("Unable to create batch writer");
  }

  if (O2_BUILTIN_UNLIKELY(table->num_rows() == 0)) {
    std::vector<std::shared_ptr<arrow::Array>> columns;
    columns.resize(table->columns().size());
    for (size_t ci = 0; ci < table->columns().size(); ci++) {
      columns[ci] = table->column(ci)->chunk(0);
    }
    auto batch = arrow::RecordBatch::Make(table->schema(), 0, columns);
    outStatus = outBatch.ValueOrDie()->WriteRecordBatch(*batch);
  } else {
    outStatus = outBatch.ValueOrDie()->WriteTable(*table);
  }

  if (outStatus.ok() == false) {
    throw std::runtime_error("Unable to Write table");
  }
}

void DataAllocator::adopt(const Output& spec, LifetimeHolder<TableBuilder>& tb)
{
  auto& timingInfo = mRegistry.get<TimingInfo>();
  RouteIndex routeIndex = matchDataHeader(spec, timingInfo.timeslice);
  auto header = headerMessageFromOutput(spec, routeIndex, o2::header::gSerializationMethodArrow, 0);
  const DataHeader* cdh = o2::header::get<DataHeader*>(header->GetData());
  O2_SIGNPOST_ID_FROM_POINTER(pid, parts, header->GetData());
  O2_SIGNPOST_START(parts, pid, "parts", "adopt %{public}s@%p %" PRIu64,
                    cdh ? fmt::format("{}/{}/{}", cdh->dataOrigin, cdh->dataDescription, cdh->subSpecification).c_str() : "unknown", header->GetData(), cdh->payloadSize);
  auto& context = mRegistry.get<ArrowContext>();

  auto creator = [transport = context.proxy().getOutputTransport(routeIndex)](size_t s) -> std::unique_ptr<fair::mq::Message> {
    return transport->CreateMessage(s);
  };
  auto buffer = std::make_shared<FairMQResizableBuffer>(creator);

  tb.callback = [buffer = buffer, transport = context.proxy().getOutputTransport(routeIndex)](TableBuilder& builder) -> void {
    auto table = builder.finalize();
    doWriteTable(buffer, table.get());
    // deletion happens in the caller
  };

  /// To finalise this we write the table to the buffer.
  auto finalizer = [](std::shared_ptr<FairMQResizableBuffer> b) -> void {
    // Finalization not needed, as we do it using the LifetimeHolder callback
  };

  context.addBuffer(std::move(header), buffer, std::move(finalizer), routeIndex);
}

void DataAllocator::adopt(const Output& spec, LifetimeHolder<FragmentToBatch>& f2b)
{
  auto& timingInfo = mRegistry.get<TimingInfo>();
  RouteIndex routeIndex = matchDataHeader(spec, timingInfo.timeslice);

  auto header = headerMessageFromOutput(spec, routeIndex, o2::header::gSerializationMethodArrow, 0);
  auto& context = mRegistry.get<ArrowContext>();

  auto creator = [transport = context.proxy().getOutputTransport(routeIndex)](size_t s) -> std::unique_ptr<fair::mq::Message> {
    return transport->CreateMessage(s);
  };
  auto buffer = std::make_shared<FairMQResizableBuffer>(creator);

  f2b.callback = [buffer = buffer, transport = context.proxy().getOutputTransport(routeIndex)](FragmentToBatch& source) {
    // Serialization happens in here, so that we can
    // get rid of the intermediate tree 2 table object, saving memory.
    auto batch = source.finalize();
    auto mock = std::make_shared<arrow::io::MockOutputStream>();
    int64_t expectedSize = 0;
    auto mockWriter = arrow::ipc::MakeStreamWriter(mock.get(), batch->schema());
    arrow::Status outStatus = mockWriter.ValueOrDie()->WriteRecordBatch(*batch);

    expectedSize = mock->Tell().ValueOrDie();
    auto reserve = buffer->Reserve(expectedSize);
    if (reserve.ok() == false) {
      throw std::runtime_error("Unable to reserve memory for table");
    }

    auto deferredWriterStream = source.streamer(buffer);

    auto outBatch = arrow::ipc::MakeStreamWriter(deferredWriterStream, batch->schema());
    if (outBatch.ok() == false) {
      throw ::std::runtime_error("Unable to create batch writer");
    }

    outStatus = outBatch.ValueOrDie()->WriteRecordBatch(*batch);

    if (outStatus.ok() == false) {
      throw std::runtime_error("Unable to Write batch");
    }
    // deletion happens in the caller
  };

  auto finalizer = [](std::shared_ptr<FairMQResizableBuffer> b) -> void {
    // This is empty because we already serialised the object when
    // the LifetimeHolder goes out of scope. See code above.
  };

  context.addBuffer(std::move(header), buffer, std::move(finalizer), routeIndex);
}

void DataAllocator::adopt(const Output& spec, std::shared_ptr<arrow::Table> ptr)
{
  auto& timingInfo = mRegistry.get<TimingInfo>();
  RouteIndex routeIndex = matchDataHeader(spec, timingInfo.timeslice);
  auto header = headerMessageFromOutput(spec, routeIndex, o2::header::gSerializationMethodArrow, 0);
  auto& context = mRegistry.get<ArrowContext>();

  auto creator = [transport = context.proxy().getOutputTransport(routeIndex)](size_t s) -> std::unique_ptr<fair::mq::Message> {
    return transport->CreateMessage(s);
  };
  auto buffer = std::make_shared<FairMQResizableBuffer>(creator);

  auto writer = [table = ptr](std::shared_ptr<FairMQResizableBuffer> b) -> void {
    doWriteTable(b, table.get());
  };

  context.addBuffer(std::move(header), buffer, std::move(writer), routeIndex);
}

void DataAllocator::snapshot(const Output& spec, const char* payload, size_t payloadSize,
                             o2::header::SerializationMethod serializationMethod)
{
  auto& proxy = mRegistry.get<FairMQDeviceProxy>();
  auto& timingInfo = mRegistry.get<TimingInfo>();

  RouteIndex routeIndex = matchDataHeader(spec, timingInfo.timeslice);
  fair::mq::MessagePtr payloadMessage(proxy.createOutputMessage(routeIndex, payloadSize));
  memcpy(payloadMessage->GetData(), payload, payloadSize);

  addPartToContext(routeIndex, std::move(payloadMessage), spec, serializationMethod);
}

Output DataAllocator::getOutputByBind(OutputRef&& ref)
{
  if (ref.label.empty()) {
    throw runtime_error("Invalid (empty) OutputRef provided.");
  }
  auto& allowedOutputRoutes = mRegistry.get<DeviceSpec const>().outputs;
  for (auto ri = 0ul, re = allowedOutputRoutes.size(); ri != re; ++ri) {
    if (allowedOutputRoutes[ri].matcher.binding.value == ref.label) {
      auto spec = allowedOutputRoutes[ri].matcher;
      auto dataType = DataSpecUtils::asConcreteDataTypeMatcher(spec);
      return Output{dataType.origin, dataType.description, ref.subSpec, std::move(ref.headerStack)};
    }
  }
  std::string availableRoutes;
  for (auto const& route : allowedOutputRoutes) {
    availableRoutes += "\n - " + route.matcher.binding.value + ": " + DataSpecUtils::describe(route.matcher);
  }
  throw runtime_error_f("Unable to find OutputSpec with label %s. Available Routes: %s", ref.label.c_str(), availableRoutes.c_str());
  O2_BUILTIN_UNREACHABLE();
}

bool DataAllocator::isAllowed(Output const& query)
{
  auto& allowedOutputRoutes = mRegistry.get<DeviceSpec const>().outputs;
  for (auto const& route : allowedOutputRoutes) {
    if (DataSpecUtils::match(route.matcher, query.origin, query.description, query.subSpec)) {
      return true;
    }
  }
  return false;
}

void DataAllocator::adoptFromCache(const Output& spec, CacheId id, header::SerializationMethod method)
{
  // Find a matching channel, extract the message for it form the container
  // and put it in the queue to be sent at the end of the processing
  auto& timingInfo = mRegistry.get<TimingInfo>();
  RouteIndex routeIndex = matchDataHeader(spec, timingInfo.timeslice);

  auto& context = mRegistry.get<MessageContext>();
  fair::mq::MessagePtr payloadMessage = context.cloneFromCache(id.value);

  fair::mq::MessagePtr headerMessage = headerMessageFromOutput(spec, routeIndex,         //
                                                               method,                   //
                                                               payloadMessage->GetSize() //
  );

  context.add<MessageContext::TrivialObject>(std::move(headerMessage), std::move(payloadMessage), routeIndex);
}

void DataAllocator::cookDeadBeef(const Output& spec)
{
  auto& proxy = mRegistry.get<FairMQDeviceProxy>();
  auto& timingInfo = mRegistry.get<TimingInfo>();

  // We get the output route from the original spec, but we send it
  // using the binding of the deadbeef subSpecification.
  RouteIndex routeIndex = matchDataHeader(spec, timingInfo.timeslice);
  auto deadBeefOutput = Output{spec.origin, spec.description, 0xdeadbeef};
  auto headerMessage = headerMessageFromOutput(deadBeefOutput, routeIndex, header::gSerializationMethodNone, 0);

  addPartToContext(routeIndex, proxy.createOutputMessage(routeIndex, 0), deadBeefOutput, header::gSerializationMethodNone);
}

} // namespace o2::framework
