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
//
#include "Framework/Plugins.h"
#include "Framework/ServiceHandle.h"
#include "Framework/ServiceSpec.h"
#include "Framework/CommonServices.h"
#include "Framework/RawDeviceService.h"
#include "Framework/DeviceSpec.h"
#include <InfoLogger/InfoLogger.hxx>
#include <fairlogger/Logger.h>
#include <fairmq/Device.h>
#include <fairmq/shmem/Monitor.h>
#include <fairmq/shmem/Common.h>
#include <fairmq/ProgOptions.h>

using AliceO2::InfoLogger::InfoLogger;
using AliceO2::InfoLogger::InfoLoggerContext;

using namespace o2::framework;

namespace o2::framework
{
/// This is a global service because read only
template <>
struct ServiceKindExtractor<InfoLoggerContext> {
  constexpr static ServiceKind kind = ServiceKind::Global;
};

} // namespace o2::framework

struct MissingService {
};

struct InfoLoggerContextPlugin : o2::framework::ServicePlugin {
  o2::framework::ServiceSpec* create() final
  {
    return new ServiceSpec{
      .name = "infologger-contex",
      .init = CommonServices::simpleServiceInit<InfoLoggerContext, InfoLoggerContext>(),
      .configure = CommonServices::noConfiguration(),
      .start = [](ServiceRegistryRef services, void* service) {
        auto& infoLoggerContext = services.get<InfoLoggerContext>();
        auto run = services.get<RawDeviceService>().device()->fConfig->GetProperty<std::string>("runNumber", "unspecified");
        infoLoggerContext.setField(InfoLoggerContext::FieldName::Run, run);
        auto partition = services.get<RawDeviceService>().device()->fConfig->GetProperty<std::string>("environment_id", "unspecified");
        infoLoggerContext.setField(InfoLoggerContext::FieldName::Partition, partition);
      },
      .kind = ServiceKind::Serial};
  }
};

// Creates the sink for FairLogger / InfoLogger integration
auto createInfoLoggerSinkHelper(InfoLogger* logger, InfoLoggerContext* ctx)
{
  return [logger,
          ctx](const std::string& content, const fair::LogMetaData& metadata) {
    // translate FMQ metadata
    InfoLogger::InfoLogger::Severity severity = InfoLogger::Severity::Undefined;
    int level = InfoLogger::undefinedMessageOption.level;

    switch (metadata.severity) {
      case fair::Severity::fatal:
        severity = InfoLogger::Severity::Fatal;
        level = 1;
        break;
      case fair::Severity::critical:
        severity = InfoLogger::Severity::Error;
        level = 1;
        break;
      case fair::Severity::error:
        severity = InfoLogger::Severity::Error;
        level = 2;
        break;
      case fair::Severity::alarm:
        severity = InfoLogger::Severity::Warning;
        level = 6;
        break;
      case fair::Severity::important:
        severity = InfoLogger::Severity::Info;
        level = 7;
        break;
      case fair::Severity::warn:
        severity = InfoLogger::Severity::Warning;
        level = 11;
        break;
      case fair::Severity::state:
        severity = InfoLogger::Severity::Info;
        level = 12;
        break;
      case fair::Severity::info:
        severity = InfoLogger::Severity::Info;
        level = 13;
        break;
      case fair::Severity::debug:
        severity = InfoLogger::Severity::Debug;
        level = 14;
        break;
      case fair::Severity::detail:
        severity = InfoLogger::Severity::Debug;
        level = 15;
        break;
      case fair::Severity::debug1:
        severity = InfoLogger::Severity::Debug;
        level = 16;
        break;
      case fair::Severity::debug2:
        severity = InfoLogger::Severity::Debug;
        level = 17;
        break;
      case fair::Severity::debug3:
        severity = InfoLogger::Severity::Debug;
        level = 18;
        break;
      case fair::Severity::debug4:
        severity = InfoLogger::Severity::Debug;
        level = 19;
        break;
      case fair::Severity::trace:
        severity = InfoLogger::Severity::Debug;
        level = 21;
        break;
      case fair::Severity::nolog: // discard
        return;
    }

    if (logger) {
      logger->log({severity,
                   level,
                   InfoLogger::undefinedMessageOption.errorCode,
                   std::string(metadata.file).c_str(),
                   atoi(std::string(metadata.line).c_str())},
                  *ctx, "%s", content.c_str());
    }
  };
};

struct InfoLoggerPlugin : o2::framework::ServicePlugin {
  o2::framework::ServiceSpec* create() final
  {
    return new ServiceSpec{
      .name = "infologger",
      .init = [](ServiceRegistryRef services, DeviceState&, fair::mq::ProgOptions& options) -> ServiceHandle {
        auto infoLoggerMode = options.GetPropertyAsString("infologger-mode");
        auto infoLoggerSeverity = options.GetPropertyAsString("infologger-severity");
        if (infoLoggerSeverity.empty() == false && options.GetPropertyAsString("infologger-mode") == "") {
          LOGP(info, "Using O2_INFOLOGGER_MODE=infoLoggerD since infologger-severity is set");
          infoLoggerMode = "infoLoggerD";
        }
        if (infoLoggerMode != "") {
          setenv("O2_INFOLOGGER_MODE", infoLoggerMode.c_str(), 1);
        }
        char const* infoLoggerEnv = getenv("O2_INFOLOGGER_MODE");
        if (infoLoggerEnv == nullptr || strcmp(infoLoggerEnv, "none") == 0) {
          return ServiceHandle{.hash = TypeIdHelpers::uniqueId<MissingService>(),
                               .instance = nullptr,
                               .kind = ServiceKind::Serial,
                               .name = "infologger"};
        }
        InfoLogger* infoLoggerService = nullptr;
        try {
          infoLoggerService = new InfoLogger;
        } catch (...) {
          LOGP(error, "Unable to initialise InfoLogger with O2_INFOLOGGER_MODE={}.", infoLoggerMode);
          return ServiceHandle{.hash = TypeIdHelpers::uniqueId<MissingService>(),
                               .instance = nullptr,
                               .kind = ServiceKind::Serial,
                               .name = "infologger"};
        }
        auto infoLoggerContext = &services.get<InfoLoggerContext>();
        // Only print the first 10 characters and the last 18 if the
        // string length is greater than 32 bytes.
        auto truncate = [](std::string in) -> std::string {
          if (in.size() < 32) {
            return in;
          }
          char name[32];
          memcpy(name, in.data(), 10);
          name[10] = '.';
          name[11] = '.';
          name[12] = '.';
          memcpy(name + 13, in.data() + in.size() - 18, 18);
          name[31] = 0;
          return name;
        };
        infoLoggerContext->setField(InfoLoggerContext::FieldName::Facility, truncate(services.get<DeviceSpec const>().name));
        infoLoggerContext->setField(InfoLoggerContext::FieldName::System, std::string("DPL"));
        infoLoggerService->setContext(*infoLoggerContext);

        if (infoLoggerSeverity != "") {
          fair::Logger::AddCustomSink("infologger", infoLoggerSeverity, createInfoLoggerSinkHelper(infoLoggerService, infoLoggerContext));
        }
        return ServiceHandle{.hash = TypeIdHelpers::uniqueId<InfoLogger>(),
                             .instance = infoLoggerService,
                             .kind = ServiceKind::Serial,
                             .name = "infologger"};
      },
      .configure = CommonServices::noConfiguration(),
      .kind = ServiceKind::Serial};
  }
};

DEFINE_DPL_PLUGINS_BEGIN
DEFINE_DPL_PLUGIN_INSTANCE(InfoLoggerContextPlugin, CustomService);
DEFINE_DPL_PLUGIN_INSTANCE(InfoLoggerPlugin, CustomService);
DEFINE_DPL_PLUGINS_END
