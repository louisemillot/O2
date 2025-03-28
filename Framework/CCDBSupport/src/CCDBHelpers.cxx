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

#include "CCDBHelpers.h"
#include "Framework/DeviceSpec.h"
#include "Framework/Logger.h"
#include "Framework/TimingInfo.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/DataTakingContext.h"
#include "Framework/RawDeviceService.h"
#include "Framework/DataSpecUtils.h"
#include "CCDB/CcdbApi.h"
#include "CommonConstants/LHCConstants.h"
#include "Framework/Signpost.h"
#include <typeinfo>
#include <TError.h>
#include <TMemFile.h>
#include <functional>

O2_DECLARE_DYNAMIC_LOG(ccdb);

namespace o2::framework
{

struct CCDBFetcherHelper {
  struct CCDBCacheInfo {
    std::string etag;
    size_t cacheValidUntil = 0;
    size_t cachePopulatedAt = 0;
    size_t cacheMiss = 0;
    size_t cacheHit = 0;
    size_t minSize = -1ULL;
    size_t maxSize = 0;
    int lastCheckedTF = 0;
  };

  struct RemapMatcher {
    std::string path;
  };

  struct RemapTarget {
    std::string url;
  };

  std::unordered_map<std::string, CCDBCacheInfo> mapURL2UUID;
  std::unordered_map<std::string, DataAllocator::CacheId> mapURL2DPLCache;
  std::string createdNotBefore = "0";
  std::string createdNotAfter = "3385078236000";
  std::unordered_map<std::string, o2::ccdb::CcdbApi> apis;
  std::vector<OutputRoute> routes;
  std::unordered_map<std::string, std::string> remappings;
  uint32_t lastCheckedTFCounterOrbReset = 0; // last checkecked TFcounter for bulk check
  int queryPeriodGlo = 1;
  int queryPeriodFactor = 1;
  int64_t timeToleranceMS = 5000;

  o2::ccdb::CcdbApi& getAPI(const std::string& path)
  {
    // find the first = sign in the string. If present drop everything after it
    // and between it and the previous /.
    auto pos = path.find('=');
    if (pos == std::string::npos) {
      auto entry = remappings.find(path);
      return apis[entry == remappings.end() ? "" : entry->second];
    }
    auto pos2 = path.rfind('/', pos);
    if (pos2 == std::string::npos || pos2 == pos - 1 || pos2 == 0) {
      throw runtime_error_f("Malformed path %s", path.c_str());
    }
    auto entry = remappings.find(path.substr(0, pos2));
    return apis[entry == remappings.end() ? "" : entry->second];
  }
};

bool isPrefix(std::string_view prefix, std::string_view full)
{
  return prefix == full.substr(0, prefix.size());
}

CCDBHelpers::ParserResult CCDBHelpers::parseRemappings(char const* str)
{
  std::unordered_map<std::string, std::string> remappings;
  std::string currentUrl = "";

  enum ParsingStates {
    IN_BEGIN,
    IN_BEGIN_URL,
    IN_BEGIN_TARGET,
    IN_END_TARGET,
    IN_END_URL
  };
  ParsingStates state = IN_BEGIN;

  while (true) {
    switch (state) {
      case IN_BEGIN: {
        if (*str == 0) {
          return {remappings, ""};
        }
        state = IN_BEGIN_URL;
      }
      case IN_BEGIN_URL: {
        if ((strncmp("http://", str, 7) != 0) && (strncmp("https://", str, 8) != 0 && (strncmp("file://", str, 7) != 0))) {
          return {remappings, "URL should start with either http:// or https:// or file://"};
        }
        state = IN_END_URL;
      } break;
      case IN_END_URL: {
        char const* c = strchr(str, '=');
        if (c == nullptr) {
          return {remappings, "Expecting at least one target path, missing `='?"};
        }
        if ((c - str) == 0) {
          return {remappings, "Empty url"};
        }
        currentUrl = std::string_view(str, c - str);
        state = IN_BEGIN_TARGET;
        str = c + 1;
      } break;
      case IN_BEGIN_TARGET: {
        if (*str == 0) {
          return {remappings, "Empty target"};
        }
        state = IN_END_TARGET;
      } break;
      case IN_END_TARGET: {
        char const* c = strpbrk(str, ",;");
        if (c == nullptr) {
          if (remappings.count(str)) {
            return {remappings, fmt::format("Path {} requested more than once.", str)};
          }
          remappings[std::string(str)] = currentUrl;
          return {remappings, ""};
        }
        if ((c - str) == 0) {
          return {remappings, "Empty target"};
        }
        auto key = std::string(str, c - str);
        if (remappings.count(str)) {
          return {remappings, fmt::format("Path {} requested more than once.", key)};
        }
        remappings[key] = currentUrl;
        if (*c == ';') {
          state = IN_BEGIN_URL;
        } else {
          state = IN_BEGIN_TARGET;
        }
        str = c + 1;
      } break;
    }
  }
}

auto getOrbitResetTime(o2::pmr::vector<char> const& v) -> Long64_t
{
  Int_t previousErrorLevel = gErrorIgnoreLevel;
  gErrorIgnoreLevel = kFatal;
  TMemFile memFile("name", const_cast<char*>(v.data()), v.size(), "READ");
  gErrorIgnoreLevel = previousErrorLevel;
  if (memFile.IsZombie()) {
    throw runtime_error("CTP is Zombie");
  }
  TClass* tcl = TClass::GetClass(typeid(std::vector<Long64_t>));
  void* result = ccdb::CcdbApi::extractFromTFile(memFile, tcl);
  if (!result) {
    throw runtime_error_f("Couldn't retrieve object corresponding to %s from TFile", tcl->GetName());
  }
  memFile.Close();
  auto* ctp = (std::vector<Long64_t>*)result;
  return (*ctp)[0];
};

bool isOnlineRun(DataTakingContext const& dtc)
{
  return dtc.deploymentMode == DeploymentMode::OnlineAUX || dtc.deploymentMode == DeploymentMode::OnlineDDS || dtc.deploymentMode == DeploymentMode::OnlineECS;
}

auto populateCacheWith(std::shared_ptr<CCDBFetcherHelper> const& helper,
                       int64_t timestamp,
                       TimingInfo& timingInfo,
                       DataTakingContext& dtc,
                       DataAllocator& allocator) -> void
{
  std::string ccdbMetadataPrefix = "ccdb-metadata-";
  int objCnt = -1;
  // We use the timeslice, so that we hook into the same interval as the rest of the
  // callback.
  static bool isOnline = isOnlineRun(dtc);

  auto sid = _o2_signpost_id_t{(int64_t)timingInfo.timeslice};
  O2_SIGNPOST_START(ccdb, sid, "populateCacheWith", "Starting to populate cache with CCDB objects");
  for (auto& route : helper->routes) {
    int64_t timestampToUse = timestamp;
    O2_SIGNPOST_EVENT_EMIT(ccdb, sid, "populateCacheWith", "Fetching object for route %{public}s", DataSpecUtils::describe(route.matcher).data());
    objCnt++;
    auto concrete = DataSpecUtils::asConcreteDataMatcher(route.matcher);
    Output output{concrete.origin, concrete.description, concrete.subSpec};
    auto&& v = allocator.makeVector<char>(output);
    std::map<std::string, std::string> metadata;
    std::map<std::string, std::string> headers;
    std::string path = "";
    std::string etag = "";
    int chRate = helper->queryPeriodGlo;
    bool checkValidity = false;
    for (auto& meta : route.matcher.metadata) {
      if (meta.name == "ccdb-path") {
        path = meta.defaultValue.get<std::string>();
      } else if (meta.name == "ccdb-run-dependent" && meta.defaultValue.get<int>() > 0) {
        if (meta.defaultValue.get<int>() == 1) {
          metadata["runNumber"] = dtc.runNumber;
        } else if (meta.defaultValue.get<int>() == 2) {
          timestampToUse = std::stoi(dtc.runNumber);
        } else {
          LOGP(fatal, "Undefined ccdb-run-dependent option {} for spec {}/{}/{}", meta.defaultValue.get<int>(), concrete.origin.as<std::string>(), concrete.description.as<std::string>(), int(concrete.subSpec));
        }
      } else if (isPrefix(ccdbMetadataPrefix, meta.name)) {
        std::string key = meta.name.substr(ccdbMetadataPrefix.size());
        auto value = meta.defaultValue.get<std::string>();
        O2_SIGNPOST_EVENT_EMIT(ccdb, sid, "populateCacheWith", "Adding metadata %{public}s: %{public}s to the request", key.data(), value.data());
        metadata[key] = value;
      } else if (meta.name == "ccdb-query-rate") {
        chRate = meta.defaultValue.get<int>() * helper->queryPeriodFactor;
      }
    }
    const auto url2uuid = helper->mapURL2UUID.find(path);
    if (url2uuid != helper->mapURL2UUID.end()) {
      etag = url2uuid->second.etag;
      // We check validity every chRate timeslices or if the cache is expired
      uint64_t validUntil = url2uuid->second.cacheValidUntil;
      // When the cache was populated. If the cache was populated after the timestamp, we need to check validity.
      uint64_t cachePopulatedAt = url2uuid->second.cachePopulatedAt;
      // If timestamp is before the time the element was cached or after the claimed validity, we need to check validity, again
      // when online.
      bool cacheExpired = (validUntil <= timestampToUse) || (timestamp < cachePopulatedAt);
      checkValidity = (std::abs(int(timingInfo.tfCounter - url2uuid->second.lastCheckedTF)) >= chRate) && (isOnline || cacheExpired);
    } else {
      checkValidity = true; // never skip check if the cache is empty
    }

    O2_SIGNPOST_EVENT_EMIT(ccdb, sid, "populateCacheWith", "checkValidity is %{public}s for tfID %d of %{public}s", checkValidity ? "true" : "false", timingInfo.tfCounter, path.data());

    const auto& api = helper->getAPI(path);
    if (checkValidity && (!api.isSnapshotMode() || etag.empty())) { // in the snapshot mode the object needs to be fetched only once
      LOGP(detail, "Loading {} for timestamp {}", path, timestampToUse);
      api.loadFileToMemory(v, path, metadata, timestampToUse, &headers, etag, helper->createdNotAfter, helper->createdNotBefore);
      if ((headers.count("Error") != 0) || (etag.empty() && v.empty())) {
        LOGP(fatal, "Unable to find CCDB object {}/{}", path, timestampToUse);
        // FIXME: I should send a dummy message.
        continue;
      }
      // printing in case we find a default entry
      if (headers.find("default") != headers.end()) {
        LOGP(detail, "******** Default entry used for {} ********", path);
      }
      helper->mapURL2UUID[path].lastCheckedTF = timingInfo.tfCounter;
      if (etag.empty()) {
        helper->mapURL2UUID[path].etag = headers["ETag"]; // update uuid
        helper->mapURL2UUID[path].cachePopulatedAt = timestampToUse;
        helper->mapURL2UUID[path].cacheMiss++;
        helper->mapURL2UUID[path].minSize = std::min(v.size(), helper->mapURL2UUID[path].minSize);
        helper->mapURL2UUID[path].maxSize = std::max(v.size(), helper->mapURL2UUID[path].maxSize);
        api.appendFlatHeader(v, headers);
        auto cacheId = allocator.adoptContainer(output, std::move(v), DataAllocator::CacheStrategy::Always, header::gSerializationMethodCCDB);
        helper->mapURL2DPLCache[path] = cacheId;
        O2_SIGNPOST_EVENT_EMIT(ccdb, sid, "populateCacheWith", "Caching %{public}s for %{public}s (DPL id %" PRIu64 ")", path.data(), headers["ETag"].data(), cacheId.value);
        continue;
      }
      if (v.size()) { // but should be overridden by fresh object
        // somewhere here pruneFromCache should be called
        helper->mapURL2UUID[path].etag = headers["ETag"]; // update uuid
        helper->mapURL2UUID[path].cachePopulatedAt = timestampToUse;
        helper->mapURL2UUID[path].cacheValidUntil = headers["Cache-Valid-Until"].empty() ? 0 : std::stoul(headers["Cache-Valid-Until"]);
        helper->mapURL2UUID[path].cacheMiss++;
        helper->mapURL2UUID[path].minSize = std::min(v.size(), helper->mapURL2UUID[path].minSize);
        helper->mapURL2UUID[path].maxSize = std::max(v.size(), helper->mapURL2UUID[path].maxSize);
        api.appendFlatHeader(v, headers);
        auto cacheId = allocator.adoptContainer(output, std::move(v), DataAllocator::CacheStrategy::Always, header::gSerializationMethodCCDB);
        helper->mapURL2DPLCache[path] = cacheId;
        O2_SIGNPOST_EVENT_EMIT(ccdb, sid, "populateCacheWith", "Caching %{public}s for %{public}s (DPL id %" PRIu64 ")", path.data(), headers["ETag"].data(), cacheId.value);
        // one could modify the    adoptContainer to take optional old cacheID to clean:
        // mapURL2DPLCache[URL] = ctx.outputs().adoptContainer(output, std::move(outputBuffer), DataAllocator::CacheStrategy::Always, mapURL2DPLCache[URL]);
        continue;
      } else {
        // Only once the etag is actually used, we get the information on how long the object is valid
        helper->mapURL2UUID[path].cacheValidUntil = headers["Cache-Valid-Until"].empty() ? 0 : std::stoul(headers["Cache-Valid-Until"]);
      }
    }
    // cached object is fine
    auto cacheId = helper->mapURL2DPLCache[path];
    O2_SIGNPOST_EVENT_EMIT(ccdb, sid, "populateCacheWith", "Reusing %{public}s for %{public}s (DPL id %" PRIu64 ")", path.data(), headers["ETag"].data(), cacheId.value);
    helper->mapURL2UUID[path].cacheHit++;
    allocator.adoptFromCache(output, cacheId, header::gSerializationMethodCCDB);
    // the outputBuffer was not used, can we destroy it?
  }
  O2_SIGNPOST_END(ccdb, sid, "populateCacheWith", "Finished populating cache with CCDB objects");
};

AlgorithmSpec CCDBHelpers::fetchFromCCDB()
{
  return adaptStateful([](CallbackService& callbacks, ConfigParamRegistry const& options, DeviceSpec const& spec) {
      std::shared_ptr<CCDBFetcherHelper> helper = std::make_shared<CCDBFetcherHelper>();
      std::unordered_map<std::string, bool> accountedSpecs;
      auto defHost = options.get<std::string>("condition-backend");
      auto checkRate = options.get<int>("condition-tf-per-query");
      auto checkMult = options.get<int>("condition-tf-per-query-multiplier");
      helper->timeToleranceMS = options.get<int64_t>("condition-time-tolerance");
      helper->queryPeriodGlo = checkRate > 0 ? checkRate : std::numeric_limits<int>::max();
      helper->queryPeriodFactor = checkMult > 0 ? checkMult : 1;
      LOGP(info, "CCDB Backend at: {}, validity check for every {} TF{}", defHost, helper->queryPeriodGlo, helper->queryPeriodFactor == 1 ? std::string{} : fmt::format(", (query for high-rate objects downscaled by {})", helper->queryPeriodFactor));
      LOGP(info, "Hook to enable signposts for CCDB messages at {}", (void*)&private_o2_log_ccdb->stacktrace);
      auto remapString = options.get<std::string>("condition-remap");
      ParserResult result = CCDBHelpers::parseRemappings(remapString.c_str());
      if (!result.error.empty()) {
        throw runtime_error_f("Error while parsing remapping string %s", result.error.c_str());
      }
      helper->remappings = result.remappings;
      helper->apis[""].init(defHost); // default backend
      LOGP(info, "Initialised default CCDB host {}", defHost);
      //
      for (auto& entry : helper->remappings) { // init api instances for every host seen in the remapping
        if (helper->apis.find(entry.second) == helper->apis.end()) {
          helper->apis[entry.second].init(entry.second);
          LOGP(info, "Initialised custom CCDB host {}", entry.second);
        }
        LOGP(info, "{} is remapped to {}", entry.first, entry.second);
      }
      helper->createdNotBefore = std::to_string(options.get<int64_t>("condition-not-before"));
      helper->createdNotAfter = std::to_string(options.get<int64_t>("condition-not-after"));

      for (auto &route : spec.outputs) {
        if (route.matcher.lifetime != Lifetime::Condition) {
          continue;
        }
        auto specStr = DataSpecUtils::describe(route.matcher);
        if (accountedSpecs.find(specStr) != accountedSpecs.end()) {
          continue;
        }
        accountedSpecs[specStr] = true;
        helper->routes.push_back(route);
        LOGP(info, "The following route is a condition {}", DataSpecUtils::describe(route.matcher));
        for (auto& metadata : route.matcher.metadata) {
          if (metadata.type == VariantType::String) {
            LOGP(info, "- {}: {}", metadata.name, metadata.defaultValue.asString());
          }
        }
      }
      /// Add a callback on stop which dumps the statistics for the caching per
      /// path
      callbacks.set<CallbackService::Id::Stop>([helper]() {
        LOGP(info, "CCDB cache miss/hit ratio:");
        for (auto& entry : helper->mapURL2UUID) {
          LOGP(info, "  {}: {}/{} ({}-{} bytes)", entry.first, entry.second.cacheMiss, entry.second.cacheHit, entry.second.minSize, entry.second.maxSize);
        }
      });

      return adaptStateless([helper](DataTakingContext& dtc, DataAllocator& allocator, TimingInfo& timingInfo) {
        auto sid = _o2_signpost_id_t{(int64_t)timingInfo.timeslice};
        O2_SIGNPOST_START(ccdb, sid, "fetchFromCCDB", "Fetching CCDB objects for timeslice %" PRIu64, (uint64_t)timingInfo.timeslice);
        static Long64_t orbitResetTime = -1;
        static size_t lastTimeUsed = -1;
        if (timingInfo.creation & DataProcessingHeader::DUMMY_CREATION_TIME_OFFSET) {
          LOGP(info, "Dummy creation time is not supported for CCDB objects. Setting creation to last one used {}.", lastTimeUsed);
          timingInfo.creation = lastTimeUsed;
        }
        lastTimeUsed = timingInfo.creation;
        // Fetch the CCDB object for the CTP
        {
          const std::string path = "CTP/Calib/OrbitReset";
          std::map<std::string, std::string> metadata;
          std::map<std::string, std::string> headers;
          std::string etag;
          bool checkValidity = std::abs(int(timingInfo.tfCounter - helper->lastCheckedTFCounterOrbReset)) >= helper->queryPeriodGlo;
          const auto url2uuid = helper->mapURL2UUID.find(path);
          if (url2uuid != helper->mapURL2UUID.end()) {
            etag = url2uuid->second.etag;
          } else {
            checkValidity = true; // never skip check if the cache is empty
          }
          O2_SIGNPOST_EVENT_EMIT(ccdb, sid, "fetchFromCCDB", "checkValidity is %{public}s for tfID %d of %{public}s",
                                 checkValidity ? "true" : "false", timingInfo.tfCounter, path.data());
          Output output{"CTP", "OrbitReset", 0};
          Long64_t newOrbitResetTime = orbitResetTime;
          auto&& v = allocator.makeVector<char>(output);
          const auto& api = helper->getAPI(path);
          if (checkValidity && (!api.isSnapshotMode() || etag.empty())) { // in the snapshot mode the object needs to be fetched only once
            helper->lastCheckedTFCounterOrbReset = timingInfo.tfCounter;
            api.loadFileToMemory(v, path, metadata, timingInfo.creation, &headers, etag, helper->createdNotAfter, helper->createdNotBefore);
            if ((headers.count("Error") != 0) || (etag.empty() && v.empty())) {
              LOGP(fatal, "Unable to find CCDB object {}/{}", path, timingInfo.creation);
              // FIXME: I should send a dummy message.
              return;
            }
            if (etag.empty()) {
              helper->mapURL2UUID[path].etag = headers["ETag"]; // update uuid
              helper->mapURL2UUID[path].cacheMiss++;
              helper->mapURL2UUID[path].minSize = std::min(v.size(), helper->mapURL2UUID[path].minSize);
              helper->mapURL2UUID[path].maxSize = std::max(v.size(), helper->mapURL2UUID[path].maxSize);
              newOrbitResetTime = getOrbitResetTime(v);
              api.appendFlatHeader(v, headers);
              auto cacheId = allocator.adoptContainer(output, std::move(v), DataAllocator::CacheStrategy::Always, header::gSerializationMethodNone);
              helper->mapURL2DPLCache[path] = cacheId;
              O2_SIGNPOST_EVENT_EMIT(ccdb, sid, "fetchFromCCDB", "Caching %{public}s for %{public}s (DPL id %" PRIu64 ")", path.data(), headers["ETag"].data(), cacheId.value);
            } else if (v.size()) { // but should be overridden by fresh object
              // somewhere here pruneFromCache should be called
              helper->mapURL2UUID[path].etag = headers["ETag"]; // update uuid
              helper->mapURL2UUID[path].cacheMiss++;
              helper->mapURL2UUID[path].minSize = std::min(v.size(), helper->mapURL2UUID[path].minSize);
              helper->mapURL2UUID[path].maxSize = std::max(v.size(), helper->mapURL2UUID[path].maxSize);
              newOrbitResetTime = getOrbitResetTime(v);
              api.appendFlatHeader(v, headers);
              auto cacheId = allocator.adoptContainer(output, std::move(v), DataAllocator::CacheStrategy::Always, header::gSerializationMethodNone);
              helper->mapURL2DPLCache[path] = cacheId;
              O2_SIGNPOST_EVENT_EMIT(ccdb, sid, "fetchFromCCDB", "Caching %{public}s for %{public}s (DPL id %" PRIu64 ")", path.data(), headers["ETag"].data(), cacheId.value);
              // one could modify the adoptContainer to take optional old cacheID to clean:
              // mapURL2DPLCache[URL] = ctx.outputs().adoptContainer(output, std::move(outputBuffer), DataAllocator::CacheStrategy::Always, mapURL2DPLCache[URL]);
            }
            // cached object is fine
          }
          auto cacheId = helper->mapURL2DPLCache[path];
          O2_SIGNPOST_EVENT_EMIT(ccdb, sid, "fetchFromCCDB", "Reusing %{public}s for %{public}s (DPL id %" PRIu64 ")", path.data(), headers["ETag"].data(), cacheId.value);
          helper->mapURL2UUID[path].cacheHit++;
          allocator.adoptFromCache(output, cacheId, header::gSerializationMethodNone);

          if (newOrbitResetTime != orbitResetTime) {
            O2_SIGNPOST_EVENT_EMIT(ccdb, sid, "fetchFromCCDB", "Orbit reset time changed from %lld to %lld", orbitResetTime, newOrbitResetTime);
            orbitResetTime = newOrbitResetTime;
            dtc.orbitResetTimeMUS = orbitResetTime;
          }
        }

        int64_t timestamp = ceil((timingInfo.firstTForbit * o2::constants::lhc::LHCOrbitNS / 1000 + orbitResetTime) / 1000); // RS ceilf precision is not enough
        if (std::abs(int64_t(timingInfo.creation) - timestamp) > helper->timeToleranceMS) {
          static bool notWarnedYet = true;
          if (notWarnedYet) {
            LOGP(warn, "timestamp {} for orbit {} and orbit reset time {} differs by >{} from the TF creation time {}, use the latter", timestamp, timingInfo.firstTForbit, orbitResetTime / 1000, helper->timeToleranceMS, timingInfo.creation);
            notWarnedYet = false;
            // apparently the orbit reset time from the CTP object makes no sense (i.e. orbit was reset for this run w/o create an object, as it happens for technical runs)
            dtc.orbitResetTimeMUS = 1000 * timingInfo.creation - timingInfo.firstTForbit * o2::constants::lhc::LHCOrbitNS / 1000;
          }
          timestamp = timingInfo.creation;
        }
        // Fetch the rest of the objects.
        O2_SIGNPOST_EVENT_EMIT(ccdb, sid, "fetchFromCCDB", "Fetching objects. Run %{public}s. OrbitResetTime %lld. Creation %lld. Timestamp %lld. firstTForbit %" PRIu32,
            dtc.runNumber.data(), orbitResetTime, timingInfo.creation, timestamp, timingInfo.firstTForbit);

        populateCacheWith(helper, timestamp, timingInfo, dtc, allocator);
        O2_SIGNPOST_END(ccdb, _o2_signpost_id_t{(int64_t)timingInfo.timeslice}, "fetchFromCCDB", "Fetching CCDB objects");
      }); });
}

} // namespace o2::framework
