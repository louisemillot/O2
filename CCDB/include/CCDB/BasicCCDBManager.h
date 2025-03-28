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
// Created by Sandro Wenzel on 2019-08-14.
//

#ifndef O2_BASICCDBMANAGER_H
#define O2_BASICCDBMANAGER_H

#include "CCDB/CcdbApi.h"
#include "CCDB/CCDBTimeStampUtils.h"
#include "CommonUtils/NameConf.h"
#include "Framework/DataTakingContext.h"
#include "Framework/DefaultsHelpers.h"
#include <string>
#include <chrono>
#include <map>
#include <unordered_map>
#include <memory>
#include <cstdlib>

class TGeoManager; // we need to forward-declare those classes which should not be cleaned up

namespace o2::ccdb
{

/// A simple class offering simplified access to CCDB (mainly for MC simulation)
/// The class encapsulates timestamp and URL and is easily usable from detector code.
///
/// The CDBManager allowing caching of retrieved objects is by definition not thread safe,
/// therefore, to provide a possibility of multithread processing, one should foresee possibility
/// of multiple instances of the manager. CCDBManagerInstance serves to this purpose
///
/// In cases where caching is not needed or just 1 instance of the manager is enough, one case use
/// a singleton version BasicCCDBManager

class CCDBManagerInstance
{
  struct CachedObject {
    std::shared_ptr<void> objPtr;
    void* noCleanupPtr = nullptr; // if assigned instead of objPtr, no cleanup will be done on exit (for global objects cleaned up by the root, e.g. gGeoManager)
    std::string uuid;
    long startvalidity = 0;
    long endvalidity = -1;
    long cacheValidFrom = 0;   // time for which the object was cached
    long cacheValidUntil = -1; // object is guaranteed to be valid till this time (modulo new updates)
    size_t minSize = -1ULL;
    size_t maxSize = 0;
    int queries = 0;
    int fetches = 0;
    int failures = 0;
    bool isValid(long ts) { return ts < endvalidity && ts >= startvalidity; }
    bool isCacheValid(long ts)
    {
      LOGP(debug, "isCacheValid : {} : {} : {} --> {}", cacheValidFrom, ts, cacheValidUntil, ts < cacheValidUntil && ts >= cacheValidFrom);
      return ts < cacheValidUntil && ts >= cacheValidFrom;
    }
    void clear()
    {
      noCleanupPtr = nullptr;
      objPtr.reset();
      uuid = "";
      startvalidity = 0;
      endvalidity = -1;
    }
  };

 public:
  using MD = std::map<std::string, std::string>;

  CCDBManagerInstance(std::string const& path) : mCCDBAccessor{}
  {
    mCCDBAccessor.init(path);
    mDeplMode = o2::framework::DefaultsHelpers::deploymentMode();
  }
  /// set a URL to query from
  void setURL(const std::string& url);

  /// set timestamp cache for all queries
  void setTimestamp(long t)
  {
    if (t >= 0) {
      mTimestamp = t;
    }
  }

  /// query current URL
  std::string const& getURL() const { return mCCDBAccessor.getURL(); }

  /// query timestamp
  long getTimestamp() const { return mTimestamp; }

  /// retrieve an object of type T from CCDB as stored under path and timestamp
  template <typename T>
  T* getForTimeStamp(std::string const& path, long timestamp);

  /// retrieve an object of type T from CCDB as stored under path and using the timestamp in the middle of the run
  template <typename T>
  T* getForRun(std::string const& path, int runNumber, bool setRunMetadata = false);

  /// retrieve an object of type T from CCDB as stored under path, timestamp and metaData
  template <typename T>
  T* getSpecific(std::string const& path, long timestamp = -1, MD metaData = MD())
  {
    // TODO: add some error info/handling when failing
    mMetaData = metaData;
    return getForTimeStamp<T>(path, timestamp);
  }

  /// retrieve an object of type T from CCDB as stored under path and using the timestamp in the middle of the run + metadata. The run number is provided separately to conform to typical analysis use (in which case metadata does not include runNumber)
  template <typename T>
  T* getSpecificForRun(std::string const& path, int runNumber, MD metaData = MD());

  /// detect online processing modes (i.e. CCDB objects may be updated in the lifetime of the manager)
  bool isOnline() const { return mDeplMode == o2::framework::DeploymentMode::OnlineAUX || mDeplMode == o2::framework::DeploymentMode::OnlineDDS || mDeplMode == o2::framework::DeploymentMode::OnlineECS; }

  /// retrieve an object of type T from CCDB as stored under path; will use the timestamp member
  template <typename T>
  T* get(std::string const& path)
  {
    return getForTimeStamp<T>(path, mTimestamp);
  }

  bool isHostReachable() const { return mCCDBAccessor.isHostReachable(); }

  /// clear all entries in the cache
  void clearCache() { mCache.clear(); }

  /// clear particular entry in the cache
  void clearCache(std::string const& path) { mCache.erase(path); }

  /// check if caching is enabled
  bool isCachingEnabled() const { return mCachingEnabled; }

  /// disable or enable caching
  void setCaching(bool v)
  {
    mCachingEnabled = v;
    if (!v) {
      clearCache();
    }
  }

  /// Check if an object in cache is valid
  bool isCachedObjectValid(std::string const& path, long timestamp)
  {
    if (!isCachingEnabled()) {
      return false;
    }
    return (mCheckObjValidityEnabled && mCache[path].isValid(timestamp)) || mCache[path].isCacheValid(timestamp); // use stricter check
  }

  /// check if checks of object validity before CCDB query is enabled
  bool isLocalObjectValidityCheckingEnabled() const { return mCheckObjValidityEnabled; }

  /// set the flag to check object validity before CCDB query
  void setLocalObjectValidityChecking(bool v = true) { mCheckObjValidityEnabled = v; }

  /// set the object upper validity limit
  void setCreatedNotAfter(long v) { mCreatedNotAfter = v; }

  /// get the object upper validity limit
  long getCreatedNotAfter() const { return mCreatedNotAfter; }

  /// reset the object upper validity limit
  void resetCreatedNotAfter() { mCreatedNotAfter = 0; }

  /// set the object upper validity limit
  void setCreatedNotBefore(long v) { mCreatedNotBefore = v; }

  /// get the object upper validity limit
  long getCreatedNotBefore() const { return mCreatedNotBefore; }

  /// reset the object upper validity limit
  void resetCreatedNotBefore() { mCreatedNotBefore = 0; }

  /// get the fatalWhenNull state
  bool getFatalWhenNull() const { return mFatalWhenNull; }
  /// set the fatal property (when false; nullptr object responses will not abort)
  void setFatalWhenNull(bool b) { mFatalWhenNull = b; }

  /// A convenience function for MC to fetch
  /// valid start and end timestamps for recorded TF data given an ALICE run number.
  /// In absence of STF/ETF fields in the RCT with fall back to CTP SOX/EOX then to
  /// ECS SOR/EOR.
  /// On error it fatals (if fatal == true) or else returns the pair -1, -1.
  std::pair<int64_t, int64_t> getRunDuration(int runnumber, bool fatal = true);
  static std::pair<int64_t, int64_t> getRunDuration(o2::ccdb::CcdbApi const& api, int runnumber, bool fatal = true);
  static std::pair<int64_t, int64_t> getRunDuration(const MD& headers);
  std::string getSummaryString() const;

  size_t getFetchedSize() const { return mFetchedSize; }

  void report(bool longrep = false);

  void endOfStream();

 private:
  // method to print (fatal) error
  void reportFatal(std::string_view s);
  // we access the CCDB via the CURL based C++ API
  o2::ccdb::CcdbApi mCCDBAccessor;
  std::unordered_map<std::string, CachedObject> mCache; //! map for {path, CachedObject} associations
  MD mMetaData;                                         // some dummy object needed to talk to CCDB API
  MD mHeaders;                                          // headers to retrieve tags
  long mTimestamp{o2::ccdb::getCurrentTimestamp()};     // timestamp to be used for query (by default "now")
  bool mCanDefault = false;                             // whether default is ok --> useful for testing purposes done standalone/isolation
  bool mCachingEnabled = true;                          // whether caching is enabled
  bool mCheckObjValidityEnabled = false;                // wether the validity of cached object is checked before proceeding to a CCDB API query
  bool mFatalWhenNull = true;                           // if nullptr blob replies should be treated as fatal (can be set by user)
  long mCreatedNotAfter = 0;                            // upper limit for object creation timestamp (TimeMachine mode) - If-Not-After HTTP header
  long mCreatedNotBefore = 0;                           // lower limit for object creation timestamp (TimeMachine mode) - If-Not-Before HTTP header
  long mTimerMS = 0;                                    // timer for queries
  size_t mFetchedSize = 0;                              // total fetched size
  int mQueries = 0;                                     // total number of object queries
  int mFetches = 0;                                     // total number of succesful fetches from CCDB
  int mFailures = 0;                                    // total number of failed fetches
  o2::framework::DeploymentMode mDeplMode;              // O2 deployment mode
  ClassDefNV(CCDBManagerInstance, 1);
};

template <typename T>
T* CCDBManagerInstance::getForTimeStamp(std::string const& path, long timestamp)
{
  T* ptr = nullptr;
  mQueries++;
  auto start = std::chrono::system_clock::now();
  if (!isCachingEnabled()) {
    ptr = mCCDBAccessor.retrieveFromTFileAny<T>(path, mMetaData, timestamp, nullptr, "",
                                                mCreatedNotAfter ? std::to_string(mCreatedNotAfter) : "",
                                                mCreatedNotBefore ? std::to_string(mCreatedNotBefore) : "");
    if (!ptr) {
      if (mFatalWhenNull) {
        reportFatal(std::string("Got nullptr from CCDB for path ") + path + std::string(" and timestamp ") + std::to_string(timestamp));
      }
      mFailures++;
    } else {
      mFetches++;
      auto sh = mHeaders.find("fileSize");
      if (sh != mHeaders.end()) {
        size_t s = atol(sh->second.c_str());
        mFetchedSize += s;
      }
    }
  } else {
    auto& cached = mCache[path];
    cached.queries++;
    if ((!isOnline() && cached.isCacheValid(timestamp)) || (mCheckObjValidityEnabled && cached.isValid(timestamp))) {
      return reinterpret_cast<T*>(cached.noCleanupPtr ? cached.noCleanupPtr : cached.objPtr.get());
    }
    ptr = mCCDBAccessor.retrieveFromTFileAny<T>(path, mMetaData, timestamp, &mHeaders, cached.uuid,
                                                mCreatedNotAfter ? std::to_string(mCreatedNotAfter) : "",
                                                mCreatedNotBefore ? std::to_string(mCreatedNotBefore) : "");
    if (ptr) { // new object was shipped, old one (if any) is not valid anymore
      cached.fetches++;
      mFetches++;
      if constexpr (std::is_same<TGeoManager, T>::value || std::is_base_of<o2::conf::ConfigurableParam, T>::value) {
        // some special objects cannot be cached to shared_ptr since root may delete their raw global pointer
        cached.noCleanupPtr = ptr;
      } else {
        cached.objPtr.reset(ptr);
      }
      cached.uuid = mHeaders["ETag"];

      try {
        if (mHeaders.find("Valid-From") != mHeaders.end()) {
          cached.startvalidity = std::stol(mHeaders["Valid-From"]);
        } else {
          // if meta-information missing assume infinit validity
          // (should happen only for locally created objects)
          cached.startvalidity = 0;
        }
        if (mHeaders.find("Valid-Until") != mHeaders.end()) {
          cached.endvalidity = std::stol(mHeaders["Valid-Until"]);
        } else {
          cached.endvalidity = std::numeric_limits<long>::max();
        }
        cached.cacheValidFrom = timestamp;
      } catch (std::exception const& e) {
        reportFatal("Failed to read validity from CCDB response (Valid-From :  " + mHeaders["Valid-From"] + std::string(" Valid-Until: ") + mHeaders["Valid-Until"] + std::string(")"));
      }
      auto sh = mHeaders.find("fileSize");
      if (sh != mHeaders.end()) {
        size_t s = atol(sh->second.c_str());
        mFetchedSize += s;
        cached.minSize = std::min(s, cached.minSize);
        cached.maxSize = std::max(s, cached.minSize);
      }
    } else if (mHeaders.count("Error")) { // in case of errors the pointer is 0 and headers["Error"] should be set
      cached.failures++;
      cached.clear(); // in case of any error clear cache for this object
    }
    // the old object is valid, fetch cache end of validity
    ptr = reinterpret_cast<T*>(cached.noCleanupPtr ? cached.noCleanupPtr : cached.objPtr.get());
    if (mHeaders.find("Cache-Valid-Until") != mHeaders.end()) {
      cached.cacheValidUntil = std::stol(mHeaders["Cache-Valid-Until"]);
    } else {
      cached.cacheValidUntil = -1;
    }
    mHeaders.clear();
    mMetaData.clear();
    if (!ptr) {
      if (mFatalWhenNull) {
        reportFatal(std::string("Got nullptr from CCDB for path ") + path + std::string(" and timestamp ") + std::to_string(timestamp));
      }
      mFailures++;
    }
  }
  auto end = std::chrono::system_clock::now();
  mTimerMS += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  return ptr;
}

template <typename T>
T* CCDBManagerInstance::getForRun(std::string const& path, int runNumber, bool setRunMetadata)
{
  auto metaData = setRunMetadata ? MD{{"runNumber", std::to_string(runNumber)}} : MD{};
  mMetaData = metaData;
  return getSpecificForRun<T>(path, runNumber, metaData);
}

template <typename T>
T* CCDBManagerInstance::getSpecificForRun(std::string const& path, int runNumber, MD metaData)
{
  auto [start, stop] = getRunDuration(runNumber);
  if (start < 0 || stop < 0) {
    if (mFatalWhenNull) {
      reportFatal(std::string("Failed to get run duration for run ") + std::to_string(runNumber) + std::string(" from CCDB"));
    }
    return nullptr;
  }
  return getSpecific<T>(path, start / 2 + stop / 2, metaData);
}

class BasicCCDBManager : public CCDBManagerInstance
{
 public:
  static BasicCCDBManager& instance()
  {
    const std::string ccdbUrl{o2::base::NameConf::getCCDBServer()};
    static BasicCCDBManager inst{ccdbUrl};
    return inst;
  }

 private:
  using CCDBManagerInstance::CCDBManagerInstance;
  BasicCCDBManager(std::string const& url) : CCDBManagerInstance(url)
  {
    const char* t = getenv("ALICEO2_CCDB_CONDITION_NOT_AFTER");
    if (t) {
      auto timeaslong = strtol(t, nullptr, 10);
      if (timeaslong != 0L) {
        LOG(info) << "CCDB Time-machine constrained detected. Setting condition-not-after constrained to timestamp " << timeaslong;
        setCreatedNotAfter(timeaslong);
      }
    }
  }
};

} // namespace o2::ccdb

#endif //O2_BASICCCDBMANAGER_H
