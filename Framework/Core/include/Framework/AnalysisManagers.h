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

#ifndef FRAMEWORK_ANALYSISMANAGERS_H
#define FRAMEWORK_ANALYSISMANAGERS_H
#include "Framework/AnalysisHelpers.h"
#include "Framework/DataSpecUtils.h"
#include "Framework/GroupedCombinations.h"
#include "Framework/ASoA.h"
#include "Framework/ProcessingContext.h"
#include "Framework/EndOfStreamContext.h"
#include "Framework/HistogramRegistry.h"
#include "Framework/CCDBParamSpec.h"
#include "Framework/ConfigParamSpec.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ConfigurableHelpers.h"
#include "Framework/Condition.h"
#include "Framework/InitContext.h"
#include "Framework/RootConfigParamHelpers.h"
#include "Framework/PluginManager.h"
#include "Framework/DeviceSpec.h"

namespace o2::framework
{

namespace
{
template <typename O>
static inline auto extractOriginal(ProcessingContext& pc)
{
  return pc.inputs().get<TableConsumer>(aod::MetadataTrait<O>::metadata::tableLabel())->asArrowTable();
}

template <typename... Os>
static inline std::vector<std::shared_ptr<arrow::Table>> extractOriginals(framework::pack<Os...>, ProcessingContext& pc)
{
  return {extractOriginal<Os>(pc)...};
}

template <size_t N, std::array<soa::TableRef, N> refs>
static inline auto extractOriginals(ProcessingContext& pc)
{
  return [&]<size_t... Is>(std::index_sequence<Is...>) -> std::vector<std::shared_ptr<arrow::Table>> {
    return {pc.inputs().get<TableConsumer>(o2::aod::label<refs[Is]>())->asArrowTable()...};
  }(std::make_index_sequence<refs.size()>());
}
} // namespace

namespace analysis_task_parsers
{

/// Options handling
template <typename O>
bool appendOption(std::vector<ConfigParamSpec>&, O&)
{
  return false;
}

template <is_configurable O>
bool appendOption(std::vector<ConfigParamSpec>& options, O& option)
{
  return ConfigurableHelpers::appendOption(options, option);
}

template <is_configurable_group O>
bool appendOption(std::vector<ConfigParamSpec>& options, O& optionGroup)
{
  if constexpr (requires { optionGroup.prefix; }) {
    homogeneous_apply_refs<true>([prefix = optionGroup.prefix]<typename C>(C& option) { // apend group prefix if set
      if constexpr (requires { option.name; }) {
        option.name.insert(0, 1, '.');
        option.name.insert(0, prefix);
      }
      return true;
    },
                                 optionGroup);
  }
  homogeneous_apply_refs<true>([&options](auto& option) { return appendOption(options, option); }, optionGroup);
  return true;
}

template <typename O>
bool prepareOption(InitContext&, O&)
{
  return false;
}

template <is_configurable O>
bool prepareOption(InitContext& context, O& configurable)
{
  if constexpr (variant_trait_v<typename O::type> != VariantType::Unknown) {
    configurable.value = context.options().get<typename O::type>(configurable.name.c_str());
  } else {
    auto pt = context.options().get<boost::property_tree::ptree>(configurable.name.c_str());
    configurable.value = RootConfigParamHelpers::as<typename O::type>(pt);
  }
  return true;
}

template <is_configurable_group O>
bool prepareOption(InitContext& context, O& configurableGroup)
{
  homogeneous_apply_refs<true>([&context](auto&& configurable) { return prepareOption(context, configurable); }, configurableGroup);
  return true;
}

/// Conditions handling
template <typename C>
bool appendCondition(std::vector<InputSpec>&, C&)
{
  return false;
}

template <is_condition C>
bool appendCondition(std::vector<InputSpec>& inputs, C& condition)
{
  inputs.emplace_back(InputSpec{condition.path, "AODC", runtime_hash(condition.path.c_str()), Lifetime::Condition, ccdbParamSpec(condition.path)});
  return true;
}

template <is_condition_group C>
bool appendCondition(std::vector<InputSpec>& inputs, C& conditionGroup)
{
  homogeneous_apply_refs<true>([&inputs](auto& condition) { return appendCondition(inputs, condition); }, conditionGroup);
  return true;
}

/// Table auto-creation handling
template <typename T>
bool requestInputs(std::vector<InputSpec>&, T const&)
{
  return false;
}

template <is_spawns T>
bool requestInputs(std::vector<InputSpec>& inputs, T const& spawns)
{
  auto base_specs = spawns.base_specs();
  for (auto base_spec : base_specs) {
    base_spec.metadata.push_back(ConfigParamSpec{std::string{"control:spawn"}, VariantType::Bool, true, {"\"\""}});
    DataSpecUtils::updateInputList(inputs, std::forward<InputSpec>(base_spec));
  }
  return true;
}

template <is_builds T>
bool requestInputs(std::vector<InputSpec>& inputs, T const& builds)
{
  auto base_specs = builds.base_specs();
  for (auto base_spec : base_specs) {
    base_spec.metadata.push_back(ConfigParamSpec{std::string{"control:build"}, VariantType::Bool, true, {"\"\""}});
    DataSpecUtils::updateInputList(inputs, std::forward<InputSpec>(base_spec));
  }
  return true;
}

template <typename C>
bool newDataframeCondition(InputRecord&, C&)
{
  return false;
}

template <is_condition C>
bool newDataframeCondition(InputRecord& record, C& condition)
{
  condition.instance = (typename C::type*)record.get<typename C::type*>(condition.path).get();
  return true;
}

template <is_condition_group C>
bool newDataframeCondition(InputRecord& record, C& conditionGroup)
{
  homogeneous_apply_refs<true>([&record](auto&& condition) { return newDataframeCondition(record, condition); }, conditionGroup);
  return true;
}

/// Outputs handling
template <typename T>
bool appendOutput(std::vector<OutputSpec>&, T&, uint32_t)
{
  return false;
}

template <is_produces T>
bool appendOutput(std::vector<OutputSpec>& outputs, T&, uint32_t)
{
  outputs.emplace_back(OutputForTable<typename T::persistent_table_t>::spec());
  return true;
}

template <is_produces_group T>
bool appendOutput(std::vector<OutputSpec>& outputs, T& producesGroup, uint32_t hash)
{
  homogeneous_apply_refs<true>([&outputs, hash](auto& produces) { return appendOutput(outputs, produces, hash); }, producesGroup);
  return true;
}

template <is_histogram_registry T>
bool appendOutput(std::vector<OutputSpec>& outputs, T& hr, uint32_t hash)
{
  hr.setHash(hash);
  outputs.emplace_back(hr.spec());
  return true;
}

template <is_outputobj T>
bool appendOutput(std::vector<OutputSpec>& outputs, T& obj, uint32_t hash)
{
  obj.setHash(hash);
  outputs.emplace_back(obj.spec());
  return true;
}

template <is_spawns T>
bool appendOutput(std::vector<OutputSpec>& outputs, T& spawns, uint32_t)
{
  outputs.emplace_back(spawns.spec());
  return true;
}

template <is_builds T>
bool appendOutput(std::vector<OutputSpec>& outputs, T& builds, uint32_t)
{
  outputs.emplace_back(builds.spec());
  return true;
}

template <typename T>
bool postRunOutput(EndOfStreamContext&, T&)
{
  return false;
}

template <is_histogram_registry T>
bool postRunOutput(EndOfStreamContext& context, T& hr)
{
  auto& deviceSpec = context.services().get<o2::framework::DeviceSpec const>();
  context.outputs().snapshot(hr.ref(deviceSpec.inputTimesliceId, deviceSpec.maxInputTimeslices), *(hr.getListOfHistograms()));
  hr.clean();
  return true;
}

template <is_outputobj T>
bool postRunOutput(EndOfStreamContext& context, T& obj)
{
  auto& deviceSpec = context.services().get<o2::framework::DeviceSpec const>();
  context.outputs().snapshot(obj.ref(deviceSpec.inputTimesliceId, deviceSpec.maxInputTimeslices), *obj);
  return true;
}

template <typename T>
bool prepareOutput(ProcessingContext&, T&)
{
  return false;
}

template <is_produces T>
bool prepareOutput(ProcessingContext& context, T& produces)
{
  produces.resetCursor(std::move(context.outputs().make<TableBuilder>(OutputForTable<typename T::persistent_table_t>::ref())));
  return true;
}

template <is_produces_group T>
bool prepareOutput(ProcessingContext& context, T& producesGroup)
{
  homogeneous_apply_refs<true>([&context](auto& produces) { return prepareOutput(context, produces); }, producesGroup);
  return true;
}

template <is_spawns T>
bool prepareOutput(ProcessingContext& context, T& spawns)
{
  using metadata = o2::aod::MetadataTrait<o2::aod::Hash<T::spawnable_t::ref.desc_hash>>::metadata;
  auto originalTable = soa::ArrowHelpers::joinTables(extractOriginals<metadata::sources.size(), metadata::sources>(context));
  if (originalTable->schema()->fields().empty() == true) {
    using base_table_t = typename T::base_table_t::table_t;
    originalTable = makeEmptyTable<base_table_t>(o2::aod::label<metadata::extension_table_t::ref>());
  }

  spawns.extension = std::make_shared<typename T::extension_t>(o2::framework::spawner<o2::aod::Hash<metadata::extension_table_t::ref.desc_hash>>(originalTable, o2::aod::label<metadata::extension_table_t::ref>()));
  spawns.table = std::make_shared<typename T::spawnable_t::table_t>(soa::ArrowHelpers::joinTables({spawns.extension->asArrowTable(), originalTable}));
  return true;
}

template <is_builds T>
bool prepareOuput(ProcessingContext& context, T& builds)
{
  using metadata = o2::aod::MetadataTrait<o2::aod::Hash<T::buildable_t::ref.desc_hash>>::metadata;
  return builds.template build<typename T::buildable_t::indexing_t>(builds.pack(), extractOriginals<metadata::sources.size(), metadata::sources>(context));
}

template <typename T>
bool finalizeOutput(ProcessingContext&, T&)
{
  return false;
}

template <is_produces T>
bool finalizeOutput(ProcessingContext&, T& produces)
{
  produces.setLabel(o2::aod::label<T::persistent_table_t::ref>());
  produces.release();
  return true;
}

template <is_produces_group T>
bool finalizeOutput(ProcessingContext& context, T& producesGroup)
{
  homogeneous_apply_refs<true>([&context](auto& produces) { return finalizeOutput(context, produces); }, producesGroup);
  return true;
}

template <is_spawns T>
bool finalizeOutput(ProcessingContext& context, T& spawns)
{
  context.outputs().adopt(spawns.output(), spawns.asArrowTable());
  return true;
}

template <is_builds T>
bool finalizeOutput(ProcessingContext& context, T& builds)
{
  context.outputs().adopt(builds.output(), builds.asArrowTable());
  return true;
}

/// Service handling
template <typename T>
bool addService(std::vector<ServiceSpec>&, T&)
{
  return false;
}

template <is_service T>
bool addService(std::vector<ServiceSpec>& specs, T&)
{
  if constexpr (o2::framework::base_of_template<LoadableServicePlugin, typename T::service_t>) {
    auto p = typename T::service_t{};
    auto loadableServices = PluginManager::parsePluginSpecString(p.loadSpec.c_str());
    PluginManager::loadFromPlugin<ServiceSpec, ServicePlugin>(loadableServices, specs);
  }
  return true;
}

template <typename T>
bool prepareService(InitContext&, T&)
{
  return false;
}

template <is_service T>
bool prepareService(InitContext& context, T& service)
{
  using S = typename T::service_t;
  if constexpr (requires { &S::instance; }) {
    service.service = &(S::instance()); // Sigh...
    return true;
  } else {
    service.service = &(context.services().get<S>());
    return true;
  }
  return false;
}

template <typename T>
bool postRunService(EndOfStreamContext&, T&)
{
  return false;
}

template <is_service T>
bool postRunService(EndOfStreamContext&, T& service)
{
  // FIXME: for the moment we only need endOfStream to be
  // stateless. In the future we might want to pass it EndOfStreamContext
  if constexpr (requires { &T::service_t::endOfStream; }) {
    service.service->endOfStream();
    return true;
  }
  return false;
}

/// Filter handling
template <typename T>
bool updatePlaceholders(InitContext&, T&)
{
  return false;
}

template <expressions::is_filter T>
bool updatePlaceholders(InitContext& context, T& filter)
{
  expressions::updatePlaceholders(filter, context);
  return true;
}

template <is_partition T>
bool updatePlaceholders(InitContext& context, T& partition)
{
  partition.updatePlaceholders(context);
  return true;
}

template <typename T>
bool createExpressionTrees(std::vector<ExpressionInfo>&, T&)
{
  return false;
}

template <expressions::is_filter T>
bool createExpressionTrees(std::vector<ExpressionInfo>& expressionInfos, T& filter)
{
  expressions::updateExpressionInfos(filter, expressionInfos);
  return true;
}

template <typename T>
bool newDataframePartition(T&)
{
  return false;
}

template <is_partition T>
bool newDataframePartition(T& partition)
{
  partition.dataframeChanged = true;
  return true;
}

template <typename P, typename... T>
void setPartition(P&, T&...)
{
}

template <is_partition P, typename... T>
void setPartition(P& partition, T&... tables)
{
  ([&]() { if constexpr (std::same_as<typename P::content_t, T>) {partition.bindTable(tables);} }(), ...);
}

template <typename P, typename T>
void bindInternalIndicesPartition(P&, T*)
{
}

template <is_partition P, typename T>
void bindInternalIndicesPartition(P& partition, T* table)
{
  if constexpr (o2::soa::is_binding_compatible_v<typename P::content_t, std::decay_t<T>>()) {
    partition.bindInternalIndicesTo(table);
  }
}

template <typename P, typename... T>
void bindExternalIndicesPartition(P&, T*...)
{
}

template <is_partition P, typename... T>
void bindExternalIndicesPartition(P& partition, T*... tables)
{
  partition.bindExternalIndices(tables...);
}

/// Cache handling
template <typename T>
bool preInitializeCache(InitContext&, T&)
{
  return false;
}

template <typename T>
bool initializeCache(ProcessingContext&, T&)
{
  return false;
}

template <is_slice_cache T>
bool initializeCache(ProcessingContext& context, T& cache)
{
  if (cache.ptr == nullptr) {
    cache.ptr = &context.services().get<ArrowTableSlicingCache>();
  }
  return true;
}

/// Combinations handling
template <typename C, typename TG, typename... Ts>
  requires(!is_combinations_generator<C>)
void setGroupedCombination(C&, TG&, Ts&...)
{
}

template <is_combinations_generator C, typename TG, typename... Ts>
  requires((sizeof...(Ts) > 0) && (C::compatible(framework::pack<Ts...>{})))
static void setGroupedCombination(C& comb, TG& grouping, std::tuple<Ts...>& associated)
{
  if constexpr (std::same_as<typename C::g_t, TG>) {
    comb.setTables(grouping, associated);
  }
}

/// Preslice handling
template <typename T>
  requires(!is_preslice<T>)
bool registerCache(T&, std::vector<StringPair>&, std::vector<StringPair>&)
{
  return false;
}

template <is_preslice T>
  requires std::same_as<typename T::policy_t, framework::PreslicePolicySorted>
bool registerCache(T& preslice, std::vector<StringPair>& bsks, std::vector<StringPair>&)
{
  if constexpr (T::optional) {
    if (preslice.binding == "[MISSING]") {
      return true;
    }
  }
  auto locate = std::find_if(bsks.begin(), bsks.end(), [&](auto const& entry) { return (entry.first == preslice.bindingKey.first) && (entry.second == preslice.bindingKey.second); });
  if (locate == bsks.end()) {
    bsks.emplace_back(preslice.getBindingKey());
  }
  return true;
}

template <is_preslice T>
  requires std::same_as<typename T::policy_t, framework::PreslicePolicyGeneral>
bool registerCache(T& preslice, std::vector<StringPair>&, std::vector<StringPair>& bsksU)
{
  if constexpr (T::optional) {
    if (preslice.binding == "[MISSING]") {
      return true;
    }
  }
  auto locate = std::find_if(bsksU.begin(), bsksU.end(), [&](auto const& entry) { return (entry.first == preslice.bindingKey.first) && (entry.second == preslice.bindingKey.second); });
  if (locate == bsksU.end()) {
    bsksU.emplace_back(preslice.getBindingKey());
  }
  return true;
}

template <typename T>
  requires(!is_preslice<T>)
bool updateSliceInfo(T&, ArrowTableSlicingCache&)
{
  return false;
}

template <is_preslice T>
static bool updateSliceInfo(T& preslice, ArrowTableSlicingCache& cache)
  requires std::same_as<typename T::policy_t, framework::PreslicePolicySorted>
{
  if constexpr (T::optional) {
    if (preslice.binding == "[MISSING]") {
      return true;
    }
  }
  preslice.updateSliceInfo(cache.getCacheFor(preslice.getBindingKey()));
  return true;
}

template <is_preslice T>
static bool updateSliceInfo(T& preslice, ArrowTableSlicingCache& cache)
  requires std::same_as<typename T::policy_t, framework::PreslicePolicyGeneral>
{
  if constexpr (T::optional) {
    if (preslice.binding == "[MISSING]") {
      return true;
    }
  }
  preslice.updateSliceInfo(cache.getCacheUnsortedFor(preslice.getBindingKey()));
  return true;
}

/// Process switches handling
template <typename T>
static bool setProcessSwitch(std::pair<std::string, bool>, T&)
{
  return false;
}

template <is_process_configurable T>
static bool setProcessSwitch(std::pair<std::string, bool> setting, T& pc)
{
  if (pc.name == setting.first) {
    pc.value = setting.second;
    return true;
  }
  return false;
}

} // namespace analysis_task_parsers
} // namespace o2::framework

#endif // ANALYSISMANAGERS_H
