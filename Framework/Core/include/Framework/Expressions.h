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
#ifndef O2_FRAMEWORK_EXPRESSIONS_H_
#define O2_FRAMEWORK_EXPRESSIONS_H_

#include "Framework/BasicOps.h"
#include "Framework/CompilerBuiltins.h"
#include "Framework/Pack.h"
#include "Framework/Configurable.h"
#include "Framework/Variant.h"
#include "Framework/InitContext.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/RuntimeError.h"
#include <arrow/type_fwd.h>
#include <gandiva/gandiva_aliases.h>
#include <arrow/type.h>
#include <gandiva/arrow.h>
#if !defined(__CLING__) && !defined(__ROOTCLING__)
#include <arrow/table.h>
#include <gandiva/selection_vector.h>
#include <gandiva/node.h>
#include <gandiva/filter.h>
#include <gandiva/projector.h>
#else
namespace gandiva
{
class SelectionVector;
class Filter;
class Projector;
} // namespace gandiva
#endif
#include <variant>
#include <string>
#include <memory>
#include <set>
namespace gandiva
{
using Selection = std::shared_ptr<gandiva::SelectionVector>;
using FilterPtr = std::shared_ptr<gandiva::Filter>;
} // namespace gandiva

using atype = arrow::Type;
struct ExpressionInfo {
  ExpressionInfo(int ai, size_t hash, std::set<uint32_t>&& hs, gandiva::SchemaPtr sc)
    : argumentIndex(ai),
      processHash(hash),
      hashes(hs),
      schema(sc)
  {
  }
  int argumentIndex;
  size_t processHash;
  std::set<uint32_t> hashes;
  gandiva::SchemaPtr schema;
  gandiva::NodePtr tree = nullptr;
  gandiva::FilterPtr filter = nullptr;
  gandiva::Selection selection = nullptr;
  bool resetSelection = false;
};

namespace o2::framework::expressions
{
const char* stringType(atype::type t);

template <typename... T>
struct LiteralStorage {
  using stored_type = std::variant<T...>;
  using stored_pack = framework::pack<T...>;
};

using LiteralValue = LiteralStorage<int, bool, float, double, uint8_t, int64_t, int16_t, uint16_t, int8_t, uint32_t, uint64_t>;

template <typename T>
constexpr auto selectArrowType()
{
  return atype::NA;
}

#define SELECT_ARROW_TYPE(_Ctype_, _Atype_) \
  template <typename T>                     \
    requires std::same_as<T, _Ctype_>       \
  constexpr auto selectArrowType()          \
  {                                         \
    return atype::_Atype_;                  \
  }

SELECT_ARROW_TYPE(bool, BOOL);
SELECT_ARROW_TYPE(float, FLOAT);
SELECT_ARROW_TYPE(double, DOUBLE);
SELECT_ARROW_TYPE(uint8_t, UINT8);
SELECT_ARROW_TYPE(int8_t, INT8);
SELECT_ARROW_TYPE(uint16_t, UINT16);
SELECT_ARROW_TYPE(int16_t, INT16);
SELECT_ARROW_TYPE(uint32_t, UINT32);
SELECT_ARROW_TYPE(int32_t, INT32);
SELECT_ARROW_TYPE(uint64_t, UINT64);
SELECT_ARROW_TYPE(int64_t, INT64);

std::shared_ptr<arrow::DataType> concreteArrowType(atype::type type);
std::string upcastTo(atype::type f);

/// An expression tree node corresponding to a literal value
struct LiteralNode {
  template <typename T>
  LiteralNode(T v) : value{v}, type{selectArrowType<T>()}
  {
  }

  using var_t = LiteralValue::stored_type;
  var_t value;
  atype::type type = atype::NA;
};

/// An expression tree node corresponding to a column binding
struct BindingNode {
  BindingNode(BindingNode const&) = default;
  BindingNode(BindingNode&&) = delete;
  constexpr BindingNode(const char* name_, uint32_t hash_, atype::type type_) : name{name_}, hash{hash_}, type{type_} {}
  const char* name;
  uint32_t hash;
  atype::type type;
};

/// An expression tree node corresponding to binary or unary operation
struct OpNode {
  OpNode(BasicOp op_) : op{op_} {}
  BasicOp op;
};

/// A placeholder node for simple type configurable
struct PlaceholderNode : LiteralNode {
  template <typename T>
  PlaceholderNode(Configurable<T> const& v) : LiteralNode{v.value}, name{v.name}
  {
    if constexpr (variant_trait_v<typename std::decay<T>::type> != VariantType::Unknown) {
      retrieve = [](InitContext& context, char const* name) { return LiteralNode::var_t{context.options().get<T>(name)}; };
    } else {
      runtime_error("Unknown parameter used in expression.");
    }
  }

  void reset(InitContext& context)
  {
    value = retrieve(context, name.data());
  }

  std::string const& name;
  LiteralNode::var_t (*retrieve)(InitContext&, char const*);
};

/// A conditional node
struct ConditionalNode {
};

/// A generic tree node
struct Node {
  Node(LiteralNode&& v) : self{std::forward<LiteralNode>(v)}, left{nullptr}, right{nullptr}, condition{nullptr}
  {
  }

  Node(PlaceholderNode&& v) : self{std::forward<PlaceholderNode>(v)}, left{nullptr}, right{nullptr}, condition{nullptr}
  {
  }

  Node(Node&& n) : self{std::forward<self_t>(n.self)}, left{std::forward<std::unique_ptr<Node>>(n.left)}, right{std::forward<std::unique_ptr<Node>>(n.right)}, condition{std::forward<std::unique_ptr<Node>>(n.condition)}
  {
  }

  Node(BindingNode const& n) : self{n}, left{nullptr}, right{nullptr}, condition{nullptr}
  {
  }

  Node(ConditionalNode op, Node&& then_, Node&& else_, Node&& condition_)
    : self{op},
      left{std::make_unique<Node>(std::forward<Node>(then_))},
      right{std::make_unique<Node>(std::forward<Node>(else_))},
      condition{std::make_unique<Node>(std::forward<Node>(condition_))} {}

  Node(OpNode op, Node&& l, Node&& r)
    : self{op},
      left{std::make_unique<Node>(std::forward<Node>(l))},
      right{std::make_unique<Node>(std::forward<Node>(r))},
      condition{nullptr} {}

  Node(OpNode op, Node&& l)
    : self{op},
      left{std::make_unique<Node>(std::forward<Node>(l))},
      right{nullptr},
      condition{nullptr} {}

  /// variant with possible nodes
  using self_t = std::variant<LiteralNode, BindingNode, OpNode, PlaceholderNode, ConditionalNode>;
  self_t self;
  size_t index = 0;
  /// pointers to children
  std::unique_ptr<Node> left;
  std::unique_ptr<Node> right;
  std::unique_ptr<Node> condition;
};

/// overloaded operators to build the tree from an expression

#define BINARY_OP_NODES(_operator_, _operation_)                                                        \
  inline Node operator _operator_(Node&& left, Node&& right)                                            \
  {                                                                                                     \
    return Node{OpNode{BasicOp::_operation_}, std::forward<Node>(left), std::forward<Node>(right)};     \
  }                                                                                                     \
  template <typename T>                                                                                 \
  inline Node operator _operator_(Node&& left, T right) requires(std::is_arithmetic_v<std::decay_t<T>>) \
  {                                                                                                     \
    return Node{OpNode{BasicOp::_operation_}, std::forward<Node>(left), LiteralNode{right}};            \
  }                                                                                                     \
  template <typename T>                                                                                 \
  inline Node operator _operator_(T left, Node&& right) requires(std::is_arithmetic_v<std::decay_t<T>>) \
  {                                                                                                     \
    return Node{OpNode{BasicOp::_operation_}, LiteralNode{left}, std::forward<Node>(right)};            \
  }                                                                                                     \
  template <typename T>                                                                                 \
  inline Node operator _operator_(Node&& left, Configurable<T> const& right)                            \
  {                                                                                                     \
    return Node{OpNode{BasicOp::_operation_}, std::forward<Node>(left), PlaceholderNode{right}};        \
  }                                                                                                     \
  template <typename T>                                                                                 \
  inline Node operator _operator_(Configurable<T> const& left, Node&& right)                            \
  {                                                                                                     \
    return Node{OpNode{BasicOp::_operation_}, PlaceholderNode{left}, std::forward<Node>(right)};        \
  }                                                                                                     \
  inline Node operator _operator_(BindingNode const& left, BindingNode const& right)                    \
  {                                                                                                     \
    return Node{OpNode{BasicOp::_operation_}, left, right};                                             \
  }                                                                                                     \
  inline Node operator _operator_(BindingNode const& left, Node&& right)                                \
  {                                                                                                     \
    return Node{OpNode{BasicOp::_operation_}, left, std::forward<Node>(right)};                         \
  }                                                                                                     \
  inline Node operator _operator_(Node&& left, BindingNode const& right)                                \
  {                                                                                                     \
    return Node{OpNode{BasicOp::_operation_}, std::forward<Node>(left), right};                         \
  }                                                                                                     \
  template <typename T>                                                                                 \
  inline Node operator _operator_(Configurable<T> const& left, BindingNode const& right)                \
  {                                                                                                     \
    return Node{OpNode{BasicOp::_operation_}, PlaceholderNode{left}, right};                            \
  }                                                                                                     \
  template <typename T>                                                                                 \
  inline Node operator _operator_(BindingNode const& left, Configurable<T> const& right)                \
  {                                                                                                     \
    return Node{OpNode{BasicOp::_operation_}, left, PlaceholderNode{right}};                            \
  }

BINARY_OP_NODES(&, BitwiseAnd);
BINARY_OP_NODES(^, BitwiseXor);
BINARY_OP_NODES(|, BitwiseOr);
BINARY_OP_NODES(+, Addition);
BINARY_OP_NODES(-, Subtraction);
BINARY_OP_NODES(*, Multiplication);
BINARY_OP_NODES(/, Division);
BINARY_OP_NODES(>, GreaterThan);
BINARY_OP_NODES(>=, GreaterThanOrEqual);
BINARY_OP_NODES(<, LessThan);
BINARY_OP_NODES(<=, LessThanOrEqual);
BINARY_OP_NODES(==, Equal);
BINARY_OP_NODES(!=, NotEqual);
BINARY_OP_NODES(&&, LogicalAnd);
BINARY_OP_NODES(||, LogicalOr);

/// functions
template <typename T>
inline Node npow(Node&& left, T right) requires(std::is_arithmetic_v<T>)
{
  return Node{OpNode{BasicOp::Power}, std::forward<Node>(left), LiteralNode{right}};
}

#define BINARY_FUNC_NODES(_func_, _node_)                                                          \
  template <typename L, typename R>                                                                \
  inline Node _node_(L left, R right) requires(std::is_arithmetic_v<L> && std::is_arithmetic_v<R>) \
  {                                                                                                \
    return Node{OpNode{BasicOp::_func_}, LiteralNode{left}, LiteralNode{right}};                   \
  }                                                                                                \
                                                                                                   \
  inline Node _node_(Node&& left, Node&& right)                                                    \
  {                                                                                                \
    return Node{OpNode{BasicOp::_func_}, std::forward<Node>(left), std::forward<Node>(right)};     \
  }                                                                                                \
                                                                                                   \
  inline Node _node_(Node&& left, BindingNode const& right)                                        \
  {                                                                                                \
    return Node{OpNode{BasicOp::_func_}, std::forward<Node>(left), right};                         \
  }                                                                                                \
                                                                                                   \
  inline Node _node_(BindingNode const& left, BindingNode const& right)                            \
  {                                                                                                \
    return Node{OpNode{BasicOp::_func_}, left, right};                                             \
  }                                                                                                \
                                                                                                   \
  inline Node _node_(BindingNode const& left, Node&& right)                                        \
  {                                                                                                \
    return Node{OpNode{BasicOp::_func_}, left, std::forward<Node>(right)};                         \
  }                                                                                                \
                                                                                                   \
  template <typename T>                                                                            \
  inline Node _node_(Node&& left, Configurable<T> const& right)                                    \
  {                                                                                                \
    return Node{OpNode{BasicOp::_func_}, std::forward<Node>(left), PlaceholderNode{right}};        \
  }                                                                                                \
                                                                                                   \
  template <typename T>                                                                            \
  inline Node _node_(Configurable<T> const& left, Node&& right)                                    \
  {                                                                                                \
    return Node{OpNode{BasicOp::_func_}, PlaceholderNode{left}, std::forward<Node>(right)};        \
  }                                                                                                \
                                                                                                   \
  template <typename T>                                                                            \
  inline Node _node_(BindingNode const& left, Configurable<T> const& right)                        \
  {                                                                                                \
    return Node{OpNode{BasicOp::_func_}, left, PlaceholderNode{right}};                            \
  }                                                                                                \
                                                                                                   \
  template <typename T>                                                                            \
  inline Node _node_(Configurable<T> const& left, BindingNode const& right)                        \
  {                                                                                                \
    return Node{OpNode{BasicOp::_func_}, PlaceholderNode{left}, right};                            \
  }

BINARY_FUNC_NODES(Atan2, natan2);
#define ncheckbit(_node_, _bit_) ((_node_ & _bit_) == _bit_)

/// unary functions on nodes
#define UNARY_FUNC_NODES(_func_, _node_)                           \
  inline Node _node_(Node&& arg)                                   \
  {                                                                \
    return Node{OpNode{BasicOp::_func_}, std::forward<Node>(arg)}; \
  }

UNARY_FUNC_NODES(Round, nround);
UNARY_FUNC_NODES(Sqrt, nsqrt);
UNARY_FUNC_NODES(Exp, nexp);
UNARY_FUNC_NODES(Log, nlog);
UNARY_FUNC_NODES(Log10, nlog10);
UNARY_FUNC_NODES(Abs, nabs);
UNARY_FUNC_NODES(Sin, nsin);
UNARY_FUNC_NODES(Cos, ncos);
UNARY_FUNC_NODES(Tan, ntan);
UNARY_FUNC_NODES(Asin, nasin);
UNARY_FUNC_NODES(Acos, nacos);
UNARY_FUNC_NODES(Atan, natan);
UNARY_FUNC_NODES(BitwiseNot, nbitwise_not);

/// conditionals
inline Node ifnode(Node&& condition_, Node&& then_, Node&& else_)
{
  return Node{ConditionalNode{}, std::forward<Node>(then_), std::forward<Node>(else_), std::forward<Node>(condition_)};
}

template <typename L>
inline Node ifnode(Node&& condition_, Node&& then_, L else_) requires(std::is_arithmetic_v<L>)
{
  return Node{ConditionalNode{}, std::forward<Node>(then_), LiteralNode{else_}, std::forward<Node>(condition_)};
}

template <typename L>
inline Node ifnode(Node&& condition_, L then_, Node&& else_) requires(std::is_arithmetic_v<L>)
{
  return Node{ConditionalNode{}, LiteralNode{then_}, std::forward<Node>(else_), std::forward<Node>(condition_)};
}

template <typename L1, typename L2>
inline Node ifnode(Node&& condition_, L1 then_, L2 else_) requires(std::is_arithmetic_v<L1>&& std::is_arithmetic_v<L2>)
{
  return Node{ConditionalNode{}, LiteralNode{then_}, LiteralNode{else_}, std::forward<Node>(condition_)};
}

template <typename T>
inline Node ifnode(Configurable<T> const& condition_, Node&& then_, Node&& else_)
{
  return Node{ConditionalNode{}, std::forward<Node>(then_), std::forward<Node>(else_), PlaceholderNode{condition_}};
}

template <typename L>
inline Node ifnode(Node&& condition_, Node&& then_, Configurable<L> const& else_)
{
  return Node{ConditionalNode{}, std::forward<Node>(then_), PlaceholderNode{else_}, std::forward<Node>(condition_)};
}

template <typename L>
inline Node ifnode(Node&& condition_, Configurable<L> const& then_, Node&& else_)
{
  return Node{ConditionalNode{}, PlaceholderNode{then_}, std::forward<Node>(else_), std::forward<Node>(condition_)};
}

template <typename L1, typename L2>
inline Node ifnode(Node&& condition_, Configurable<L1> const& then_, Configurable<L2> const& else_)
{
  return Node{ConditionalNode{}, PlaceholderNode{then_}, PlaceholderNode{else_}, std::forward<Node>(condition_)};
}

/// A struct, containing the root of the expression tree
struct Filter {
  Filter(Node&& node_) : node{std::make_unique<Node>(std::forward<Node>(node_))}
  {
    (void)designateSubtrees(node.get());
  }

  Filter(Filter&& other) : node{std::forward<std::unique_ptr<Node>>(other.node)}
  {
    (void)designateSubtrees(node.get());
  }
  std::unique_ptr<Node> node;

  size_t designateSubtrees(Node* node, size_t index = 0);
};

template <typename T>
concept is_filter = std::same_as<T, Filter>;

using Projector = Filter;

/// Function for creating gandiva selection from our internal filter tree
gandiva::Selection createSelection(std::shared_ptr<arrow::Table> const& table, Filter const& expression);
/// Function for creating gandiva selection from prepared gandiva expressions tree
gandiva::Selection createSelection(std::shared_ptr<arrow::Table> const& table, std::shared_ptr<gandiva::Filter> const& gfilter);

struct ColumnOperationSpec;
using Operations = std::vector<ColumnOperationSpec>;

/// Function to create an internal operation sequence from a filter tree
Operations createOperations(Filter const& expression);

/// Function to check compatibility of a given arrow schema with operation sequence
bool isTableCompatible(std::set<uint32_t> const& hashes, Operations const& specs);
/// Function to create gandiva expression tree from operation sequence
gandiva::NodePtr createExpressionTree(Operations const& opSpecs,
                                      gandiva::SchemaPtr const& Schema);
/// Function to create gandiva filter from gandiva condition
std::shared_ptr<gandiva::Filter> createFilter(gandiva::SchemaPtr const& Schema,
                                              gandiva::ConditionPtr condition);
/// Function to create gandiva filter from operation sequence
std::shared_ptr<gandiva::Filter> createFilter(gandiva::SchemaPtr const& Schema,
                                              Operations const& opSpecs);
/// Function to create gandiva projector from operation sequence
std::shared_ptr<gandiva::Projector> createProjector(gandiva::SchemaPtr const& Schema,
                                                    Operations const& opSpecs,
                                                    gandiva::FieldPtr result);
/// Function to create gandiva projector directly from expression
std::shared_ptr<gandiva::Projector> createProjector(gandiva::SchemaPtr const& Schema,
                                                    Projector&& p,
                                                    gandiva::FieldPtr result);
/// Function for attaching gandiva filters to to compatible task inputs
void updateExpressionInfos(expressions::Filter const& filter, std::vector<ExpressionInfo>& eInfos);
/// Function to create gandiva condition expression from generic gandiva expression tree
gandiva::ConditionPtr makeCondition(gandiva::NodePtr node);
/// Function to create gandiva projecting expression from generic gandiva expression tree
gandiva::ExpressionPtr makeExpression(gandiva::NodePtr node, gandiva::FieldPtr result);
/// Update placeholder nodes from context
void updatePlaceholders(Filter& filter, InitContext& context);

template <typename... C>
std::vector<expressions::Projector> makeProjectors(framework::pack<C...>)
{
  return {C::Projector()...};
}

std::shared_ptr<gandiva::Projector> createProjectorHelper(size_t nColumns, expressions::Projector* projectors,
                                                          std::shared_ptr<arrow::Schema> schema,
                                                          std::vector<std::shared_ptr<arrow::Field>> const& fields);

template <typename... C>
std::shared_ptr<gandiva::Projector> createProjectors(framework::pack<C...>, std::vector<std::shared_ptr<arrow::Field>> const& fields, gandiva::SchemaPtr schema)
{
  std::array<expressions::Projector, sizeof...(C)> projectors{{std::move(C::Projector())...}};

  return createProjectorHelper(sizeof...(C), projectors.data(), schema, fields);
}

void updateFilterInfo(ExpressionInfo& info, std::shared_ptr<arrow::Table>& table);
} // namespace o2::framework::expressions

#endif // O2_FRAMEWORK_EXPRESSIONS_H_
