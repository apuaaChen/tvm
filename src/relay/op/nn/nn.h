/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file src/relay/op/nn/nn.h
 * \brief Properties def of nn operators for sharing.
 */
#ifndef TVM_RELAY_OP_NN_NN_H_
#define TVM_RELAY_OP_NN_NN_H_

#include <tvm/auto_scheduler/compute_dag.h>
#include <tvm/ir/attrs.h>
#include <tvm/ir/expr.h>
#include <tvm/relay/type.h>

#include <algorithm>
#include <utility>
#include <bitset>

#include "../op_common.h"
#include "tvm/runtime/logging.h"

#define LABELRANGE 128

namespace tvm {
namespace relay {

inline int64_t GetConstInt(PrimExpr expr) {
  if (expr->IsInstance<tvm::IntImmNode>()) {
    return expr.as<tvm::IntImmNode>()->value;
  }
  return -1;
}

inline std::vector<std::string> Split(const std::string& str, const std::string& sub) {
  std::string::size_type pos = 0;
  std::string::size_type start = 0;
  std::vector<std::string> ret;
  while ((pos = str.find(sub, start)) != std::string::npos) {
    ret.push_back(str.substr(start, pos - start));
    start = pos + sub.length();
  }
  ret.push_back(str.substr(start));
  return ret;
}

inline int CountSubstring(const std::string& str, const std::string& sub) {
  int count = 0;
  std::string::size_type pos = 0;
  while ((pos = str.find(sub, pos)) != std::string::npos) {
    ++count;
    pos += sub.length();
  }
  return count;
}

inline std::bitset<LABELRANGE> Str2Set(const std::string& str) {
  std::bitset<LABELRANGE> ret;
  for (const char& c : str) {
    ret.set(static_cast<int>(c));
  }
  return ret;
}

inline std::tuple<std::string, std::string> ParseEinsumInput(
    std::string subscripts, const std::vector<Array<PrimExpr>>& operands) {
  const std::string einsum_symbols = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
  std::bitset<LABELRANGE> einsum_symbols_set;
  for (const char& c : einsum_symbols) {
    einsum_symbols_set.set(c);
  }

  CHECK_NE(operands.size(), 0U) << "No input operands";

  auto end_pos = std::remove(subscripts.begin(), subscripts.end(), ' ');
  subscripts.erase(end_pos, subscripts.end());

  // Ensure all characters are valid
  for (const char& c : subscripts) {
    if (c == '.' || c == ',' || c == '-' || c == '>') {
      continue;
    }
    CHECK(einsum_symbols_set.test(c)) << "Character " << c << " is not a valid symbol.";
  }

  // Check for proper "->"
  if (subscripts.find('-') != std::string::npos || subscripts.find('>') != std::string::npos) {
    bool invalid = (std::count(subscripts.begin(), subscripts.end(), '-') > 1 ||
                    std::count(subscripts.begin(), subscripts.end(), '>') > 1);
    CHECK(!invalid && CountSubstring(subscripts, "->") == 1)
        << "Subscripts can only contain one '->'.";
  }

  // Parse ellipses
  if (subscripts.find('.') != std::string::npos) {
    std::string used = subscripts;
    used.erase(
        std::remove_if(used.begin(), used.end(),
                       [](const char& c) { return c == '.' || c == ',' || c == '-' || c == '>'; }),
        used.end());

    std::bitset<LABELRANGE> used_set = Str2Set(used);
    std::string ellipse_inds = "";
    for (const char& c : einsum_symbols) {
      if (!used_set.test(static_cast<int>(c))) {
        ellipse_inds.append(1, c);
      }
    }
    int longest = 0;
    std::string input_tmp, output_sub;
    std::vector<std::string> split_subscripts;
    bool out_sub;

    if (subscripts.find("->") != std::string::npos) {
      std::vector<std::string> tmp = Split(subscripts, "->");
      input_tmp = tmp[0];
      output_sub = tmp[1];
      split_subscripts = Split(input_tmp, ",");
      out_sub = true;
    } else {
      split_subscripts = Split(subscripts, ",");
      out_sub = false;
    }

    size_t size_split_subscripts = split_subscripts.size();
    subscripts = "";
    for (size_t i = 0; i < size_split_subscripts; ++i) {
      const std::string& sub = split_subscripts[i];
      if (sub.find('.') != std::string::npos) {
        CHECK_EQ(std::count(sub.begin(), sub.end(), '.'), 3) << "Invalid Ellipses";
        CHECK_EQ(CountSubstring(sub, "..."), 1) << "Invalid Ellipses";

        // Take into account numerical values
        int ellipse_count = 0;
        if (operands[i].size() == 0) {
          ellipse_count = 0;
        } else {
          ellipse_count = std::max(operands[i].size(), static_cast<size_t>(1));
          ellipse_count -= sub.length() - 3;
        }

        if (ellipse_count > longest) {
          longest = ellipse_count;
        }

        CHECK_GE(ellipse_count, 0) << "Ellipses lengths do not match.";
        if (ellipse_count == 0) {
          split_subscripts[i].erase(sub.find("..."), 3);
        } else {
          std::string rep_inds = ellipse_inds.substr(ellipse_inds.length() - ellipse_count);
          split_subscripts[i].replace(sub.find("..."), 3, rep_inds);
        }
      }
      subscripts += split_subscripts[i];
      if (i + 1 < size_split_subscripts) {
        subscripts += ",";
      }
    }
    std::string out_ellipse;
    if (longest == 0) {
      out_ellipse = "";
    } else {
      out_ellipse = ellipse_inds.substr(ellipse_inds.length() - longest);
    }

    if (out_sub) {
      output_sub.replace(output_sub.find("..."), 3, out_ellipse);
      subscripts += "->" + output_sub;
    } else {
      // Special care for outputless ellipses
      std::bitset<LABELRANGE> out_ellipse_set = Str2Set(out_ellipse);
      std::string tmp_subscripts = subscripts, output_subscript = "";
      size_t len_tmp_subscripts = tmp_subscripts.length();
      std::sort(tmp_subscripts.begin(), tmp_subscripts.end());
      for (size_t i = 0; i < len_tmp_subscripts; ++i) {
        const char& c = tmp_subscripts[i];
        if (c == ',') {
          continue;
        }
        CHECK(einsum_symbols_set.test(c)) << "Character " << c << " is not a valid symbol.";
        if ((i == 0 || tmp_subscripts[i - 1] != c) &&
            (i == len_tmp_subscripts - 1 || tmp_subscripts[i + 1] != c) &&
            !out_ellipse_set.test(c)) {
          output_subscript.append(1, c);
        }
      }
      subscripts += "->" + out_ellipse + output_subscript;
    }
  }

  // Build output string if does not exist
  std::tuple<std::string, std::string> ret;
  if (subscripts.find("->") != std::string::npos) {
    std::vector<std::string> tmp(2);
    tmp = Split(subscripts, "->");
    ret = std::make_tuple(tmp[0], tmp[1]);
  } else {
    std::string first = subscripts;
    std::string second = "";
    // Build output subscripts
    std::string tmp_subscripts = subscripts;
    size_t len_tmp_subscripts = tmp_subscripts.length();
    std::sort(tmp_subscripts.begin(), tmp_subscripts.end());
    for (size_t i = 0; i < len_tmp_subscripts; ++i) {
      const char& c = tmp_subscripts[i];
      if (c == ',') {
        continue;
      }
      CHECK(einsum_symbols_set.test(c)) << "Character " << c << " is not a valid symbol.";
      if ((i == 0 || tmp_subscripts[i - 1] != c) &&
          (i == len_tmp_subscripts - 1 || tmp_subscripts[i + 1] != c)) {
        second.append(1, c);
      }
    }
    ret = std::make_tuple(first, second);
  }

  // Make sure output subscripts are in the input
  std::bitset<LABELRANGE> input_subscripts_set = Str2Set(std::get<0>(ret));
  for (const char& c : std::get<1>(ret)) {
    CHECK(input_subscripts_set.test(c))
        << "Output character " << c << " did not appear in the input";
  }

  // Make sure number operands is equivalent to the number of terms
  CHECK_EQ(std::count(std::get<0>(ret).begin(), std::get<0>(ret).end(), ',') + 1, operands.size())
      << "Number of einsum subscripts must be equal to the "
      << "number of operands.";

  return ret;
}


inline Array<PrimExpr> NumpyEinsumShape(const std::string subscripts,
                                        const std::vector<Array<PrimExpr>>& operands) {
  // Parsing
  std::tuple<std::string, std::string> parsed_subscripts = ParseEinsumInput(subscripts, operands);

  // Build a few useful list and sets
  std::vector<std::string> input_list = Split(std::get<0>(parsed_subscripts), ",");
  size_t isize = input_list.size();

  // Get length of each unique dimension and ensure all dimensions are correct
  int dimension_dict[LABELRANGE];
  memset(dimension_dict, -1, sizeof(dimension_dict));
  for (size_t i = 0; i < isize; ++i) {
    const std::string& term = input_list[i];
    const Array<PrimExpr>& sh = operands[i];
    CHECK_EQ(sh.size(), term.length())
        << "Einstein sum subscript " << input_list[i] << " does not contain the "
        << "correct number of indices for operand " << i << ".";
    size_t len_term = term.length();
    for (size_t j = 0; j < len_term; ++j) {
      int64_t dim = GetConstInt(sh[j]);
      const char& c = term[j];

      if (dimension_dict[static_cast<int>(c)] != -1) {
        // For broadcasting cases we always want the largest dim size
        if (dimension_dict[static_cast<int>(c)] == 1) {
          dimension_dict[static_cast<int>(c)] = dim;
        }
        CHECK(dim == 1 || dim == dimension_dict[static_cast<int>(c)])
            << "Size of label '" << c << "' for operand  " << i << " ("
            << dimension_dict[static_cast<int>(c)] << ") does not match previous terms (" << dim
            << ").";
      } else {
        dimension_dict[static_cast<int>(c)] = dim;
      }
    }
  }

  // Get oshape
  const std::string& output_str = std::get<1>(parsed_subscripts);
  size_t odim = output_str.size();
  Array<PrimExpr> oshape(odim, -1);
  for (size_t i = 0; i < odim; ++i) {
    oshape.Set(i, dimension_dict[static_cast<int>(output_str[i])]);
  }
  // Neglecting oshape assign check temporally
  return oshape;
}

template <typename AttrType>
bool EinsumRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
               const TypeReporter& reporter) {
  
  std::vector<Array<PrimExpr>> operands;
  const auto* tensor_tuple = types[0].as<TupleTypeNode>();
  for (size_t i=0; i < tensor_tuple->fields.size(); i++){
    const auto& tensor = Downcast<TensorType>(tensor_tuple->fields[i]);
    operands.push_back(tensor->shape);
  }

  const auto* param = attrs.as<AttrType>();
  if (param == nullptr) {
    reporter->GetDiagCtx().EmitFatal(Diagnostic::Error(reporter->GetSpan())
                                     << "the call attributes are not defined");
    return false;
  }
  std::string subscripts = param->subscripts;
  Array<PrimExpr> oshape = NumpyEinsumShape(subscripts, operands);

  const auto& first = Downcast<TensorType>(tensor_tuple->fields[0]);
  const DataType dtype = first->dtype;

  // Write the output shape
  auto rtype = TensorType(oshape, dtype);
  reporter->Assign(types[1], rtype);
  return true;
}

template <typename AttrType>
bool MatmulRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
               const TypeReporter& reporter) {
  ICHECK_EQ(types.size(), 3);
  const auto* tensor_a = types[0].as<TensorTypeNode>();
  const auto* tensor_b = types[1].as<TensorTypeNode>();
  if (tensor_a == nullptr) return false;
  ICHECK(static_cast<int>(tensor_a->shape.size()) != 0);

  const AttrType* param = attrs.as<AttrType>();
  ICHECK(param != nullptr);
  // Default set to dense layout
  bool transpose_a = false;
  bool transpose_b = true;
  const auto& mattrs = attrs.as<MatmulAttrs>();
  if (mattrs != nullptr) {
    transpose_a = mattrs->transpose_a;
    transpose_b = mattrs->transpose_b;
  }

  const Array<tvm::PrimExpr>& dshape = tensor_a->shape;
  Array<tvm::PrimExpr> oshape = dshape;
  tvm::PrimExpr reduce = dshape[dshape.size() - 1];
  if (transpose_a) {
    reduce = dshape[dshape.size() - 2];
    oshape.Set((oshape.size() - 2), dshape[oshape.size() - 1]);
  }
  if (param->units.defined()) {
    // validate the tensor_b shape is proper if defined
    // Assign tensor_b type
    const Array<IndexExpr>& wshape = transpose_b ? Array<IndexExpr>({param->units, reduce})
                                                 : Array<IndexExpr>({reduce, param->units});
    // It is possible for tensor_b to be nullptr in which case we will use
    // data dtype as the tensor_b dtype. However if tensor_b dtype is explicitly
    // present we will use that.
    auto tensor_b_dtype = (tensor_b == nullptr ? tensor_a->dtype : tensor_b->dtype);
    if (param->auto_scheduler_rewritten_layout.size() == 0) {
      // Normal case: assign result to reporter
      reporter->Assign(types[1], TensorType(wshape, tensor_b_dtype));
    } else {
      // If the layout is rewritten by auto-scheduler,
      // we just forcly apply the layout provided by auto-scheduler and
      // skip the normal inference logic.
      {}  // do nothing
    }
    oshape.Set((oshape.size() - 1), param->units);
  } else {
    if (tensor_b == nullptr) return false;
    const Array<tvm::PrimExpr>& wshape = tensor_b->shape;
    // When tensor_b's layout has been rewritten, figure it out based on the
    // total number of elements and input dimensions.
    if (param->auto_scheduler_rewritten_layout.size() != 0) {
      PrimExpr tensor_b_elements = 1;
      for (size_t i = 0; i < wshape.size(); i++) {
        tensor_b_elements = tensor_b_elements * wshape[i];
      }
      oshape.Set(oshape.size() - 1, tensor_b_elements / dshape[dshape.size() - 1]);
      // Otherwise just pull it out of the tensor_b shape directly.
    } else {
      ICHECK(static_cast<int>(tensor_b->shape.size()) == 2);
      if (!tensor_a->shape.back().as<tir::AnyNode>()) {
        ICHECK((transpose_b && reporter->AssertEQ(reduce, tensor_b->shape[1])) ||
               (!transpose_b && reporter->AssertEQ(reduce, tensor_b->shape[0])))
            << "MatmulRel: input dimension doesn't match,"
            << " tensor_a shape=" << tensor_a->shape << ", tensor_b shape=" << tensor_b->shape;
      }
      oshape.Set((oshape.size() - 1), transpose_b ? wshape[0] : wshape[1]);
    }
  }

  DataType out_dtype = param->out_dtype;
  if (out_dtype.bits() == 0) {
    out_dtype = tensor_a->dtype;
  }
  // assign output type
  reporter->Assign(types[2], TensorType(oshape, out_dtype));
  return true;
}

template <typename AttrType>
bool BatchMatmulRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                    const TypeReporter& reporter) {
  ICHECK_EQ(types.size(), 3);
  const auto* x = types[0].as<TensorTypeNode>();
  const auto* y = types[1].as<TensorTypeNode>();
  if (x == nullptr || y == nullptr) return false;

  const AttrType* param = attrs.as<AttrType>();
  ICHECK(param != nullptr);
  bool transpose_a = param->transpose_a;
  bool transpose_b = param->transpose_b;
  const Array<PrimExpr>& y_shape =
      param->auto_scheduler_rewritten_layout.size() == 0
          ? y->shape
          : auto_scheduler::GetShapeFromRewrittenLayout(
                param->auto_scheduler_rewritten_layout,
                transpose_b ? tvm::runtime::Array<tvm::runtime::String>({"b", "j", "k"})
                            : tvm::runtime::Array<tvm::runtime::String>({"b", "k", "j"}));
  ICHECK(x->shape.size() == 3 && y_shape.size() == 3);
  const PrimExpr& xb = x->shape[0];
  const PrimExpr& xi = x->shape[transpose_a ? 2 : 1];
  const PrimExpr& xk = x->shape[transpose_a ? 1 : 2];
  const PrimExpr& yb = y_shape[0];
  const PrimExpr& yk = y_shape[transpose_b ? 2 : 1];
  const PrimExpr& yj = y_shape[transpose_b ? 1 : 2];

  bool is_dyn = false;
  for (size_t i = 0; i < 3; ++i) {
    if (x->shape[i].as<tir::AnyNode>() != nullptr || y_shape[i].as<tir::AnyNode>() != nullptr) {
      is_dyn = true;
      break;
    }
  }
  if (!is_dyn) {
    ICHECK(reporter->AssertEQ(xb, yb) || reporter->AssertEQ(xb, 1) || reporter->AssertEQ(yb, 1))
        << "BatchDot: batch dimensions don't match, "
        << " x shape=" << x->shape << ", y shape=" << y_shape;
    ICHECK(reporter->AssertEQ(xk, yk)) << "BatchDot: shapes of x and y is inconsistent, "
                                       << " x shape=" << x->shape << ", y shape=" << y_shape;
  }

  DataType out_dtype = param->out_dtype;
  if (out_dtype.bits() == 0) {
    out_dtype = x->dtype;
  }
  // assign output type
  const auto& out_b =
      xb->IsInstance<tir::AnyNode>() || yb->IsInstance<tir::AnyNode>() ? tir::Any() : max(xb, yb);
  reporter->Assign(types[2], TensorType(Array<tvm::PrimExpr>({out_b, xi, yj}), out_dtype));
  return true;
}

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_OP_NN_NN_H_
