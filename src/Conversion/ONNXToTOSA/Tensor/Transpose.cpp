/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- Transpose.cpp - Transpose Op -----------------------------===//
//
// Copyright (c) d-Matrix Inc. 2023
//
// =============================================================================
//
// This file lowers ONNX transpose operator to TOSA dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToTOSA/ONNXToTOSACommon.hpp"

using namespace mlir;

namespace onnx_mlir {

namespace {

class ONNXTransposeOpLoweringToTOSA : public OpConversionPattern<ONNXTransposeOp> {
public:
  using OpConversionPattern<ONNXTransposeOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(ONNXTransposeOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ArrayAttr permAttr = op.getPermAttr();
    ArrayRef<Attribute> permArray = permAttr.getValue();
    int64_t permArraySize = (int64_t)permArray.size();
    auto permType = RankedTensorType::get({permArraySize}, rewriter.getI64Type());
    auto shapeConstOp = rewriter.create<tosa::ConstOp>(op.getLoc(), permType, DenseElementsAttr::get(permType, permArray));
  
    SmallVector<Value> inputs;
    inputs.push_back(op.getData());
    inputs.push_back(shapeConstOp);
  
    rewriter.replaceOpWithNewOp<tosa::TransposeOp>(op, op.getType(), inputs);
    return success();
  }
};

} // namespace

void populateLoweringONNXTransposeOpToTOSAPattern(ConversionTarget &target,
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *ctx) {
  patterns.insert<ONNXTransposeOpLoweringToTOSA>(typeConverter, ctx);
}

} // namespace onnx_mlir
