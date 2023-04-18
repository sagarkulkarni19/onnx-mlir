// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-tosa %s -split-input-file | FileCheck %s

func.func @test_transpose(%arg0 : tensor<1x128x16x64xf32>) -> tensor<1x16x128x64xf32> {
  %0 = "onnx.Transpose"(%arg0) {perm = [0, 2, 1, 3]} : (tensor<1x128x16x64xf32>) -> tensor<1x16x128x64xf32>
  "func.return"(%0) : (tensor<1x16x128x64xf32>) -> ()
// CHECK-LABEL:  func @test_transpose
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x128x16x64xf32>) -> tensor<1x16x128x64xf32> {
// CHECK-NEXT:      [[VAR_0_:%.+]] = "tosa.const"() {value = dense<[0, 2, 1, 3]> : tensor<4xi64>} : () -> tensor<4xi64>
// CHECK-NEXT:      [[VAR_1_:%.+]] = "tosa.transpose"([[PARAM_0_]], [[VAR_0_]]) : (tensor<1x128x16x64xf32>, tensor<4xi64>) -> tensor<1x16x128x64xf32>
// CHECK-NEXT:      return [[VAR_1_]] : tensor<1x16x128x64xf32>
}
