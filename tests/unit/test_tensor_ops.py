"""Comprehensive tests for Tensor operations."""

import numpy as np
import pytest

from cuDeep import Tensor, DType


class TestTensorCreation:
    def test_zeros_f32(self):
        t = Tensor.zeros([3, 4])
        arr = t.numpy()
        assert arr.shape == (3, 4)
        assert arr.dtype == np.float32
        np.testing.assert_array_equal(arr, 0.0)

    def test_zeros_f64(self):
        t = Tensor.zeros([2, 5], DType.float64)
        arr = t.numpy()
        assert arr.shape == (2, 5)
        assert arr.dtype == np.float64
        np.testing.assert_array_equal(arr, 0.0)

    def test_ones_f32(self):
        t = Tensor.ones([4, 3])
        arr = t.numpy()
        np.testing.assert_allclose(arr, 1.0)

    def test_ones_f64(self):
        t = Tensor.ones([2, 2], DType.float64)
        arr = t.numpy()
        np.testing.assert_allclose(arr, 1.0)

    def test_randn_shape(self):
        t = Tensor.randn([100, 50])
        arr = t.numpy()
        assert arr.shape == (100, 50)
        assert not np.allclose(arr, 0.0)

    def test_randn_statistics(self):
        t = Tensor.randn([10000])
        arr = t.numpy()
        assert abs(arr.mean()) < 0.1
        assert abs(arr.std() - 1.0) < 0.1

    def test_randn_odd_numel(self):
        t = Tensor.randn([7, 3])
        arr = t.numpy()
        assert arr.shape == (7, 3)
        assert np.all(np.isfinite(arr))

    def test_randn_f64(self):
        t = Tensor.randn([256], DType.float64)
        arr = t.numpy()
        assert arr.dtype == np.float64
        assert np.all(np.isfinite(arr))

    def test_from_numpy_f32(self):
        original = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        t = Tensor.from_numpy(original)
        result = t.numpy()
        np.testing.assert_array_equal(original, result)

    def test_from_numpy_f64(self):
        original = np.array([1.5, 2.5, 3.5], dtype=np.float64)
        t = Tensor.from_numpy(original)
        result = t.numpy()
        np.testing.assert_array_equal(original, result)


class TestTensorMetadata:
    def test_shape(self):
        t = Tensor([2, 3, 4])
        assert t.shape() == [2, 3, 4]

    def test_ndim(self):
        t = Tensor([2, 3, 4])
        assert t.ndim() == 3

    def test_numel(self):
        t = Tensor([2, 3, 4])
        assert t.numel() == 24

    def test_nbytes_f32(self):
        t = Tensor([10])
        assert t.nbytes() == 40

    def test_nbytes_f64(self):
        t = Tensor([10], DType.float64)
        assert t.nbytes() == 80

    def test_dtype(self):
        t32 = Tensor([1], DType.float32)
        t64 = Tensor([1], DType.float64)
        assert t32.dtype() == DType.float32
        assert t64.dtype() == DType.float64

    def test_is_contiguous(self):
        t = Tensor([2, 3])
        assert t.is_contiguous()

    def test_repr(self):
        t = Tensor([2, 3])
        r = repr(t)
        assert "2" in r and "3" in r
        assert "float32" in r


class TestTensorReshape:
    def test_reshape_basic(self):
        data = np.arange(12, dtype=np.float32).reshape(3, 4)
        t = Tensor.from_numpy(data)
        r = t.reshape([4, 3])
        assert r.shape() == [4, 3]
        assert r.numel() == 12

    def test_reshape_preserves_data(self):
        data = np.arange(6, dtype=np.float32)
        t = Tensor.from_numpy(data)
        r = t.reshape([2, 3])
        result = r.numpy()
        np.testing.assert_array_equal(result.ravel(), data)

    def test_reshape_to_1d(self):
        data = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        t = Tensor.from_numpy(data)
        r = t.reshape([24])
        assert r.shape() == [24]
        np.testing.assert_array_equal(r.numpy(), data.ravel())


class TestElementwiseOps:
    def test_add(self):
        a = np.array([1, 2, 3, 4], dtype=np.float32)
        b = np.array([10, 20, 30, 40], dtype=np.float32)
        ta = Tensor.from_numpy(a)
        tb = Tensor.from_numpy(b)
        tc = ta + tb
        np.testing.assert_allclose(tc.numpy(), a + b)

    def test_add_2d(self):
        a = np.random.randn(8, 16).astype(np.float32)
        b = np.random.randn(8, 16).astype(np.float32)
        ta = Tensor.from_numpy(a)
        tb = Tensor.from_numpy(b)
        tc = ta + tb
        np.testing.assert_allclose(tc.numpy(), a + b, rtol=1e-5)

    def test_add_f64(self):
        a = np.array([1.0, 2.0], dtype=np.float64)
        b = np.array([3.0, 4.0], dtype=np.float64)
        ta = Tensor.from_numpy(a)
        tb = Tensor.from_numpy(b)
        tc = ta + tb
        np.testing.assert_allclose(tc.numpy(), a + b)

    def test_sub(self):
        a = np.array([10, 20, 30], dtype=np.float32)
        b = np.array([1, 2, 3], dtype=np.float32)
        ta = Tensor.from_numpy(a)
        tb = Tensor.from_numpy(b)
        tc = ta - tb
        np.testing.assert_allclose(tc.numpy(), a - b)

    def test_sub_negative_result(self):
        a = np.array([1, 2], dtype=np.float32)
        b = np.array([5, 10], dtype=np.float32)
        ta = Tensor.from_numpy(a)
        tb = Tensor.from_numpy(b)
        tc = ta - tb
        np.testing.assert_allclose(tc.numpy(), a - b)

    def test_mul(self):
        a = np.array([2, 3, 4], dtype=np.float32)
        b = np.array([5, 6, 7], dtype=np.float32)
        ta = Tensor.from_numpy(a)
        tb = Tensor.from_numpy(b)
        tc = ta * tb
        np.testing.assert_allclose(tc.numpy(), a * b)

    def test_mul_large(self):
        a = np.random.randn(1024).astype(np.float32)
        b = np.random.randn(1024).astype(np.float32)
        ta = Tensor.from_numpy(a)
        tb = Tensor.from_numpy(b)
        tc = ta * tb
        np.testing.assert_allclose(tc.numpy(), a * b, rtol=1e-5)

    def test_chained_ops(self):
        a = np.array([1, 2, 3], dtype=np.float32)
        b = np.array([4, 5, 6], dtype=np.float32)
        c = np.array([7, 8, 9], dtype=np.float32)
        ta = Tensor.from_numpy(a)
        tb = Tensor.from_numpy(b)
        tc = Tensor.from_numpy(c)
        result = (ta + tb) * tc
        expected = (a + b) * c
        np.testing.assert_allclose(result.numpy(), expected)

    def test_ops_preserve_shape(self):
        a = Tensor.ones([3, 4])
        b = Tensor.ones([3, 4])
        c = a + b
        assert c.shape() == [3, 4]
        assert c.is_contiguous()


class TestMatmul:
    def test_matmul_2x2(self):
        A = np.array([[1, 2], [3, 4]], dtype=np.float32)
        B = np.array([[5, 6], [7, 8]], dtype=np.float32)
        tA = Tensor.from_numpy(A)
        tB = Tensor.from_numpy(B)
        tC = tA.matmul(tB)
        expected = A @ B
        np.testing.assert_allclose(tC.numpy(), expected, rtol=1e-5)

    def test_matmul_identity(self):
        N = 8
        A = np.random.randn(N, N).astype(np.float32)
        I = np.eye(N, dtype=np.float32)
        tA = Tensor.from_numpy(A)
        tI = Tensor.from_numpy(I)
        tC = tA.matmul(tI)
        np.testing.assert_allclose(tC.numpy(), A, rtol=1e-5)

    def test_matmul_nonsquare(self):
        A = np.random.randn(3, 5).astype(np.float32)
        B = np.random.randn(5, 7).astype(np.float32)
        tA = Tensor.from_numpy(A)
        tB = Tensor.from_numpy(B)
        tC = tA.matmul(tB)
        expected = A @ B
        assert tC.shape() == [3, 7]
        np.testing.assert_allclose(tC.numpy(), expected, rtol=1e-4)

    def test_matmul_large(self):
        A = np.random.randn(64, 128).astype(np.float32)
        B = np.random.randn(128, 32).astype(np.float32)
        tA = Tensor.from_numpy(A)
        tB = Tensor.from_numpy(B)
        tC = tA.matmul(tB)
        expected = A @ B
        assert tC.shape() == [64, 32]
        np.testing.assert_allclose(tC.numpy(), expected, rtol=1e-3, atol=1e-3)

    def test_matmul_f64(self):
        A = np.array([[1, 2], [3, 4]], dtype=np.float64)
        B = np.array([[5, 6], [7, 8]], dtype=np.float64)
        tA = Tensor.from_numpy(A)
        tB = Tensor.from_numpy(B)
        tC = tA.matmul(tB)
        expected = A @ B
        np.testing.assert_allclose(tC.numpy(), expected)

    def test_matmul_output_shape(self):
        tA = Tensor.randn([10, 20])
        tB = Tensor.randn([20, 30])
        tC = tA.matmul(tB)
        assert tC.shape() == [10, 30]


class TestTranspose:
    def test_transpose_2d(self):
        data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        t = Tensor.from_numpy(data)
        tt = t.transpose(0, 1)
        assert tt.shape() == [3, 2]

    def test_transpose_not_contiguous(self):
        t = Tensor.from_numpy(np.arange(6, dtype=np.float32).reshape(2, 3))
        tt = t.transpose(0, 1)
        assert not tt.is_contiguous()

    def test_transpose_self_inverse(self):
        data = np.random.randn(4, 6).astype(np.float32)
        t = Tensor.from_numpy(data)
        tt = t.transpose(0, 1).transpose(0, 1)
        assert tt.shape() == [4, 6]
        assert tt.is_contiguous()

    def test_transpose_3d(self):
        t = Tensor([2, 3, 4])
        tt = t.transpose(0, 2)
        assert tt.shape() == [4, 3, 2]

    def test_transpose_contiguous_roundtrip(self):
        data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        t = Tensor.from_numpy(data)
        tt = t.transpose(0, 1).contiguous()
        assert tt.is_contiguous()
        assert tt.shape() == [3, 2]
        result = tt.numpy()
        np.testing.assert_allclose(result, data.T)


class TestContiguous:
    def test_contiguous_already_contiguous(self):
        data = np.arange(12, dtype=np.float32).reshape(3, 4)
        t = Tensor.from_numpy(data)
        tc = t.contiguous()
        assert tc.is_contiguous()
        np.testing.assert_array_equal(tc.numpy(), data)

    def test_contiguous_after_transpose(self):
        data = np.arange(6, dtype=np.float32).reshape(2, 3)
        t = Tensor.from_numpy(data)
        tt = t.transpose(0, 1)
        tc = tt.contiguous()
        assert tc.is_contiguous()
        assert tc.shape() == [3, 2]
        np.testing.assert_allclose(tc.numpy(), data.T)

    def test_contiguous_f64(self):
        data = np.arange(12, dtype=np.float64).reshape(3, 4)
        t = Tensor.from_numpy(data)
        tt = t.transpose(0, 1)
        tc = tt.contiguous()
        assert tc.is_contiguous()
        np.testing.assert_allclose(tc.numpy(), data.T)


class TestFillAndZero:
    def test_fill_f32(self):
        t = Tensor([4, 4])
        t.fill_(3.14)
        arr = t.numpy()
        np.testing.assert_allclose(arr, 3.14, rtol=1e-5)

    def test_fill_f64(self):
        t = Tensor([3, 3], DType.float64)
        t.fill_(2.718)
        arr = t.numpy()
        np.testing.assert_allclose(arr, 2.718, rtol=1e-10)

    def test_zero(self):
        t = Tensor.ones([5, 5])
        t.zero_()
        arr = t.numpy()
        np.testing.assert_array_equal(arr, 0.0)

    def test_fill_then_zero(self):
        t = Tensor([10])
        t.fill_(99.0)
        np.testing.assert_allclose(t.numpy(), 99.0)
        t.zero_()
        np.testing.assert_array_equal(t.numpy(), 0.0)


class TestEdgeCases:
    def test_single_element(self):
        t = Tensor.ones([1])
        assert t.numel() == 1
        np.testing.assert_allclose(t.numpy(), [1.0])

    def test_large_tensor(self):
        t = Tensor.zeros([1000, 1000])
        assert t.numel() == 1_000_000
        assert t.nbytes() == 4_000_000

    def test_add_ones(self):
        a = Tensor.ones([100])
        b = Tensor.ones([100])
        c = a + b
        np.testing.assert_allclose(c.numpy(), 2.0)

    def test_self_multiply(self):
        data = np.array([1, 2, 3, 4], dtype=np.float32)
        t = Tensor.from_numpy(data)
        sq = t * t
        np.testing.assert_allclose(sq.numpy(), data ** 2)
