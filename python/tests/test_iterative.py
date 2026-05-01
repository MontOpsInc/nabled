"""Tests for iterative solver bindings (CG, GMRES)."""

import numpy as np
import pytest

import pynabled


def _make_spd(n):
    np.random.seed(42)
    x = np.random.randn(n, n).astype(np.float64)
    return x.T @ x + 0.5 * np.eye(n)


def test_conjugate_gradient():
    a = _make_spd(5)
    b = np.random.randn(5).astype(np.float64)
    x = pynabled.conjugate_gradient(
        a,
        b,
        config=pynabled.IterativeConfig(tolerance=1e-12, max_iterations=128),
    )
    np.testing.assert_allclose(a @ x, b, rtol=1e-10)


def test_gmres():
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    b = np.array([5.0, 6.0], dtype=np.float64)
    x = pynabled.gmres(a, b, None, None)
    np.testing.assert_allclose(a @ x, b, rtol=1e-10)


def test_conjugate_gradient_complex():
    a = np.array([[2.0 + 0j, 0.0 + 0j], [0.0 + 0j, 3.0 + 0j]], dtype=np.complex128)
    b = np.array([2.0 + 1.0j, 3.0 - 2.0j], dtype=np.complex128)
    x = pynabled.conjugate_gradient_complex(a, b, None, None)
    np.testing.assert_allclose(a @ x, b, rtol=1e-10, atol=1e-10)


def test_gmres_complex():
    a = np.array([[1.0 + 1.0j, 0.0 + 0j], [0.0 + 0j, 2.0 - 1.0j]], dtype=np.complex128)
    b = np.array([1.0 + 2.0j, 4.0 - 2.0j], dtype=np.complex128)
    x = pynabled.gmres_complex(a, b, None, None)
    np.testing.assert_allclose(a @ x, b, rtol=1e-10, atol=1e-10)


def test_iterative_real_solvers_accept_float32():
    spd = np.array(
        [[6.0, 2.0, 0.0], [2.0, 5.0, 1.0], [0.0, 1.0, 4.0]],
        dtype=np.float32,
    )
    rhs_spd = np.array([1.0, -2.0, 3.0], dtype=np.float32)
    general = np.array([[4.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    rhs_general = np.array([1.0, 2.0], dtype=np.float32)

    cg = pynabled.conjugate_gradient(spd, rhs_spd, None, None)
    gmres = pynabled.gmres(general, rhs_general, None, None)

    assert cg.dtype == np.float32
    assert gmres.dtype == np.float32
    np.testing.assert_allclose(spd @ cg, rhs_spd, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(general @ gmres, rhs_general, rtol=1e-4, atol=1e-5)


def test_iterative_rejects_config_and_explicit_kwargs():
    a = _make_spd(3)
    b = np.ones(3, dtype=np.float64)

    with pytest.raises(TypeError, match="config="):
        pynabled.conjugate_gradient(
            a,
            b,
            tolerance=1e-8,
            config=pynabled.IterativeConfig(max_iterations=64),
        )


def test_iterative_reuses_output_buffers():
    spd = np.array([[6.0, 2.0, 0.0], [2.0, 5.0, 1.0], [0.0, 1.0, 4.0]], dtype=np.float64)
    rhs_spd = np.array([1.0, -2.0, 3.0], dtype=np.float64)
    general = np.array([[4.0, 1.0], [2.0, 3.0]], dtype=np.float64)
    rhs_general = np.array([1.0, 2.0], dtype=np.float64)

    cg_out = np.empty_like(rhs_spd)
    gmres_out = np.empty_like(rhs_general)

    returned_cg = pynabled.conjugate_gradient(spd, rhs_spd, out=cg_out)
    returned_gmres = pynabled.gmres(general, rhs_general, out=gmres_out)

    assert returned_cg is cg_out
    assert returned_gmres is gmres_out
    np.testing.assert_allclose(spd @ cg_out, rhs_spd, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(general @ gmres_out, rhs_general, rtol=1e-10, atol=1e-12)


def test_iterative_complex_accepts_strided_inputs_and_reuses_output_buffers():
    a = np.asfortranarray(
        np.array([[2.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 3.0 + 0.0j]], dtype=np.complex128)
    )
    b_base = np.array([2.0 + 1.0j, 99.0 + 0.0j, 3.0 - 2.0j, -11.0 + 0.0j], dtype=np.complex128)
    b = b_base[::2]
    general = np.asfortranarray(
        np.array([[1.0 + 1.0j, 0.0 + 0.0j], [0.0 + 0.0j, 2.0 - 1.0j]], dtype=np.complex128)
    )
    rhs_base = np.array([1.0 + 2.0j, 8.0 + 0.0j, 4.0 - 2.0j, 9.0 + 0.0j], dtype=np.complex128)
    rhs = rhs_base[::2]

    cg_out = np.empty(2, dtype=np.complex128)
    gmres_out = np.empty(2, dtype=np.complex128)

    returned_cg = pynabled.conjugate_gradient_complex(a, b, out=cg_out)
    returned_gmres = pynabled.gmres_complex(general, rhs, out=gmres_out)

    assert returned_cg is cg_out
    assert returned_gmres is gmres_out
    np.testing.assert_allclose(a @ cg_out, b, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(general @ gmres_out, rhs, rtol=1e-10, atol=1e-10)


def test_iterative_out_rejects_wrong_output_dtype():
    a = _make_spd(3)
    b = np.ones(3, dtype=np.float64)
    bad_out = np.empty(3, dtype=np.float32)

    with pytest.raises(TypeError, match="output must be a writable NumPy array with dtype float64"):
        pynabled.conjugate_gradient(a, b, out=bad_out)
