/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * @file staticlu.cpp
 * @author Gautier Bureau <gautier.bureau at rte-france.com>
 *
 * A sparse LU with a static elimination order and no pivoting, for a matrix whose pattern does not change while its
 * values do: the order (AMD of the symmetrized pattern), the exact fill and the list of the operations of the
 * factorization are computed once, a factorization only runs the list. A pivot too small, or a multiplier too large,
 * is counted: the caller then factorizes with pivoting instead. It is the CPU counterpart of the fast engine of a
 * batched LU on a GPU.
 *
 * The matrix A is given in CSC. It is A^T that is factorized: its rows are the columns of A, so A^T x = b is a direct
 * solve on the factors and A x = b a transposed one, as klu_tsolve and klu_solve are on A.
 */

#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>
#include <queue>
#include <stdexcept>
#include <string>
#include <vector>
#include <amd.h>
#include "jniwrapper.hpp"

namespace {

const double PIVOT_EPSILON = 1e-10;
const double MULTIPLIER_LIMIT = 1e8;

class StaticLu {
public:
    // The plan of M = A^T, whose rows are given by the CSC arrays (ap, ai) of A.
    StaticLu(int n, const int* ap, const int* ai);

    // Factorizes M of the values of A, in the order of the CSC arrays of the plan: the number of pivots too small and
    // of multipliers too large, 0 when the factors can be trusted.
    int factor(const double* ax, int nnz);

    // A^T x = b when transpose (direct on M), A x = b otherwise (transposed on M), for nrhs right-hand sides of ldim
    // rows each, overwritten by the solutions.
    void solve(double* b, int ldim, int nrhs, bool transpose);

private:
    void solveM(double* b);
    void solveMT(double* b);

    int _n;
    int _nnz;
    std::vector<int> _perm;          // new index -> M's
    std::vector<int> _rowStart;      // the LU pattern by rows, columns ascending, diagonal included
    std::vector<int> _col;
    std::vector<int> _diag;          // position of the diagonal of each row
    std::vector<int> _entryOfA;      // A's entry -> LU position
    std::vector<int> _lowerStart;    // per row, its first lower entry in _lowerPos
    std::vector<int> _lowerPos;      // per lower entry, its LU position
    std::vector<int> _lowerDiag;     // ...the position of its divisor
    std::vector<int> _opStart;       // ...its first operation (one more entry: the end)
    std::vector<int> _opSrc;         // per operation: the position of u(k, j)...
    std::vector<int> _opDst;         // ...and of (i, j)
    std::vector<double> _lu;
    std::vector<double> _work;
};

StaticLu::StaticLu(int n, const int* ap, const int* ai)
    : _n(n),
      _nnz(ap[n]) {
    _perm.resize(n);
    std::vector<int> inv(n);
    double info[AMD_INFO];
    if (amd_order(n, ap, ai, _perm.data(), nullptr, info) < AMD_OK) {
        throw std::runtime_error("amd_order error");
    }
    for (int k = 0; k < n; k++) {
        inv[_perm[k]] = k;
    }
    // the rows of B = P M P^T: row r of B is row perm[r] of M, its columns mapped by inv
    std::vector<std::vector<int>> bRows(n);
    for (int r = 0; r < n; r++) {
        int old = _perm[r];
        for (int e = ap[old]; e < ap[old + 1]; e++) {
            bRows[r].push_back(inv[ai[e]]);
        }
        std::sort(bRows[r].begin(), bRows[r].end());
    }
    // the exact fill, row by row: row i of B united with U(k, > k) for each k < i of its pattern
    std::vector<std::vector<int>> rows(n);
    std::vector<int> mark(n, -1);
    for (int i = 0; i < n; i++) {
        std::vector<int> pattern;
        std::priority_queue<int, std::vector<int>, std::greater<int>> lower;
        for (int j : bRows[i]) {
            mark[j] = i;
            pattern.push_back(j);
            if (j < i) {
                lower.push(j);
            }
        }
        while (!lower.empty()) {
            int k = lower.top();
            lower.pop();
            for (int j : rows[k]) {
                if (j > k && mark[j] != i) {
                    mark[j] = i;
                    pattern.push_back(j);
                    if (j < i) {
                        lower.push(j);
                    }
                }
            }
        }
        std::sort(pattern.begin(), pattern.end());
        rows[i] = pattern;
    }
    _rowStart.assign(n + 1, 0);
    for (int i = 0; i < n; i++) {
        _rowStart[i + 1] = _rowStart[i] + static_cast<int>(rows[i].size());
    }
    _col.resize(_rowStart[n]);
    _diag.resize(n);
    for (int i = 0; i < n; i++) {
        std::copy(rows[i].begin(), rows[i].end(), _col.begin() + _rowStart[i]);
        auto it = std::lower_bound(rows[i].begin(), rows[i].end(), i);
        if (it == rows[i].end() || *it != i) {
            throw std::runtime_error("No diagonal entry in row " + std::to_string(i) + ": a static order needs one");
        }
        _diag[i] = _rowStart[i] + static_cast<int>(it - rows[i].begin());
    }
    auto position = [this](int i, int j) {
        auto begin = _col.begin() + _rowStart[i];
        auto end = _col.begin() + _rowStart[i + 1];
        return static_cast<int>(std::lower_bound(begin, end, j) - _col.begin());
    };
    _entryOfA.resize(_nnz);
    for (int old = 0; old < n; old++) {
        for (int e = ap[old]; e < ap[old + 1]; e++) {
            _entryOfA[e] = position(inv[old], inv[ai[e]]);
        }
    }
    // the operation list, row by row, each lower entry in ascending column with the updates of its pivot row
    _lowerStart.assign(n + 1, 0);
    for (int i = 0; i < n; i++) {
        _lowerStart[i] = static_cast<int>(_lowerPos.size());
        for (int q = _rowStart[i]; q < _diag[i]; q++) {
            int k = _col[q];
            _lowerPos.push_back(q);
            _lowerDiag.push_back(_diag[k]);
            _opStart.push_back(static_cast<int>(_opSrc.size()));
            for (int s = _diag[k] + 1; s < _rowStart[k + 1]; s++) {
                _opSrc.push_back(s);
                _opDst.push_back(position(i, _col[s]));
            }
        }
    }
    _lowerStart[n] = static_cast<int>(_lowerPos.size());
    _opStart.push_back(static_cast<int>(_opSrc.size()));
    _lu.assign(_col.size(), 0.0);
    _work.assign(n, 0.0);
}

int StaticLu::factor(const double* ax, int nnz) {
    if (nnz != _nnz) {
        throw std::runtime_error("The values are not those of the pattern of the plan");
    }
    std::fill(_lu.begin(), _lu.end(), 0.0);
    double* v = _lu.data();
    for (int e = 0; e < nnz; e++) {
        v[_entryOfA[e]] = ax[e];
    }
    const int* lowerStart = _lowerStart.data();
    const int* lowerPos = _lowerPos.data();
    const int* lowerDiag = _lowerDiag.data();
    const int* opStart = _opStart.data();
    const int* src = _opSrc.data();
    const int* dst = _opDst.data();
    int trouble = 0;
    for (int i = 0; i < _n; i++) {
        for (int q = lowerStart[i]; q < lowerStart[i + 1]; q++) {
            double l = v[lowerPos[q]] / v[lowerDiag[q]];
            if (!(std::fabs(l) < MULTIPLIER_LIMIT)) {
                trouble++;
            }
            v[lowerPos[q]] = l;
            for (int o = opStart[q]; o < opStart[q + 1]; o++) {
                v[dst[o]] -= l * v[src[o]];
            }
        }
        if (!(std::fabs(v[_diag[i]]) > PIVOT_EPSILON)) {
            trouble++;
        }
    }
    return trouble;
}

// M x = b: B = P M P^T = L U, so x = P^T U^-1 L^-1 P b
void StaticLu::solveM(double* b) {
    double* w = _work.data();
    for (int r = 0; r < _n; r++) {
        w[r] = b[_perm[r]];
    }
    for (int i = 0; i < _n; i++) {
        double s = w[i];
        for (int q = _rowStart[i]; q < _diag[i]; q++) {
            s -= _lu[q] * w[_col[q]];
        }
        w[i] = s;
    }
    for (int i = _n - 1; i >= 0; i--) {
        double s = w[i];
        for (int q = _diag[i] + 1; q < _rowStart[i + 1]; q++) {
            s -= _lu[q] * w[_col[q]];
        }
        w[i] = s / _lu[_diag[i]];
    }
    for (int r = 0; r < _n; r++) {
        b[_perm[r]] = w[r];
    }
}

// M^T x = b: B^T = U^T L^T, solved by rows of the factors, scattering
void StaticLu::solveMT(double* b) {
    double* w = _work.data();
    for (int r = 0; r < _n; r++) {
        w[r] = b[_perm[r]];
    }
    for (int i = 0; i < _n; i++) {
        double y = w[i] / _lu[_diag[i]];
        w[i] = y;
        for (int q = _diag[i] + 1; q < _rowStart[i + 1]; q++) {
            w[_col[q]] -= _lu[q] * y;
        }
    }
    for (int i = _n - 1; i >= 0; i--) {
        double z = w[i];
        for (int q = _rowStart[i]; q < _diag[i]; q++) {
            w[_col[q]] -= _lu[q] * z;
        }
    }
    for (int r = 0; r < _n; r++) {
        b[_perm[r]] = w[r];
    }
}

void StaticLu::solve(double* b, int ldim, int nrhs, bool transpose) {
    for (int k = 0; k < nrhs; k++) {
        if (transpose) {
            solveM(b + static_cast<size_t>(k) * ldim);
        } else {
            solveMT(b + static_cast<size_t>(k) * ldim);
        }
    }
}

StaticLu* handle(jlong h) {
    if (h == 0) {
        throw std::runtime_error("Static LU released");
    }
    return reinterpret_cast<StaticLu*>(h);
}

}  // namespace

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Class:     com_powsybl_math_matrix_StaticLUDecomposition
 * Method:    create
 * Signature: ([I[I)J
 */
JNIEXPORT jlong JNICALL Java_com_powsybl_math_matrix_StaticLUDecomposition_create(JNIEnv* env, jclass, jintArray j_ap, jintArray j_ai) {
    try {
        powsybl::jni::IntArray ap(env, j_ap);
        powsybl::jni::IntArray ai(env, j_ai);
        auto* lu = new StaticLu(static_cast<int>(ap.length()) - 1, ap.get(), ai.get());
        return reinterpret_cast<jlong>(lu);
    } catch (const std::exception& e) {
        powsybl::jni::throwMatrixException(env, e.what());
    } catch (...) {
        powsybl::jni::throwMatrixException(env, "Unknown exception");
    }
    return 0;
}

/*
 * Class:     com_powsybl_math_matrix_StaticLUDecomposition
 * Method:    factor
 * Signature: (J[D)I
 */
JNIEXPORT jint JNICALL Java_com_powsybl_math_matrix_StaticLUDecomposition_factor(JNIEnv* env, jclass, jlong h, jdoubleArray j_ax) {
    try {
        powsybl::jni::DoubleArray ax(env, j_ax);
        return handle(h)->factor(ax.get(), static_cast<int>(ax.length()));
    } catch (const std::exception& e) {
        powsybl::jni::throwMatrixException(env, e.what());
    } catch (...) {
        powsybl::jni::throwMatrixException(env, "Unknown exception");
    }
    return -1;
}

/*
 * Class:     com_powsybl_math_matrix_StaticLUDecomposition
 * Method:    solve
 * Signature: (J[DZ)V
 */
JNIEXPORT void JNICALL Java_com_powsybl_math_matrix_StaticLUDecomposition_solve(JNIEnv* env, jclass, jlong h, jdoubleArray j_b, jboolean transpose) {
    try {
        powsybl::jni::DoubleArray b(env, j_b);
        handle(h)->solve(b.get(), static_cast<int>(b.length()), 1, transpose);
    } catch (const std::exception& e) {
        powsybl::jni::throwMatrixException(env, e.what());
    } catch (...) {
        powsybl::jni::throwMatrixException(env, "Unknown exception");
    }
}

/*
 * Class:     com_powsybl_math_matrix_StaticLUDecomposition
 * Method:    solve2
 * Signature: (JIILjava/nio/ByteBuffer;Z)V
 */
JNIEXPORT void JNICALL Java_com_powsybl_math_matrix_StaticLUDecomposition_solve2(JNIEnv* env, jclass, jlong h, jint m, jint n, jobject j_b, jboolean transpose) {
    try {
        auto* b = static_cast<double*>(env->GetDirectBufferAddress(j_b));
        if (!b) {
            throw std::runtime_error("GetDirectBufferAddress error");
        }
        handle(h)->solve(b, m, n, transpose);
    } catch (const std::exception& e) {
        powsybl::jni::throwMatrixException(env, e.what());
    } catch (...) {
        powsybl::jni::throwMatrixException(env, "Unknown exception");
    }
}

/*
 * Class:     com_powsybl_math_matrix_StaticLUDecomposition
 * Method:    release
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_com_powsybl_math_matrix_StaticLUDecomposition_release(JNIEnv* env, jclass, jlong h) {
    try {
        delete reinterpret_cast<StaticLu*>(h);
    } catch (...) {
        powsybl::jni::throwMatrixException(env, "Unknown exception");
    }
}

#ifdef __cplusplus
}
#endif
