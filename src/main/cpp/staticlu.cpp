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
 * values do: the order (minimum fill or AMD of the symmetrized pattern, the one with the fewer operations), the exact
 * fill and the list of the operations of the factorization are computed once, a factorization only runs the list. A pivot too small, or a multiplier too large,
 * is counted: the caller then factorizes with pivoting instead. It is the CPU counterpart of the fast engine of a
 * batched LU on a GPU.
 *
 * The matrix A is given in CSC. It is A^T that is factorized: its rows are the columns of A, so A^T x = b is a direct
 * solve on the factors and A x = b a transposed one, as klu_tsolve and klu_solve are on A.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <queue>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <amd.h>
#include "jniwrapper.hpp"

namespace {

const double PIVOT_EPSILON = 1e-10;
const double MULTIPLIER_LIMIT = 1e8;
// the updates of a row by a pivot row are padded to a multiple of this many, so that the loop runs a whole number of
// unrolled iterations: fewer, and better predicted, loop exits on the short runs of a power network's factors
const int UNROLL = 4;

/*
 * A greedy minimum-fill order (Tinney's scheme 3) of the graph of M + M^T: the next node is the one whose elimination
 * adds the fewest edges, fill plus degree, ties by the smaller degree. It counts the operations of a factorization
 * better than a minimum degree, at the cost of a slower ordering, done once per pattern. The fills are kept up to
 * date: an elimination recomputes those of the neighbours only, and each edge it adds is one pair less to fill for
 * the other common neighbours of its ends.
 */
std::vector<int> minimumFillOrder(int n, const int* ap, const int* ai) {
    std::vector<std::vector<int>> adj(n);
    for (int j = 0; j < n; j++) {
        for (int e = ap[j]; e < ap[j + 1]; e++) {
            int i = ai[e];
            if (i != j) {
                adj[i].push_back(j);
                adj[j].push_back(i);
            }
        }
    }
    for (auto& a : adj) {
        std::sort(a.begin(), a.end());
        a.erase(std::unique(a.begin(), a.end()), a.end());
    }
    std::vector<int> mark(n, -1);
    int stamp = 0;
    std::vector<long long> fill(n);
    auto computeFill = [&](int v) {
        stamp++;
        for (int u : adj[v]) {
            mark[u] = stamp;
        }
        long long d = static_cast<long long>(adj[v].size());
        long long adjacentPairs = 0;
        for (int u : adj[v]) {
            for (int w : adj[u]) {
                if (mark[w] == stamp) {
                    adjacentPairs++;
                }
            }
        }
        return d * (d - 1) / 2 - adjacentPairs / 2;
    };
    auto key = [&](int v) {
        long long d = static_cast<long long>(adj[v].size());
        return (fill[v] + d) * (static_cast<long long>(n) + 1) + d;
    };
    using Item = std::pair<long long, int>;
    std::priority_queue<Item, std::vector<Item>, std::greater<Item>> queue;
    std::vector<long long> current(n);
    for (int v = 0; v < n; v++) {
        fill[v] = computeFill(v);
        current[v] = key(v);
        queue.push({current[v], v});
    }
    std::vector<char> eliminated(n, 0);
    std::vector<int> changed(n, -1);
    std::vector<int> changedList;
    std::vector<int> mark2(n, -1);
    int stamp2 = 0;
    std::vector<int> order;
    order.reserve(n);
    std::vector<std::pair<int, int>> newEdges;
    while (!queue.empty()) {
        long long k = queue.top().first;
        int v = queue.top().second;
        queue.pop();
        if (eliminated[v] || k != current[v]) {
            continue;
        }
        eliminated[v] = 1;
        order.push_back(v);
        int round = static_cast<int>(order.size());
        std::vector<int> nb = std::move(adj[v]);
        adj[v].clear();
        changedList.clear();
        newEdges.clear();
        // the neighbours become a clique, without v
        for (int u : nb) {
            auto& a = adj[u];
            a.erase(std::lower_bound(a.begin(), a.end(), v));
            stamp++;
            for (int w : a) {
                mark[w] = stamp;
            }
            size_t before = a.size();
            for (int w : nb) {
                if (w != u && mark[w] != stamp) {
                    a.push_back(w);
                    if (u < w) {
                        newEdges.push_back({u, w});
                    }
                }
            }
            if (a.size() != before) {
                std::sort(a.begin(), a.end());
            }
        }
        // a new edge (a, b) is one pair less to fill for each other common neighbour of a and b
        for (const auto& edge : newEdges) {
            int a = edge.first;
            int b = edge.second;
            stamp2++;
            for (int c : adj[a]) {
                mark2[c] = stamp2;
            }
            for (int c : adj[b]) {
                if (mark2[c] == stamp2) {
                    fill[c]--;
                    if (changed[c] != round) {
                        changed[c] = round;
                        changedList.push_back(c);
                    }
                }
            }
        }
        // the neighbours: their own fill recomputed (their adjacency changed)
        for (int u : nb) {
            fill[u] = computeFill(u);
            if (changed[u] != round) {
                changed[u] = round;
                changedList.push_back(u);
            }
        }
        for (int c : changedList) {
            current[c] = key(c);
            queue.push({current[c], c});
        }
    }
    return order;
}

std::vector<int> amdOrder(int n, const int* ap, const int* ai) {
    std::vector<int> perm(n);
    double info[AMD_INFO];
    if (amd_order(n, ap, ai, perm.data(), nullptr, info) < AMD_OK) {
        throw std::runtime_error("amd_order error");
    }
    return perm;
}

class StaticLu {
public:
    // The plan of M = A^T, whose rows are given by the CSC arrays (ap, ai) of A: on the order, minimum fill or AMD,
    // whose factorization has the fewer operations.
    StaticLu(int n, const int* ap, const int* ai);

    // Factorizes M of the values of A, in the order of the CSC arrays of the plan: the number of pivots too small and
    // of multipliers too large, 0 when the factors can be trusted.
    int factor(const double* ax, int nnz);

    // A^T x = b when transpose (direct on M), A x = b otherwise (transposed on M), for nrhs right-hand sides of ldim
    // rows each, overwritten by the solutions.
    void solve(double* b, int ldim, int nrhs, bool transpose);

    long long operationCount() const { return _opCount; }

private:
    // the plan on an order (new index -> M's), or only its operation count when countOnly
    long long plan(const int* ap, const int* ai, const std::vector<int>& perm, bool countOnly);
    // the elimination on the values in place: the number of pivots too small and of multipliers too large
    int eliminate();
#if (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
    int eliminateAvx512();
#endif
    void solveM(double* b);
    void solveMT(double* b);

    int _n;
    int _nnz;
    long long _opCount = 0;
    std::vector<int> _perm;          // new index -> M's
    std::vector<int> _rowStart;      // the LU pattern by rows, columns ascending, diagonal included
    std::vector<int> _col;
    std::vector<int> _diag;          // position of the diagonal of each row
    std::vector<int> _entryOfA;      // A's entry -> LU position
    std::vector<int> _fillPos;       // the LU positions that are not A's entries: zeroed before a factorization
    std::vector<int> _lowerStart;    // per row, its first lower entry in _lowerPos
    std::vector<int> _lowerPos;      // per lower entry, its LU position
    std::vector<int> _lowerDiag;     // ...the position of its divisor, the updates reading the values right after it
    std::vector<int> _opStart;       // ...its first update (one more entry: the end), a multiple of UNROLL
    std::vector<int> _opDst;         // per update: the position of (i, j), or the dummy slot for the padding
    std::vector<double> _lu;         // the values, then the dummy slot and room for the padded reads
    std::vector<double> _work;
};

StaticLu::StaticLu(int n, const int* ap, const int* ai)
    : _n(n),
      _nnz(ap[n]) {
    std::vector<int> amd = amdOrder(n, ap, ai);
    std::vector<int> fill = minimumFillOrder(n, ap, ai);
    bool fillIsBetter = plan(ap, ai, fill, true) <= plan(ap, ai, amd, true);
    plan(ap, ai, fillIsBetter ? fill : amd, false);
}

long long StaticLu::plan(const int* ap, const int* ai, const std::vector<int>& perm, bool countOnly) {
    int n = _n;
    std::vector<int> inv(n);
    for (int k = 0; k < n; k++) {
        inv[perm[k]] = k;
    }
    // the rows of B = P M P^T: row r of B is row perm[r] of M, its columns mapped by inv
    std::vector<std::vector<int>> bRows(n);
    for (int r = 0; r < n; r++) {
        int old = perm[r];
        for (int e = ap[old]; e < ap[old + 1]; e++) {
            bRows[r].push_back(inv[ai[e]]);
        }
        std::sort(bRows[r].begin(), bRows[r].end());
    }
    // the exact fill, row by row: row i of B united with U(k, > k) for each k < i of its pattern
    std::vector<std::vector<int>> rows(n);
    std::vector<int> upperCount(n, 0);
    std::vector<int> mark(n, -1);
    long long ops = 0;
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
            ops += upperCount[k];
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
        upperCount[i] = static_cast<int>(pattern.end() - std::upper_bound(pattern.begin(), pattern.end(), i));
        rows[i] = std::move(pattern);
    }
    if (countOnly) {
        return ops;
    }
    _opCount = ops;
    _perm = perm;
    _rowStart.assign(n + 1, 0);
    for (int i = 0; i < n; i++) {
        _rowStart[i + 1] = _rowStart[i] + static_cast<int>(rows[i].size());
    }
    int valueCount = _rowStart[n];
    _col.resize(valueCount);
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
    std::vector<char> ofA(valueCount, 0);
    for (int old = 0; old < n; old++) {
        for (int e = ap[old]; e < ap[old + 1]; e++) {
            _entryOfA[e] = position(inv[old], inv[ai[e]]);
            ofA[_entryOfA[e]] = 1;
        }
    }
    for (int q = 0; q < valueCount; q++) {
        if (!ofA[q]) {
            _fillPos.push_back(q);
        }
    }
    // the operation list, row by row, each lower entry in ascending column with the updates of its pivot row: they
    // read u(k, j) at the positions right after the diagonal of row k, the destinations are padded with a dummy slot
    int dummy = valueCount;
    _lowerStart.assign(n + 1, 0);
    for (int i = 0; i < n; i++) {
        _lowerStart[i] = static_cast<int>(_lowerPos.size());
        for (int q = _rowStart[i]; q < _diag[i]; q++) {
            int k = _col[q];
            _lowerPos.push_back(q);
            _lowerDiag.push_back(_diag[k]);
            _opStart.push_back(static_cast<int>(_opDst.size()));
            for (int s = _diag[k] + 1; s < _rowStart[k + 1]; s++) {
                _opDst.push_back(position(i, _col[s]));
            }
            while (_opDst.size() % UNROLL != 0) {
                _opDst.push_back(dummy);
            }
        }
    }
    _lowerStart[n] = static_cast<int>(_lowerPos.size());
    _opStart.push_back(static_cast<int>(_opDst.size()));
    // the padded reads of the last rows go past the values: the dummy slot and UNROLL more
    _lu.assign(static_cast<size_t>(valueCount) + 1 + UNROLL, 0.0);
    _work.assign(n, 0.0);
    return ops;
}

#if (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
#define STATICLU_AVX512 1
#include <immintrin.h>

bool hasAvx512() {
    static const bool has = __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
    return has;
}
#endif

// runs at least this long are vectorized when the CPU has AVX-512 (a multiple of UNROLL)
const int LONG_RUN = 16;

int StaticLu::factor(const double* ax, int nnz) {
    if (nnz != _nnz) {
        throw std::runtime_error("The values are not those of the pattern of the plan");
    }
    double* v = _lu.data();
    for (int q : _fillPos) {
        v[q] = 0.0;
    }
    for (int e = 0; e < nnz; e++) {
        v[_entryOfA[e]] = ax[e];
    }
#ifdef STATICLU_AVX512
    if (hasAvx512()) {
        return eliminateAvx512();
    }
#endif
    return eliminate();
}

int StaticLu::eliminate() {
    double* v = _lu.data();
    const int* lowerStart = _lowerStart.data();
    const int* lowerPos = _lowerPos.data();
    const int* lowerDiag = _lowerDiag.data();
    const int* opStart = _opStart.data();
    const int* dst = _opDst.data();
    int trouble = 0;
    for (int i = 0; i < _n; i++) {
        for (int q = lowerStart[i]; q < lowerStart[i + 1]; q++) {
            double l = v[lowerPos[q]] / v[lowerDiag[q]];
            if (!(std::fabs(l) < MULTIPLIER_LIMIT)) {
                trouble++;
            }
            v[lowerPos[q]] = l;
            const double* u = v + lowerDiag[q] + 1 - opStart[q];
            for (int o = opStart[q]; o < opStart[q + 1]; o += UNROLL) {
                double u0 = u[o];
                double u1 = u[o + 1];
                double u2 = u[o + 2];
                double u3 = u[o + 3];
                v[dst[o]] -= l * u0;
                v[dst[o + 1]] -= l * u1;
                v[dst[o + 2]] -= l * u2;
                v[dst[o + 3]] -= l * u3;
            }
        }
        if (!(std::fabs(v[_diag[i]]) > PIVOT_EPSILON)) {
            trouble++;
        }
    }
    return trouble;
}

#ifdef STATICLU_AVX512
// The same elimination compiled for AVX-512: the long runs 8 updates at a time with a gather and a scatter, the
// others as above (with fused multiply-adds: the factors differ from the scalar ones by roundings only).
__attribute__((target("avx512f,avx2,fma")))
int StaticLu::eliminateAvx512() {
    double* v = _lu.data();
    const int* lowerStart = _lowerStart.data();
    const int* lowerPos = _lowerPos.data();
    const int* lowerDiag = _lowerDiag.data();
    const int* opStart = _opStart.data();
    const int* dst = _opDst.data();
    int trouble = 0;
    for (int i = 0; i < _n; i++) {
        for (int q = lowerStart[i]; q < lowerStart[i + 1]; q++) {
            double l = v[lowerPos[q]] / v[lowerDiag[q]];
            if (!(std::fabs(l) < MULTIPLIER_LIMIT)) {
                trouble++;
            }
            v[lowerPos[q]] = l;
            const double* u = v + lowerDiag[q] + 1 - opStart[q];
            int o = opStart[q];
            int end = opStart[q + 1];
            if (end - o >= LONG_RUN) {
                __m512d l8 = _mm512_set1_pd(l);
                for (; o + 8 <= end; o += 8) {
                    __m256i idx = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(dst + o));
                    __m512d x = _mm512_i32gather_pd(idx, v, 8);
                    x = _mm512_fnmadd_pd(l8, _mm512_loadu_pd(u + o), x);
                    _mm512_i32scatter_pd(v, idx, x, 8);
                }
                if (o < end) {
                    // the last 4: a gather, the stores one by one
                    __m128i idx = _mm_loadu_si128(reinterpret_cast<const __m128i*>(dst + o));
                    __m256d x = _mm256_i32gather_pd(v, idx, 8);
                    x = _mm256_fnmadd_pd(_mm256_set1_pd(l), _mm256_loadu_pd(u + o), x);
                    alignas(32) double t[4];
                    _mm256_store_pd(t, x);
                    v[dst[o]] = t[0];
                    v[dst[o + 1]] = t[1];
                    v[dst[o + 2]] = t[2];
                    v[dst[o + 3]] = t[3];
                    continue;
                }
            }
            for (; o < end; o += UNROLL) {
                double u0 = u[o];
                double u1 = u[o + 1];
                double u2 = u[o + 2];
                double u3 = u[o + 3];
                v[dst[o]] -= l * u0;
                v[dst[o + 1]] -= l * u1;
                v[dst[o + 2]] -= l * u2;
                v[dst[o + 3]] -= l * u3;
            }
        }
        if (!(std::fabs(v[_diag[i]]) > PIVOT_EPSILON)) {
            trouble++;
        }
    }
    return trouble;
}
#endif

// M x = b: B = P M P^T = L U, so x = P^T U^-1 L^-1 P b
void StaticLu::solveM(double* b) {
    double* w = _work.data();
    const double* lu = _lu.data();
    const int* col = _col.data();
    for (int r = 0; r < _n; r++) {
        w[r] = b[_perm[r]];
    }
    for (int i = 0; i < _n; i++) {
        double s = w[i];
        for (int q = _rowStart[i]; q < _diag[i]; q++) {
            s -= lu[q] * w[col[q]];
        }
        w[i] = s;
    }
    for (int i = _n - 1; i >= 0; i--) {
        double s = w[i];
        for (int q = _diag[i] + 1; q < _rowStart[i + 1]; q++) {
            s -= lu[q] * w[col[q]];
        }
        w[i] = s / lu[_diag[i]];
    }
    for (int r = 0; r < _n; r++) {
        b[_perm[r]] = w[r];
    }
}

// M^T x = b: B^T = U^T L^T, solved by rows of the factors, scattering
void StaticLu::solveMT(double* b) {
    double* w = _work.data();
    const double* lu = _lu.data();
    const int* col = _col.data();
    for (int r = 0; r < _n; r++) {
        w[r] = b[_perm[r]];
    }
    for (int i = 0; i < _n; i++) {
        double y = w[i] / lu[_diag[i]];
        w[i] = y;
        for (int q = _diag[i] + 1; q < _rowStart[i + 1]; q++) {
            w[col[q]] -= lu[q] * y;
        }
    }
    for (int i = _n - 1; i >= 0; i--) {
        double z = w[i];
        for (int q = _rowStart[i]; q < _diag[i]; q++) {
            w[col[q]] -= lu[q] * z;
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

// A Java array of doubles, pinned rather than copied: nothing else is called while it is held
class CriticalDoubles {
public:
    CriticalDoubles(JNIEnv* env, jdoubleArray array, bool readOnly)
        : _env(env),
          _array(array),
          _length(env->GetArrayLength(array)),
          _mode(readOnly ? JNI_ABORT : 0) {
        _ptr = static_cast<double*>(env->GetPrimitiveArrayCritical(array, nullptr));
        if (!_ptr) {
            throw std::runtime_error("GetPrimitiveArrayCritical error");
        }
    }

    CriticalDoubles(const CriticalDoubles&) = delete;
    CriticalDoubles& operator=(const CriticalDoubles&) = delete;

    ~CriticalDoubles() {
        _env->ReleasePrimitiveArrayCritical(_array, _ptr, _mode);
    }

    double* get() const { return _ptr; }
    int length() const { return _length; }

private:
    JNIEnv* _env;
    jdoubleArray _array;
    int _length;
    jint _mode;
    double* _ptr;
};

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
        StaticLu* lu = handle(h);
        CriticalDoubles ax(env, j_ax, true);
        return lu->factor(ax.get(), ax.length());
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
        StaticLu* lu = handle(h);
        CriticalDoubles b(env, j_b, false);
        lu->solve(b.get(), b.length(), 1, transpose);
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
