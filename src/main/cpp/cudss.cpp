/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 *
 * @file cudss.cpp
 *
 * JNI bindings for an LU decomposition backed by NVIDIA cuDSS, mirroring the
 * KLU implementation in lu.cpp. Built into a SEPARATE shared library
 * (libmathcudss) so the CUDA dependency never touches the CPU-only libmath.
 *
 * Matrix convention: the (ap, ai, ax) arrays are the CSC of a matrix M (powsybl
 * SparseMatrix layout). They are fed to cuDSS AS CSR, so cuDSS factorizes
 * N = M^T. A plain cuDSS solve therefore computes M^T x = b, i.e.
 * solve(transpose=true) == solveTransposed, which is the path open-loadflow
 * uses. cuDSS has no transpose solve, so the (unused on sparse) non-transposed
 * solve throws.
 */
#include <algorithm>
#include <limits>
#include <string>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>
#include <cuda_runtime.h>
#include <cudss.h>
#include "jniwrapper.hpp"

#define CUDA_CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(e_)); } } while (0)
#define CUDSS_CHECK(x) do { cudssStatus_t s_ = (x); if (s_ != CUDSS_STATUS_SUCCESS) { \
    throw std::runtime_error("cuDSS error, status " + std::to_string((int) s_)); } } while (0)

namespace {

// Number of iterative-refinement steps. cuDSS default pivoting leaves ~1e-3
// error on unsymmetric systems; 2 steps recover KLU-grade accuracy.
constexpr int IR_N_STEPS = 2;

class CuDssContext {
public:
    CuDssContext() = default;
    CuDssContext(const CuDssContext&) = delete;
    CuDssContext& operator=(const CuDssContext&) = delete;
    ~CuDssContext() { destroy(); }

    // ensure the rhs/solution device buffers and dense matrices match (rows, cols)
    void ensureRhs(int rows, int cols) {
        size_t needed = static_cast<size_t>(rows) * static_cast<size_t>(cols);
        if (needed > rhsCapacity) {
            // Invalidate every piece of dependent state BEFORE the calls that can
            // throw. Otherwise a failed cudaMalloc (GPU OOM) would leave a stale
            // rhsCapacity next to a null d_b, and matB/matX bound to freed device
            // memory, so a later solve would fault instead of retrying cleanly.
            if (matB) { cudssMatrixDestroy(matB); matB = nullptr; }
            if (matX) { cudssMatrixDestroy(matX); matX = nullptr; }
            if (d_b) { cudaFree(d_b); d_b = nullptr; }
            if (d_x) { cudaFree(d_x); d_x = nullptr; }
            rhsCapacity = 0;
            rhsRows = 0;
            rhsCols = 0;
            CUDA_CHECK(cudaMalloc(&d_b, needed * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&d_x, needed * sizeof(double)));
            rhsCapacity = needed;
        }
        if (matB == nullptr || matX == nullptr || rhsRows != rows || rhsCols != cols) {
            if (matB) { cudssMatrixDestroy(matB); matB = nullptr; }
            if (matX) { cudssMatrixDestroy(matX); matX = nullptr; }
            rhsRows = 0;
            rhsCols = 0;
            CUDSS_CHECK(cudssMatrixCreateDn(&matB, rows, cols, rows, d_b, CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR));
            CUDSS_CHECK(cudssMatrixCreateDn(&matX, rows, cols, rows, d_x, CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR));
            rhsRows = rows;
            rhsCols = cols;
        }
    }

    // The refactorization path reuses the sparsity pattern captured at init, so an
    // update that silently changed it would apply the new values to the wrong
    // positions. Comparing nonzero counts alone does not catch a pattern change
    // that preserves nnz (e.g. a topology change), hence the full comparison.
    void checkSameStructure(const int* newAp, size_t apLength, const int* newAi, size_t aiCount) const {
        if (apLength != hostAp.size() || aiCount != hostAi.size()) {
            throw std::runtime_error("Matrix structure changed since initial decomposition "
                                     "(nonzero count differs)");
        }
        if (!std::equal(hostAp.begin(), hostAp.end(), newAp) ||
            !std::equal(hostAi.begin(), hostAi.end(), newAi)) {
            throw std::runtime_error("Matrix structure changed since initial decomposition "
                                     "(sparsity pattern differs)");
        }
    }

    void destroy() {
        if (matA) { cudssMatrixDestroy(matA); matA = nullptr; }
        if (matB) { cudssMatrixDestroy(matB); matB = nullptr; }
        if (matX) { cudssMatrixDestroy(matX); matX = nullptr; }
        if (data && handle) { cudssDataDestroy(handle, data); data = nullptr; }
        if (config) { cudssConfigDestroy(config); config = nullptr; }
        if (handle) { cudssDestroy(handle); handle = nullptr; }
        if (stream) { cudaStreamDestroy(stream); stream = nullptr; }
        if (d_ap) { cudaFree(d_ap); d_ap = nullptr; }
        if (d_ai) { cudaFree(d_ai); d_ai = nullptr; }
        if (d_ax) { cudaFree(d_ax); d_ax = nullptr; }
        if (d_b) { cudaFree(d_b); d_b = nullptr; }
        if (d_x) { cudaFree(d_x); d_x = nullptr; }
    }

    cudaStream_t stream = nullptr;
    cudssHandle_t handle = nullptr;
    cudssConfig_t config = nullptr;
    cudssData_t data = nullptr;
    cudssMatrix_t matA = nullptr;
    cudssMatrix_t matB = nullptr;
    cudssMatrix_t matX = nullptr;
    int* d_ap = nullptr;
    int* d_ai = nullptr;
    double* d_ax = nullptr;
    double* d_b = nullptr;
    double* d_x = nullptr;
    std::vector<int> hostAp;  // sparsity pattern captured at init, to validate update()
    std::vector<int> hostAi;
    int n = 0;
    int nnz = 0;
    size_t rhsCapacity = 0;
    int rhsRows = 0;
    int rhsCols = 0;
};

class CuDssContextManager {
public:
    CuDssContext& createContext(const std::string& id) {
        std::lock_guard<std::mutex> lk(_mutex);
        if (_contexts.find(id) != _contexts.end()) {
            throw std::runtime_error("Context " + id + " already exists");
        }
        auto it = _contexts.insert(std::make_pair(id, std::unique_ptr<CuDssContext>(new CuDssContext())));
        return *it.first->second;
    }

    CuDssContext& findContext(const std::string& id) {
        std::lock_guard<std::mutex> lk(_mutex);
        auto it = _contexts.find(id);
        if (it == _contexts.end()) {
            throw std::runtime_error("Context " + id + " not found");
        }
        return *it->second;
    }

    void removeContext(const std::string& id) {
        std::lock_guard<std::mutex> lk(_mutex);
        _contexts.erase(id);
    }

private:
    std::map<std::string, std::unique_ptr<CuDssContext>> _contexts;
    std::mutex _mutex;
};

std::unique_ptr<CuDssContextManager> MANAGER(new CuDssContextManager());

// cuDSS signals a zero pivot through CUDSS_DATA_INFO while cudssExecute itself
// still returns CUDSS_STATUS_SUCCESS, so checking the status is not enough to
// notice a singular matrix. The caller must have synchronized the stream first.
int getInfo(CuDssContext& ctx) {
    int info = 0;
    size_t written = 0;
    CUDSS_CHECK(cudssDataGet(ctx.handle, ctx.data, CUDSS_DATA_INFO, &info, sizeof(info), &written));
    return info;
}

// Mirrors the KLU backend, which throws on KLU_SINGULAR rather than handing back a
// silently unusable factorization.
void checkInfo(CuDssContext& ctx, const char* phase) {
    int info = getInfo(ctx);
    if (info != 0) {
        throw std::runtime_error(std::string(phase) + " error, matrix is singular (cuDSS info "
                                 + std::to_string(info) + ")");
    }
}

void solveInto(CuDssContext& ctx, double* host, int rows, int cols, bool transpose) {
    if (!transpose) {
        throw std::runtime_error("cuDSS backend supports only the transposed solve "
                                 "(non-transposed sparse solve is not implemented)");
    }
    if (rows != ctx.n) {
        throw std::runtime_error("Right-hand side size " + std::to_string(rows) +
                                 " does not match matrix order " + std::to_string(ctx.n));
    }
    if (cols <= 0) {
        throw std::runtime_error("Invalid number of right-hand sides: " + std::to_string(cols));
    }
    ctx.ensureRhs(rows, cols);
    size_t total = static_cast<size_t>(rows) * static_cast<size_t>(cols);
    CUDA_CHECK(cudaMemcpyAsync(ctx.d_b, host, total * sizeof(double), cudaMemcpyHostToDevice, ctx.stream));
    CUDSS_CHECK(cudssExecute(ctx.handle, CUDSS_PHASE_SOLVE, ctx.config, ctx.data, ctx.matA, ctx.matX, ctx.matB));
    CUDA_CHECK(cudaMemcpyAsync(host, ctx.d_x, total * sizeof(double), cudaMemcpyDeviceToHost, ctx.stream));
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
}

}  // namespace

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Class:     com_powsybl_math_matrix_CuDssLUDecomposition
 * Method:    init
 * Signature: (Ljava/lang/String;[I[I[D)V
 */
JNIEXPORT void JNICALL Java_com_powsybl_math_matrix_CuDssLUDecomposition_init(JNIEnv* env, jobject, jstring j_id, jintArray j_ap, jintArray j_ai, jdoubleArray j_ax) {
    std::string id;
    bool created = false;
    try {
        id = powsybl::jni::StringUTF(env, j_id).toStr();
        powsybl::jni::IntArray ap(env, j_ap);
        powsybl::jni::IntArray ai(env, j_ai);
        powsybl::jni::DoubleArray ax(env, j_ax);

        if (ap.length() < 1) {
            throw std::runtime_error("Invalid column pointer array: at least one element expected");
        }
        int n = static_cast<int>(ap.length()) - 1;
        const int* apData = ap.get();
        // The nonzero count is ap[n], NOT ax.length(). powsybl's SparseMatrix hands
        // out the backing arrays of its Trove lists, whose length is the allocated
        // capacity and is routinely larger than the number of nonzeros.
        int nnz = apData[n];
        if (nnz < 0 || static_cast<size_t>(nnz) > ai.length() || static_cast<size_t>(nnz) > ax.length()) {
            throw std::runtime_error("Declared nonzero count " + std::to_string(nnz) +
                                     " does not fit in the row index (" + std::to_string(ai.length()) +
                                     ") and value (" + std::to_string(ax.length()) + ") arrays");
        }

        CuDssContext& ctx = MANAGER->createContext(id);
        created = true;
        ctx.n = n;
        ctx.nnz = nnz;
        ctx.hostAp.assign(apData, apData + n + 1);
        ctx.hostAi.assign(ai.get(), ai.get() + nnz);

        CUDA_CHECK(cudaStreamCreate(&ctx.stream));
        CUDSS_CHECK(cudssCreate(&ctx.handle));
        CUDSS_CHECK(cudssSetStream(ctx.handle, ctx.stream));
        CUDSS_CHECK(cudssConfigCreate(&ctx.config));
        int irSteps = IR_N_STEPS;
        CUDSS_CHECK(cudssConfigSet(ctx.config, CUDSS_CONFIG_IR_N_STEPS, &irSteps, sizeof(irSteps)));
        CUDSS_CHECK(cudssDataCreate(ctx.handle, &ctx.data));

        CUDA_CHECK(cudaMalloc(&ctx.d_ap, (ctx.n + 1) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&ctx.d_ai, ctx.nnz * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&ctx.d_ax, ctx.nnz * sizeof(double)));
        CUDA_CHECK(cudaMemcpyAsync(ctx.d_ap, ap.get(), (ctx.n + 1) * sizeof(int), cudaMemcpyHostToDevice, ctx.stream));
        CUDA_CHECK(cudaMemcpyAsync(ctx.d_ai, ai.get(), ctx.nnz * sizeof(int), cudaMemcpyHostToDevice, ctx.stream));
        CUDA_CHECK(cudaMemcpyAsync(ctx.d_ax, ax.get(), ctx.nnz * sizeof(double), cudaMemcpyHostToDevice, ctx.stream));

        // CSC of M passed as CSR => cuDSS factorizes N = M^T (see file header).
        CUDSS_CHECK(cudssMatrixCreateCsr(&ctx.matA, ctx.n, ctx.n, ctx.nnz,
                                         ctx.d_ap, nullptr, ctx.d_ai, ctx.d_ax,
                                         CUDSS_R_32I, CUDSS_R_32I, CUDSS_R_64F,
                                         CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO));
        ctx.ensureRhs(ctx.n, 1);

        CUDSS_CHECK(cudssExecute(ctx.handle, CUDSS_PHASE_ANALYSIS, ctx.config, ctx.data, ctx.matA, ctx.matX, ctx.matB));
        CUDSS_CHECK(cudssExecute(ctx.handle, CUDSS_PHASE_FACTORIZATION, ctx.config, ctx.data, ctx.matA, ctx.matX, ctx.matB));
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
        checkInfo(ctx, "cuDSS factorization");
    } catch (const std::exception& e) {
        // Drop the half-built context: it holds a stream, a cuDSS handle and device
        // allocations that nothing else would ever free (Java will not call release
        // after a failed init), and leaving it registered would make every retry with
        // the same id fail with "already exists" instead of reporting the real error.
        if (created) {
            MANAGER->removeContext(id);
        }
        powsybl::jni::throwMatrixException(env, e.what());
    } catch (...) {
        if (created) {
            MANAGER->removeContext(id);
        }
        powsybl::jni::throwMatrixException(env, "Unknown exception");
    }
}

/*
 * Class:     com_powsybl_math_matrix_CuDssLUDecomposition
 * Method:    update
 * Signature: (Ljava/lang/String;[I[I[DD)D
 */
JNIEXPORT jdouble JNICALL Java_com_powsybl_math_matrix_CuDssLUDecomposition_update(JNIEnv* env, jobject, jstring j_id, jintArray j_ap, jintArray j_ai, jdoubleArray j_ax, jdouble rgrowthThreshold) {
    try {
        std::string id = powsybl::jni::StringUTF(env, j_id).toStr();
        powsybl::jni::IntArray ap(env, j_ap);
        powsybl::jni::IntArray ai(env, j_ai);
        powsybl::jni::DoubleArray ax(env, j_ax);

        CuDssContext& ctx = MANAGER->findContext(id);
        if (ap.length() != static_cast<size_t>(ctx.n) + 1) {
            throw std::runtime_error("Matrix structure changed since initial decomposition "
                                     "(column count differs)");
        }
        const int* apData = ap.get();
        // As in init(), the nonzero count is ap[n]; the arrays themselves are Trove
        // backing arrays and may be longer.
        if (apData[ctx.n] != ctx.nnz) {
            throw std::runtime_error("Matrix structure changed since initial decomposition "
                                     "(nonzero count differs)");
        }
        if (static_cast<size_t>(ctx.nnz) > ai.length() || static_cast<size_t>(ctx.nnz) > ax.length()) {
            throw std::runtime_error("Row index / value arrays are shorter than the nonzero count");
        }
        ctx.checkSameStructure(apData, ap.length(), ai.get(), static_cast<size_t>(ctx.nnz));

        // structure unchanged: refresh values in place and refactorize
        CUDA_CHECK(cudaMemcpyAsync(ctx.d_ax, ax.get(), ctx.nnz * sizeof(double), cudaMemcpyHostToDevice, ctx.stream));

        // Mirror the KLU contract as closely as cuDSS allows. rgrowthThreshold <= 0
        // means the caller wants a full factorization with fresh pivoting: that is
        // CUDSS_PHASE_FACTORIZATION, which reuses the symbolic analysis but re-pivots
        // (the equivalent of klu_factor after klu_analyze). A positive threshold takes
        // the cheap CUDSS_PHASE_REFACTORIZATION, which reuses the pivot order.
        //
        // cuDSS exposes no reciprocal pivot growth, so KLU's "measure growth, redo the
        // factorization if it degraded" check cannot be reproduced exactly. What we can
        // detect is the reused pivot order hitting a zero pivot, and in that case we
        // fall back to a full factorization instead of returning a bad solve.
        if (rgrowthThreshold > 0) {
            CUDSS_CHECK(cudssExecute(ctx.handle, CUDSS_PHASE_REFACTORIZATION, ctx.config, ctx.data, ctx.matA, ctx.matX, ctx.matB));
            CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
            if (getInfo(ctx) != 0) {
                CUDSS_CHECK(cudssExecute(ctx.handle, CUDSS_PHASE_FACTORIZATION, ctx.config, ctx.data, ctx.matA, ctx.matX, ctx.matB));
                CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
                checkInfo(ctx, "cuDSS factorization");
            }
        } else {
            CUDSS_CHECK(cudssExecute(ctx.handle, CUDSS_PHASE_FACTORIZATION, ctx.config, ctx.data, ctx.matA, ctx.matX, ctx.matB));
            CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
            checkInfo(ctx, "cuDSS factorization");
        }
    } catch (const std::exception& e) {
        powsybl::jni::throwMatrixException(env, e.what());
    } catch (...) {
        powsybl::jni::throwMatrixException(env, "Unknown exception");
    }
    // cuDSS has no reciprocal pivot growth metric. Reporting NaN ("not available")
    // rather than a plausible-looking 1.0 keeps a caller's `rgrowth < threshold`
    // health check from silently reading as "healthy" on every single update.
    return std::numeric_limits<double>::quiet_NaN();
}

/*
 * Class:     com_powsybl_math_matrix_CuDssLUDecomposition
 * Method:    release
 * Signature: (Ljava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_com_powsybl_math_matrix_CuDssLUDecomposition_release(JNIEnv* env, jobject, jstring j_id) {
    try {
        std::string id = powsybl::jni::StringUTF(env, j_id).toStr();
        CuDssContext& ctx = MANAGER->findContext(id);
        ctx.destroy();
        MANAGER->removeContext(id);
    } catch (const std::exception& e) {
        powsybl::jni::throwMatrixException(env, e.what());
    } catch (...) {
        powsybl::jni::throwMatrixException(env, "Unknown exception");
    }
}

/*
 * Class:     com_powsybl_math_matrix_CuDssLUDecomposition
 * Method:    solve
 * Signature: (Ljava/lang/String;[DZ)V
 */
JNIEXPORT void JNICALL Java_com_powsybl_math_matrix_CuDssLUDecomposition_solve(JNIEnv* env, jobject, jstring j_id, jdoubleArray j_b, jboolean transpose) {
    try {
        std::string id = powsybl::jni::StringUTF(env, j_id).toStr();
        powsybl::jni::DoubleArray b(env, j_b);
        CuDssContext& ctx = MANAGER->findContext(id);
        solveInto(ctx, b.get(), static_cast<int>(b.length()), 1, transpose);
    } catch (const std::exception& e) {
        powsybl::jni::throwMatrixException(env, e.what());
    } catch (...) {
        powsybl::jni::throwMatrixException(env, "Unknown exception");
    }
}

/*
 * Class:     com_powsybl_math_matrix_CuDssLUDecomposition
 * Method:    solve2
 * Signature: (Ljava/lang/String;IILjava/nio/ByteBuffer;Z)V
 */
JNIEXPORT void JNICALL Java_com_powsybl_math_matrix_CuDssLUDecomposition_solve2(JNIEnv* env, jobject, jstring j_id, jint m, jint n, jobject j_b, jboolean transpose) {
    try {
        std::string id = powsybl::jni::StringUTF(env, j_id).toStr();
        auto* b = static_cast<double*>(env->GetDirectBufferAddress(j_b));
        if (!b) {
            throw std::runtime_error("GetDirectBufferAddress error");
        }
        CuDssContext& ctx = MANAGER->findContext(id);
        solveInto(ctx, b, m, n, transpose);
    } catch (const std::exception& e) {
        powsybl::jni::throwMatrixException(env, e.what());
    } catch (...) {
        powsybl::jni::throwMatrixException(env, "Unknown exception");
    }
}

#ifdef __cplusplus
}
#endif
