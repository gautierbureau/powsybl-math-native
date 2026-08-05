/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.math.matrix;

import org.scijava.nativelib.NativeLoader;

import java.io.IOException;
import java.nio.ByteBuffer;

/**
 * cuDSS-backed sparse LU decomposition (test mirror of the powsybl-core class).
 *
 * <p>The native methods live in the separate {@code libmathcudss} library, loaded
 * lazily so this class is usable (with {@link #isAvailable()} == false) on a build
 * without cuDSS or a machine without a GPU.
 */
public class CuDssLUDecomposition {

    private static final boolean AVAILABLE = load();

    private static boolean load() {
        try {
            NativeLoader.loadLibrary("mathcudss");
            return true;
        } catch (IOException | UnsatisfiedLinkError e) {
            // built without cuDSS, or libcudss/libcudart not reachable by the loader
            return false;
        }
    }

    /**
     * @return true if the cuDSS native library could be loaded.
     */
    public static boolean isAvailable() {
        return AVAILABLE;
    }

    public native void init(String id, int[] ap, int[] ai, double[] ax);

    public native void release(String id);

    /**
     * Refreshes the matrix values and refactorizes. The sparsity pattern must be
     * identical to the one passed to {@link #init}; a {@link MatrixException} is
     * thrown otherwise.
     *
     * <p>A {@code rgrowthThreshold} of zero or less requests a full factorization
     * with fresh pivoting; a positive value takes the cheaper refactorization, which
     * reuses the pivot order computed at {@code init}.
     *
     * @return always {@code NaN}: cuDSS exposes no reciprocal pivot growth metric, so
     *         unlike the KLU backend this value carries no information and must not be
     *         compared against {@code rgrowthThreshold}. The native code falls back to
     *         a full factorization on its own when a reused pivot order fails.
     */
    public native double update(String id, int[] ap, int[] ai, double[] ax, double rgrowthThreshold);

    /**
     * Solves in place. Only {@code transpose == true} is supported: the cuDSS backend
     * factorizes M^T (the CSC arrays are read as CSR), so this computes M^T x = b.
     */
    public native void solve(String id, double[] b, boolean transpose);

    public native void solve2(String id, int m, int n, ByteBuffer b, boolean transpose);
}
