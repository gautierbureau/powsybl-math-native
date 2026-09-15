/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.mathnative;

import com.powsybl.math.matrix.CuDssLUDecomposition;
import org.junit.jupiter.api.Test;

import java.nio.ByteBuffer;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Validates the cuDSS LU binding on the 5x5 system used by {@link MatrixTest}.
 *
 * <p>The cuDSS backend factorizes M^T (the CSC arrays read as CSR), so a plain
 * solve computes M^T x = b — i.e. {@code solve(transpose=true)} == solveTransposed,
 * the path open-loadflow uses. The reference {@code b = M^T x_known} is computed
 * directly from the CSC arrays, so this test needs only libmathcudss (no KLU).
 *
 * Skipped when cuDSS is not available (build without it, or no GPU).
 */
class CuDssMatrixTest {

    private static final double EPSILON = 1e-9;

    // CSC of M (columnStart, rowIndices, values) — same system as MatrixTest.
    private static final int[] AP = {0, 2, 5, 9, 10, 12};
    private static final int[] AI = {0, 1, 0, 2, 4, 1, 2, 3, 4, 2, 1, 4};
    private static final double[] AX = {2.0, 3.0, 3.0, -1.0, 4.0, 4.0, -3.0, 1.0, 2.0, 2.0, 6.0, 1.0};

    private static double[] transposeTimes(double[] x) {
        int n = AP.length - 1;
        double[] b = new double[n];
        for (int col = 0; col < n; col++) {
            for (int k = AP[col]; k < AP[col + 1]; k++) {
                b[col] += AX[k] * x[AI[k]]; // (M^T x)_col = sum over column col of M
            }
        }
        return b;
    }

    @Test
    void solveTransposed() {
        assumeTrue(CuDssLUDecomposition.isAvailable(), "cuDSS native library not available");

        double[] expected = {1, 2, 3, 4, 5};
        double[] b = transposeTimes(expected); // so that M^T expected = b

        CuDssLUDecomposition lu = new CuDssLUDecomposition();
        String id = "test";
        lu.init(id, AP, AI, AX);
        lu.solve(id, b, true); // solves M^T x = b in place
        lu.release(id);

        assertArrayEquals(expected, b, EPSILON);
    }

    /**
     * OLF's Jacobian has rows = variables and columns = equations, numbered independently, so most
     * columns carry NO diagonal entry (a DC system: every bus after the slack). KLU pivots through
     * that; cuDSS without matching returned NaN. Same 5x5 system with its columns permuted so that
     * every column lacks its diagonal: column j' = original column PERM[j'].
     */
    @Test
    void solvesWithEmptyStructuralDiagonal() {
        assumeTrue(CuDssLUDecomposition.isAvailable(), "cuDSS native library not available");

        int[] perm = {2, 1, 4, 3, 0};
        int n = AP.length - 1;
        int[] ap = new int[n + 1];
        int[] ai = new int[AI.length];
        double[] ax = new double[AX.length];
        int k = 0;
        for (int col = 0; col < n; col++) {
            int src = perm[col];
            for (int q = AP[src]; q < AP[src + 1]; q++) {
                assertTrue(AI[q] != col, "the permuted system must have an empty structural diagonal");
                ai[k] = AI[q];
                ax[k] = AX[q];
                k++;
            }
            ap[col + 1] = k;
        }
        double[] expected = {1, 2, 3, 4, 5};
        double[] b = new double[n];
        for (int col = 0; col < n; col++) {
            for (int q = ap[col]; q < ap[col + 1]; q++) {
                b[col] += ax[q] * expected[ai[q]];
            }
        }

        CuDssLUDecomposition lu = new CuDssLUDecomposition();
        String id = "test-empty-diagonal";
        lu.init(id, ap, ai, ax);
        lu.solve(id, b, true);
        lu.release(id);

        for (double v : b) {
            assertFalse(Double.isNaN(v), "cuDSS returned NaN on an empty structural diagonal (matching off?)");
        }
        assertArrayEquals(expected, b, EPSILON);
    }

    /**
     * OLF's Woodbury security analysis solves an n x 0 dense right-hand side on every connected
     * component that owns no contingency element; cuDSS SIGFPEs on it unless the binding skips it.
     */
    @Test
    void zeroColumnDenseSolveIsANoOp() {
        assumeTrue(CuDssLUDecomposition.isAvailable(), "cuDSS native library not available");

        CuDssLUDecomposition lu = new CuDssLUDecomposition();
        String id = "test-zero-columns";
        lu.init(id, AP, AI, AX);
        lu.solve2(id, AP.length - 1, 0, ByteBuffer.allocateDirect(0), true);   // must not crash

        double[] expected = {1, 2, 3, 4, 5};
        double[] b = transposeTimes(expected);
        lu.solve(id, b, true);                                                 // and the context still works
        lu.release(id);
        assertArrayEquals(expected, b, EPSILON);
    }

    @Test
    void updateThenSolve() {
        assumeTrue(CuDssLUDecomposition.isAvailable(), "cuDSS native library not available");

        double[] expected = {5, 4, 3, 2, 1};
        double[] b = transposeTimes(expected);

        CuDssLUDecomposition lu = new CuDssLUDecomposition();
        String id = "test2";
        lu.init(id, AP, AI, AX);
        lu.update(id, AP, AI, AX, 0); // refactorize same values
        lu.solve(id, b, true);
        lu.release(id);

        assertArrayEquals(expected, b, EPSILON);
    }
}
