/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.mathnative;

import com.powsybl.math.matrix.CuDssLUDecomposition;
import com.powsybl.math.matrix.MatrixException;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
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

    /**
     * powsybl's {@code SparseMatrix} hands out the backing arrays of its Trove lists,
     * so {@code getRowIndices()} and {@code getValues()} are normally longer than the
     * nonzero count — that is the shape open-loadflow builds its Jacobian in. The
     * nonzero count must come from {@code ap[n]}, and the trailing capacity must be
     * ignored rather than fed to cuDSS as extra nonzeros.
     */
    @Test
    void trailingArrayCapacityIsIgnored() {
        assumeTrue(CuDssLUDecomposition.isAvailable(), "cuDSS native library not available");

        int[] paddedAi = Arrays.copyOf(AI, AI.length + 7);
        double[] paddedAx = Arrays.copyOf(AX, AX.length + 7);
        Arrays.fill(paddedAi, AI.length, paddedAi.length, 3);        // garbage tail
        Arrays.fill(paddedAx, AX.length, paddedAx.length, 999.0);

        double[] expected = {1, 2, 3, 4, 5};
        double[] b = transposeTimes(expected);

        CuDssLUDecomposition lu = new CuDssLUDecomposition();
        String id = "padded";
        lu.init(id, AP, paddedAi, paddedAx);
        lu.update(id, AP, paddedAi, paddedAx, 0);
        lu.solve(id, b, true);
        lu.release(id);

        assertArrayEquals(expected, b, EPSILON);
    }

    /**
     * A structurally singular matrix (column 1 is empty) must fail loudly, the way the
     * KLU backend throws on KLU_SINGULAR, rather than yielding a NaN solution.
     */
    @Test
    void singularMatrixThrows() {
        assumeTrue(CuDssLUDecomposition.isAvailable(), "cuDSS native library not available");

        int[] ap = {0, 1, 1, 2};
        int[] ai = {0, 2};
        double[] ax = {1.0, 1.0};

        CuDssLUDecomposition lu = new CuDssLUDecomposition();
        assertThrows(MatrixException.class, () -> lu.init("singular", ap, ai, ax));
    }

    /**
     * A failed init must not leave the id registered, otherwise every later attempt
     * would report "already exists" instead of the real error.
     */
    @Test
    void failedInitDoesNotKeepTheId() {
        assumeTrue(CuDssLUDecomposition.isAvailable(), "cuDSS native library not available");

        CuDssLUDecomposition lu = new CuDssLUDecomposition();
        String id = "retry";
        // row index array shorter than the nonzero count: rejected before any GPU work
        assertThrows(MatrixException.class, () -> lu.init(id, AP, new int[]{0}, AX));

        // the same id must still be usable
        lu.init(id, AP, AI, AX);
        lu.release(id);
    }

    /**
     * A sparsity pattern change that preserves the nonzero count must be rejected: the
     * refactorization reuses the pattern captured at init, so accepting it would apply
     * the new values to the wrong positions.
     */
    @Test
    void updateWithChangedPatternThrows() {
        assumeTrue(CuDssLUDecomposition.isAvailable(), "cuDSS native library not available");

        CuDssLUDecomposition lu = new CuDssLUDecomposition();
        String id = "pattern";
        lu.init(id, AP, AI, AX);

        int[] changedAi = AI.clone();
        changedAi[0] = 2; // same nnz, different pattern
        MatrixException e = assertThrows(MatrixException.class, () -> lu.update(id, AP, changedAi, AX, 0));
        assertTrue(e.getMessage().contains("structure changed"), e.getMessage());

        lu.release(id);
    }

    /**
     * A right-hand side whose size does not match the factorized order must be reported
     * as such, not as an opaque cuDSS status code.
     */
    @Test
    void solveWithWrongRhsSizeThrows() {
        assumeTrue(CuDssLUDecomposition.isAvailable(), "cuDSS native library not available");

        CuDssLUDecomposition lu = new CuDssLUDecomposition();
        String id = "rhs";
        lu.init(id, AP, AI, AX);

        double[] tooShort = new double[3];
        MatrixException e = assertThrows(MatrixException.class, () -> lu.solve(id, tooShort, true));
        assertTrue(e.getMessage().contains("does not match matrix order"), e.getMessage());

        lu.release(id);
    }

    /**
     * The non-transposed sparse solve is not implemented and must say so.
     */
    @Test
    void nonTransposedSolveThrows() {
        assumeTrue(CuDssLUDecomposition.isAvailable(), "cuDSS native library not available");

        CuDssLUDecomposition lu = new CuDssLUDecomposition();
        String id = "notranspose";
        lu.init(id, AP, AI, AX);

        double[] b = new double[AP.length - 1];
        assertThrows(MatrixException.class, () -> lu.solve(id, b, false));

        lu.release(id);
    }
}
