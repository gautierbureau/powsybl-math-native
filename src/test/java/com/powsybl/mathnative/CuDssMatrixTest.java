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
     * columns carry NO diagonal entry: in a DC system every bus after the slack, whose inactive P
     * equation shifts the equation numbering by one. KLU pivots through that; cuDSS without matching
     * returned NaN on rte6515 (6530 of 6536 columns without a diagonal). This is a synthetic replica:
     * a 2-D grid of buses with susceptance-like values spread over five decades, the slack bus's
     * P equation replaced by the reference-angle identity, and the equations numbered so that no
     * column of the CSC (rows = angle variables, columns = equations) carries a diagonal entry.
     */
    @Test
    void solvesWithEmptyStructuralDiagonal() {
        assumeTrue(CuDssLUDecomposition.isAvailable(), "cuDSS native library not available");

        int side = 80;
        int n = side * side;
        int slack = n / 2 + side / 2;
        // equation numbering: equation e is the P balance of bus (e + 2) mod n; the slack bus's
        // slot carries the reference-angle identity (row = the slack variable). Bus e + 2 is never
        // bus e nor one of its four grid neighbours, so column e carries no entry in row e.
        double[][] cols = new double[n][];
        int[][] rows = new int[n][];
        java.util.Random random = new java.util.Random(6515);
        double[] bx = new double[n];   // horizontal susceptances, log-uniform in [0.5, 4e4]
        double[] by = new double[n];
        for (int i = 0; i < n; i++) {
            bx[i] = Math.exp(Math.log(0.5) + random.nextDouble() * Math.log(8e4));
            by[i] = Math.exp(Math.log(0.5) + random.nextDouble() * Math.log(8e4));
        }
        int nnz = 0;
        for (int e = 0; e < n; e++) {
            int bus = (e + 2) % n;
            java.util.TreeMap<Integer, Double> entries = new java.util.TreeMap<>();
            if (bus == slack) {
                entries.put(slack, 1.0);
            } else {
                int r = bus / side;
                int c = bus % side;
                double diag = 0;
                if (c > 0) {
                    entries.merge(bus - 1, -bx[bus - 1], Double::sum);
                    diag += bx[bus - 1];
                }
                if (c < side - 1) {
                    entries.merge(bus + 1, -bx[bus], Double::sum);
                    diag += bx[bus];
                }
                if (r > 0) {
                    entries.merge(bus - side, -by[bus - side], Double::sum);
                    diag += by[bus - side];
                }
                if (r < side - 1) {
                    entries.merge(bus + side, -by[bus], Double::sum);
                    diag += by[bus];
                }
                entries.merge(bus, diag, Double::sum);
            }
            rows[e] = entries.keySet().stream().mapToInt(Integer::intValue).toArray();
            cols[e] = entries.values().stream().mapToDouble(Double::doubleValue).toArray();
            nnz += rows[e].length;
        }
        int[] ap = new int[n + 1];
        int[] ai = new int[nnz];
        double[] ax = new double[nnz];
        int k = 0;
        int columnsWithDiagonal = 0;
        for (int e = 0; e < n; e++) {
            for (int q = 0; q < rows[e].length; q++) {
                if (rows[e][q] == e) {
                    columnsWithDiagonal++;
                }
                ai[k] = rows[e][q];
                ax[k] = cols[e][q];
                k++;
            }
            ap[e + 1] = k;
        }
        assertTrue(columnsWithDiagonal < n / 100, "the system must have an (almost) empty structural diagonal, got "
                + columnsWithDiagonal + " columns with one");
        double[] expected = new double[n];
        for (int i = 0; i < n; i++) {
            expected[i] = Math.sin(0.01 * i);
        }
        expected[slack] = 0;
        double[] b = new double[n];
        for (int e = 0; e < n; e++) {
            for (int q = ap[e]; q < ap[e + 1]; q++) {
                b[e] += ax[q] * expected[ai[q]];        // (M^T x)_e = sum over column e
            }
        }

        CuDssLUDecomposition lu = new CuDssLUDecomposition();
        String id = "test-empty-diagonal";
        lu.init(id, ap, ai, ax);
        lu.solve(id, b, true);
        lu.release(id);

        double maxErr = 0;
        for (int i = 0; i < n; i++) {
            assertFalse(Double.isNaN(b[i]), "cuDSS returned NaN on an empty structural diagonal (matching off?)");
            maxErr = Math.max(maxErr, Math.abs(b[i] - expected[i]));
        }
        assertTrue(maxErr < 1e-7, "max error " + maxErr);
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
