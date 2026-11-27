/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * @file ordering.hpp
 * @author Gautier Bureau <gautier.bureau at rte-france.com>
 */

#ifndef ORDERING_HPP
#define ORDERING_HPP

#include <vector>

namespace powsybl {

namespace math {

/**
 * A greedy minimum-fill order (Tinney's scheme 3) of the graph of A + A^T, A given in CSC: the next node eliminated
 * is the one that adds the fewest edges, fill plus degree, ties by the smaller degree. perm[k] is the k-th node
 * eliminated, and lowerCount the number of entries of L it leads to, the diagonal included.
 */
std::vector<int> minimumFillOrder(int n, const int* ap, const int* ai, long long& lowerCount);

}  // namespace math

}  // namespace powsybl

#endif  // ORDERING_HPP
