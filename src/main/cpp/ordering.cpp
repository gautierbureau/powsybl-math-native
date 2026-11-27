/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * @file ordering.cpp
 * @author Gautier Bureau <gautier.bureau at rte-france.com>
 *
 * The fills are kept up to date: an elimination recomputes those of the neighbours of the eliminated node only, and
 * each edge it adds is one pair less to fill for the other common neighbours of its ends.
 */

#include <algorithm>
#include <functional>
#include <queue>
#include <utility>
#include <vector>
#include "ordering.hpp"

namespace powsybl {

namespace math {

std::vector<int> minimumFillOrder(int n, const int* ap, const int* ai, long long& lowerCount) {
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
    lowerCount = n;
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
        lowerCount += static_cast<long long>(nb.size());
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

}  // namespace math

}  // namespace powsybl
