/*
 * vineyard_ext.cpp — pybind11 wrapper around GUDHI's persistence Matrix
 * with vine_swap support for vineyard computation.
 *
 * Uses a **chain matrix** (is_of_boundary_type = false) with:
 *   has_column_pairings  = true   (stores the barcode)
 *   has_vine_update      = true   (enables vine_swap)
 *   has_map_column_container = true  (required for chain + vine)
 *   has_row_access        = true  (required for chain vine swaps)
 *
 * The Matrix<> wrapper applies Position_to_index_overlay for chain matrices,
 * so vine_swap(PosIdx) works uniformly — the overlay translates PosIdx → MatIdx
 * internally.
 *
 * Build (adjust include paths to your GUDHI install):
 *
 *   c++ -O2 -shared -std=c++17 -fPIC \
 *       $(python3 -m pybind11 --includes) \
 *       vineyard_ext.cpp -o vineyard_ext$(python3-config --extension-suffix) \
 *       -I<path-to-gudhi>/include -I<path-to-boost>
 *
 * Requires: GUDHI >= 3.10.0, pybind11, Boost
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>
#include <algorithm>
#include <numeric>
#include <tuple>
#include <cmath>
#include <map>
#include <set>
#include <limits>
#include <stdexcept>
#include <iostream>

#include <gudhi/Matrix.h>
#include <gudhi/persistence_matrix_options.h>

namespace py = pybind11;

using namespace Gudhi::persistence_matrix;

/* =========================================================================
 * Option struct: chain matrix with vineyard (vine swap) support.
 *
 * Inherits from Default_options (which provides Index, Dimension, and all
 * the other required fields) and overrides only what we need.
 *
 * Key difference from Multi_persistence_options (which is RU / boundary):
 *   is_of_boundary_type = false  →  Matrix<> instantiates Chain_matrix
 *
 * Chain matrix + vine_swap additionally requires:
 *   has_map_column_container = true
 *   has_row_access           = true
 * ======================================================================= */
struct Vineyard_chain_options : Default_options<Column_types::INTRUSIVE_SET, true> {
    /* Chain matrix (not boundary/RU). */
    static const bool is_of_boundary_type = false;

    /* Vineyard support. */
    static const bool has_vine_update     = true;
    static const bool has_column_pairings = true;

    /* Required for chain + vine_swap. */
    static const bool has_map_column_container = true;
    static const bool has_row_access    = true;
    static const bool has_removable_rows = true;

    /* POSITION indexation enables vine_swap(pos) for chain matrices —
     * the overlay translates PosIdx → MatIdx internally.  */
    static const Column_indexation_types column_indexation_type =
        Column_indexation_types::POSITION;
};

/* The vineyard matrix type — wraps Chain_matrix with vine-swap support. */
using VineyardMatrix = Matrix<Vineyard_chain_options>;

/* =========================================================================
 * Rips-filtration helpers (called from Python, but the heavy lifting can
 * also live entirely on the Python/GUDHI side — see vineyard_rips.py).
 * ======================================================================= */

/* Compute the Rips filtration value for a simplex given its vertex set
 * and a pairwise-distance matrix.  For a k-simplex, the filtration value
 * is the maximum pairwise distance among its vertices.                    */
static double rips_filtration_value(
    const std::vector<int>& simplex,
    const std::vector<std::vector<double>>& dist_matrix)
{
    double mx = 0.0;
    for (size_t i = 0; i < simplex.size(); ++i)
        for (size_t j = i + 1; j < simplex.size(); ++j)
            mx = std::max(mx, dist_matrix[simplex[i]][simplex[j]]);
    return mx;
}

/* =========================================================================
 * Core vineyard function.
 *
 * Parameters
 * ----------
 * simplex_list : list[list[int]]
 *     Every simplex in the complex, as sorted vertex tuples.
 *     Must be *closed* (every face of every simplex is also present).
 *
 * filtration_per_frame : list[list[float]]
 *     filtration_per_frame[t][i] = filtration value of simplex_list[i]
 *     at time-step t.  All frames must cover the same simplices.
 *
 * Returns
 * -------
 * list[list[tuple(dim, birth_value, death_value)]]
 *     One barcode per frame.
 * ======================================================================= */
static py::list compute_vineyard(
    const std::vector<std::vector<int>>& simplex_list,
    const std::vector<std::vector<double>>& filtration_per_frame)
{
    const int n_simplices = static_cast<int>(simplex_list.size());
    const int n_frames    = static_cast<int>(filtration_per_frame.size());

    if (n_frames == 0 || n_simplices == 0) return py::list();

    /* ------------------------------------------------------------------
     * 1.  Combinatorial preprocessing (independent of filtration values).
     * ------------------------------------------------------------------ */

    // Dimension of each simplex and a reverse lookup from vertex-set → idx.
    std::vector<int> dims(n_simplices);
    std::map<std::vector<int>, int> simplex_to_idx;

    for (int i = 0; i < n_simplices; ++i) {
        std::vector<int> s = simplex_list[i];
        std::sort(s.begin(), s.end());
        simplex_to_idx[s] = i;
        dims[i] = static_cast<int>(s.size()) - 1;
    }

    // Boundary of each simplex (list of face indices in simplex_list).
    std::vector<std::vector<unsigned int>> boundaries(n_simplices);
    for (int i = 0; i < n_simplices; ++i) {
        if (dims[i] == 0) {
            boundaries[i] = {};
            continue;
        }
        std::vector<int> s = simplex_list[i];
        std::sort(s.begin(), s.end());
        std::vector<unsigned int> bdry;
        bdry.reserve(s.size());
        for (int k = 0; k < static_cast<int>(s.size()); ++k) {
            std::vector<int> face;
            face.reserve(s.size() - 1);
            for (int j = 0; j < static_cast<int>(s.size()); ++j)
                if (j != k) face.push_back(s[j]);
            auto it = simplex_to_idx.find(face);
            if (it != simplex_to_idx.end())
                bdry.push_back(static_cast<unsigned int>(it->second));
        }
        boundaries[i] = std::move(bdry);
    }

    /* ------------------------------------------------------------------
     * 2.  Frame 0 — sort by filtration & build the chain matrix.
     * ------------------------------------------------------------------ */
    const auto& filt0 = filtration_per_frame[0];

    // Compute initial filtration ordering.
    std::vector<int> order(n_simplices);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        if (filt0[a] != filt0[b]) return filt0[a] < filt0[b];
        if (dims[a] != dims[b])   return dims[a]  < dims[b];   // faces before cofaces
        return a < b;
    });

    // Inverse map: order_pos[simplex_idx] = position in filtration.
    std::vector<int> order_pos(n_simplices);
    for (int pos = 0; pos < n_simplices; ++pos)
        order_pos[order[pos]] = pos;

    // Insert boundaries into the chain matrix in filtration order.
    // The Matrix<> wrapper expects boundaries as position indices, sorted.
    VineyardMatrix matrix(static_cast<unsigned int>(n_simplices));
    for (int pos = 0; pos < n_simplices; ++pos) {
        int sx = order[pos];
        std::vector<unsigned int> bdry;
        bdry.reserve(boundaries[sx].size());
        for (unsigned int face_idx : boundaries[sx])
            bdry.push_back(static_cast<unsigned int>(order_pos[face_idx]));
        std::sort(bdry.begin(), bdry.end());
        matrix.insert_boundary(bdry, dims[sx]);
    }

    /* ------------------------------------------------------------------
     * Helper: read the current barcode and translate PosIdx → filtration
     * values using the current ordering.
     * ------------------------------------------------------------------ */
    auto extract_barcode = [&](const std::vector<int>& cur_order,
                               const std::vector<double>& filt) -> py::list
    {
        const auto& barcode = matrix.get_current_barcode();
        py::list frame_bars;
        for (const auto& bar : barcode) {
            unsigned int birth_pos = bar.birth;
            unsigned int death_pos = bar.death;
            int dim = bar.dim;

            double bval = (birth_pos < static_cast<unsigned int>(n_simplices))
                              ? filt[cur_order[birth_pos]]
                              : 0.0;
            double dval = (death_pos >= static_cast<unsigned int>(n_simplices))
                              ? std::numeric_limits<double>::infinity()
                              : filt[cur_order[death_pos]];

            frame_bars.append(py::make_tuple(dim, bval, dval));
        }
        return frame_bars;
    };

    // Collect frame-0 barcode.
    py::list all_barcodes;
    all_barcodes.append(extract_barcode(order, filt0));

    /* ------------------------------------------------------------------
     * 3.  Frames 1 … N-1 — vineyard via bubble-sort + vine_swap.
     * ------------------------------------------------------------------ */
    std::vector<int> current_order = order;

    for (int frame = 1; frame < n_frames; ++frame) {
        const auto& filt = filtration_per_frame[frame];

        // Bubble-sort current_order to match the new filtration.
        // Every adjacent swap in the ordering calls matrix.vine_swap(pos),
        // which updates the chain basis and barcode in O(column-size) time.
        bool swapped = true;
        while (swapped) {
            swapped = false;
            for (int pos = 0; pos < n_simplices - 1; ++pos) {
                int sx_a = current_order[pos];
                int sx_b = current_order[pos + 1];

                // Determine if (pos, pos+1) are out of order under new filt.
                bool need_swap = false;
                if (filt[sx_a] > filt[sx_b]) {
                    need_swap = true;
                } else if (filt[sx_a] == filt[sx_b]) {
                    if (dims[sx_a] > dims[sx_b])
                        need_swap = true;
                    else if (dims[sx_a] == dims[sx_b] && sx_a > sx_b)
                        need_swap = true;
                }

                if (!need_swap) continue;

                // Guard: never swap a face past its coface — that would
                // violate the requirement that faces precede cofaces.
                if (dims[sx_a] != dims[sx_b]) {
                    int lo = (dims[sx_a] < dims[sx_b]) ? sx_a : sx_b;
                    int hi = (dims[sx_a] < dims[sx_b]) ? sx_b : sx_a;
                    bool face_coface = false;
                    for (unsigned int f : boundaries[hi]) {
                        if (static_cast<int>(f) == lo) {
                            face_coface = true;
                            break;
                        }
                    }
                    if (face_coface) continue;
                }

                // Perform the vine swap at position pos.
                // For the chain matrix, vine_swap returns the new MatIdx
                // of the column that was at position pos.  We don't need
                // the return value — the overlay keeps track internally.
                matrix.vine_swap(pos);
                std::swap(current_order[pos], current_order[pos + 1]);
                swapped = true;
            }
        }

        all_barcodes.append(extract_barcode(current_order, filt));
    }

    return all_barcodes;
}


/* =========================================================================
 * Convenience: build Rips simplicial complex from a distance matrix up to
 * a given max_edge_length and max dimension.  Returns (simplex_list,
 * filtration_values) matching the format expected by compute_vineyard.
 *
 * This is a simple flag-complex construction, so the user doesn't need to
 * depend on GUDHI's Python SimplexTree just to build the input.
 * ======================================================================= */
static std::pair<std::vector<std::vector<int>>, std::vector<double>>
build_rips_complex(const std::vector<std::vector<double>>& dist_matrix,
                   double max_edge_length,
                   int max_dim)
{
    const int n = static_cast<int>(dist_matrix.size());

    // Vertices.
    std::vector<std::vector<int>> simplices;
    std::vector<double> filtrations;
    for (int i = 0; i < n; ++i) {
        simplices.push_back({i});
        filtrations.push_back(0.0);  // vertices are born at 0
    }

    // Edges.
    std::vector<std::tuple<double, int, int>> edges;
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j)
            if (dist_matrix[i][j] <= max_edge_length)
                edges.emplace_back(dist_matrix[i][j], i, j);

    std::sort(edges.begin(), edges.end());
    for (auto& [d, i, j] : edges) {
        simplices.push_back({i, j});
        filtrations.push_back(d);
    }

    // Higher-dimensional simplices via flag expansion.
    // adjacency[v] = set of neighbors with index > v.
    std::vector<std::set<int>> adj(n);
    for (auto& [d, i, j] : edges) {
        adj[i].insert(j);
        adj[j].insert(i);
    }

    // Expand cliques dimension by dimension up to max_dim.
    // current_simplices = simplices of current dimension.
    if (max_dim >= 2) {
        // Start from edges.
        std::vector<std::vector<int>> prev_dim_simplices;
        for (auto& [d, i, j] : edges)
            prev_dim_simplices.push_back({i, j});

        for (int dim = 2; dim <= max_dim; ++dim) {
            std::vector<std::vector<int>> next_dim_simplices;
            for (const auto& sigma : prev_dim_simplices) {
                // Find vertices adjacent to ALL vertices in sigma.
                // Candidates: neighbors of sigma[0] with index > max(sigma).
                int max_v = *std::max_element(sigma.begin(), sigma.end());
                std::set<int> candidates = adj[sigma[0]];
                for (size_t k = 1; k < sigma.size(); ++k) {
                    std::set<int> inter;
                    std::set_intersection(
                        candidates.begin(), candidates.end(),
                        adj[sigma[k]].begin(), adj[sigma[k]].end(),
                        std::inserter(inter, inter.begin()));
                    candidates = std::move(inter);
                }

                for (int v : candidates) {
                    if (v <= max_v) continue;  // avoid duplicates
                    std::vector<int> tau = sigma;
                    tau.push_back(v);
                    std::sort(tau.begin(), tau.end());

                    double fval = rips_filtration_value(tau, dist_matrix);
                    if (fval <= max_edge_length) {
                        simplices.push_back(tau);
                        filtrations.push_back(fval);
                        next_dim_simplices.push_back(std::move(tau));
                    }
                }
            }
            prev_dim_simplices = std::move(next_dim_simplices);
            if (prev_dim_simplices.empty()) break;
        }
    }

    return {simplices, filtrations};
}


/* =========================================================================
 * Full pipeline: point clouds → Rips vineyard.
 *
 * Parameters
 * ----------
 * point_clouds : list of numpy arrays, shape (n_points, ambient_dim)
 *     One array per frame.  All must have the same n_points.
 * max_edge_length : float
 *     Rips threshold.  The complex is built once using the *maximum*
 *     distance that any edge attains across all frames, capped by this.
 * max_dim : int
 *     Maximum simplex dimension to include (e.g. 2 for H0 and H1).
 *
 * Returns
 * -------
 * list[list[tuple(dim, birth_value, death_value)]]
 *     One barcode per frame.
 * ======================================================================= */
static py::list compute_rips_vineyard(
    const std::vector<py::array_t<double>>& point_clouds,
    double max_edge_length,
    int max_dim)
{
    const int n_frames = static_cast<int>(point_clouds.size());
    if (n_frames == 0) return py::list();

    auto pc0 = point_clouds[0].unchecked<2>();
    const int n_pts = static_cast<int>(pc0.shape(0));
    const int ambient = static_cast<int>(pc0.shape(1));

    // Compute distance matrices for every frame.
    auto dist_matrix_for = [&](int frame) -> std::vector<std::vector<double>> {
        auto pc = point_clouds[frame].unchecked<2>();
        std::vector<std::vector<double>> D(n_pts, std::vector<double>(n_pts, 0.0));
        for (int i = 0; i < n_pts; ++i)
            for (int j = i + 1; j < n_pts; ++j) {
                double d2 = 0.0;
                for (int k = 0; k < ambient; ++k) {
                    double diff = pc(i, k) - pc(j, k);
                    d2 += diff * diff;
                }
                double d = std::sqrt(d2);
                D[i][j] = d;
                D[j][i] = d;
            }
        return D;
    };

    // Build the *union* distance matrix: take the max distance per edge
    // across all frames, then build the Rips complex on that.
    // This guarantees the combinatorial complex contains every simplex
    // that appears in any frame.
    std::vector<std::vector<double>> union_dist(n_pts, std::vector<double>(n_pts, 0.0));
    std::vector<std::vector<std::vector<double>>> all_dists(n_frames);
    for (int f = 0; f < n_frames; ++f) {
        all_dists[f] = dist_matrix_for(f);
        for (int i = 0; i < n_pts; ++i)
            for (int j = i + 1; j < n_pts; ++j) {
                double d = all_dists[f][i][j];
                if (d > union_dist[i][j]) {
                    union_dist[i][j] = d;
                    union_dist[j][i] = d;
                }
            }
    }

    // Build the flag complex from the union distance matrix.
    auto [simplex_list, _unused_filt] = build_rips_complex(union_dist, max_edge_length, max_dim);

    const int n_simplices = static_cast<int>(simplex_list.size());

    // Compute per-frame filtration values.
    std::vector<std::vector<double>> filtration_per_frame(n_frames);
    for (int f = 0; f < n_frames; ++f) {
        filtration_per_frame[f].resize(n_simplices);
        for (int i = 0; i < n_simplices; ++i) {
            filtration_per_frame[f][i] =
                rips_filtration_value(simplex_list[i], all_dists[f]);
        }
    }

    // Run the vineyard.
    return compute_vineyard(simplex_list, filtration_per_frame);
}


/* =========================================================================
 * pybind11 module definition.
 * ======================================================================= */
PYBIND11_MODULE(vineyard_ext, m) {
    m.doc() = "GUDHI chain-matrix vineyard computation for moving point clouds";

    m.def("compute_vineyard", &compute_vineyard,
          py::arg("simplex_list"),
          py::arg("filtration_per_frame"),
          R"doc(
Compute a vineyard over a fixed simplicial complex with varying filtration.

Parameters
----------
simplex_list : list[list[int]]
    All simplices (as vertex index lists).  Must be closed under taking faces.
filtration_per_frame : list[list[float]]
    filtration_per_frame[t][i] = filtration value of simplex_list[i] at frame t.

Returns
-------
list[list[tuple(dim, birth_value, death_value)]]
    One barcode per frame.
)doc");

    m.def("build_rips_complex", &build_rips_complex,
          py::arg("dist_matrix"),
          py::arg("max_edge_length"),
          py::arg("max_dim"),
          R"doc(
Build a Rips (flag) complex from a distance matrix.

Returns (simplex_list, filtration_values).
)doc");

    m.def("compute_rips_vineyard", &compute_rips_vineyard,
          py::arg("point_clouds"),
          py::arg("max_edge_length"),
          py::arg("max_dim"),
          R"doc(
Full pipeline: point clouds → Rips vineyard.

Parameters
----------
point_clouds : list of np.ndarray, each shape (n_points, dim)
    One point cloud per time frame.  All must have the same number of points.
max_edge_length : float
    Rips threshold for complex construction.
max_dim : int
    Maximum simplex dimension (e.g. 2 for H0 and H1).

Returns
-------
list[list[tuple(dim, birth_value, death_value)]]
    One barcode per frame.
)doc");
}