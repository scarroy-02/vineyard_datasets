/*
 * vineyard_ext.cpp — pybind11 wrapper around GUDHI's persistence Matrix
 * with vine_swap support for vineyard computation.
 *
 * Uses an RU boundary matrix (Multi_persistence_options):
 *   has_column_pairings = true
 *   has_vine_update     = true
 *   is_of_boundary_type = true  (inherited default)
 *
 * Build:
 *   c++ -O2 -shared -std=c++17 -fPIC \
 *       $(python3 -m pybind11 --includes) \
 *       vineyard_ext.cpp -o vineyard_ext$(python3-config --extension-suffix) \
 *       -I/usr/include
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <algorithm>
#include <numeric>
#include <tuple>
#include <cmath>

#include <gudhi/Matrix.h>
#include <gudhi/persistence_matrix_options.h>

namespace py = pybind11;

using namespace Gudhi::persistence_matrix;

// RU boundary matrix with column pairings + vine swap
using VineyardMatrix = Matrix<Multi_persistence_options<>>;

/*
 * Compute a vineyard over a fixed simplicial complex with varying filtration.
 *
 * simplex_list[i] = vertex indices of simplex i
 * filtration_per_frame[t][i] = filtration value of simplex i at frame t
 *
 * Returns: list of barcodes, one per frame.
 *   Each barcode = list of (dim, birth_filt_value, death_filt_value).
 */
static py::list compute_vineyard(
    const std::vector<std::vector<int>>& simplex_list,
    const std::vector<std::vector<double>>& filtration_per_frame)
{
    const int n_simplices = (int)simplex_list.size();
    const int n_frames = (int)filtration_per_frame.size();

    if (n_frames == 0 || n_simplices == 0) return py::list();

    // Dimension and boundary of each simplex
    std::vector<int> dims(n_simplices);
    std::vector<std::vector<unsigned int>> boundaries(n_simplices);

    // Map sorted vertex tuple → simplex index
    std::map<std::vector<int>, int> simplex_to_idx;
    for (int i = 0; i < n_simplices; i++) {
        std::vector<int> s = simplex_list[i];
        std::sort(s.begin(), s.end());
        simplex_to_idx[s] = i;
        dims[i] = (int)s.size() - 1;
    }

    // Compute boundaries (face indices in simplex_list)
    for (int i = 0; i < n_simplices; i++) {
        std::vector<int> s = simplex_list[i];
        std::sort(s.begin(), s.end());
        if (dims[i] == 0) {
            boundaries[i] = {};
        } else {
            std::vector<unsigned int> bdry;
            for (int k = 0; k < (int)s.size(); k++) {
                std::vector<int> face;
                for (int j = 0; j < (int)s.size(); j++) {
                    if (j != k) face.push_back(s[j]);
                }
                auto it = simplex_to_idx.find(face);
                if (it != simplex_to_idx.end()) {
                    bdry.push_back((unsigned int)it->second);
                }
            }
            boundaries[i] = bdry;
        }
    }

    // --- Frame 0: sort simplices and build initial matrix ---
    const auto& filt0 = filtration_per_frame[0];
    std::vector<int> order(n_simplices);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        if (filt0[a] != filt0[b]) return filt0[a] < filt0[b];
        if (dims[a] != dims[b]) return dims[a] < dims[b];
        return a < b;
    });

    // order_pos[simplex_idx] = position in filtration
    std::vector<int> order_pos(n_simplices);
    for (int pos = 0; pos < n_simplices; pos++) {
        order_pos[order[pos]] = pos;
    }

    // Insert boundaries in filtration order
    VineyardMatrix matrix;
    for (int pos = 0; pos < n_simplices; pos++) {
        int sx = order[pos];
        std::vector<unsigned int> bdry;
        for (unsigned int face_idx : boundaries[sx]) {
            bdry.push_back((unsigned int)order_pos[face_idx]);
        }
        std::sort(bdry.begin(), bdry.end());
        matrix.insert_boundary(bdry, dims[sx]);
    }

    // Helper: extract barcode, converting PosIdx to filtration values
    // Bar.get<0>() = birth (PosIdx), get<1>() = death (PosIdx), get<2>() = dim
    auto get_barcode_as_filt = [&](const std::vector<int>& cur_order,
                                    const std::vector<double>& filt) -> py::list
    {
        const auto& barcode = matrix.get_current_barcode();
        py::list frame_bars;
        for (const auto& bar : barcode) {
            auto birth_pos = bar.birth;
            auto death_pos = bar.death;
            int dim = bar.dim;

            double bval = (birth_pos < (unsigned int)n_simplices)
                          ? filt[cur_order[birth_pos]] : 0.0;
            double dval;
            if (death_pos >= (unsigned int)n_simplices) {
                dval = std::numeric_limits<double>::infinity();
            } else {
                dval = filt[cur_order[death_pos]];
            }
            frame_bars.append(py::make_tuple(dim, bval, dval));
        }
        return frame_bars;
    };

    py::list all_barcodes;
    all_barcodes.append(get_barcode_as_filt(order, filt0));

    // --- Frames 1..n-1: vine swaps via bubble sort ---
    std::vector<int> current_order = order;

    for (int frame = 1; frame < n_frames; frame++) {
        const auto& filt = filtration_per_frame[frame];

        // Bubble sort current_order by new filtration, calling vine_swap
        bool did_swap = true;
        while (did_swap) {
            did_swap = false;
            for (int pos = 0; pos < n_simplices - 1; pos++) {
                int sx_a = current_order[pos];
                int sx_b = current_order[pos + 1];

                bool need_swap = false;
                if (filt[sx_a] > filt[sx_b]) {
                    need_swap = true;
                } else if (filt[sx_a] == filt[sx_b]) {
                    if (dims[sx_a] > dims[sx_b]) {
                        need_swap = true;
                    } else if (dims[sx_a] == dims[sx_b] && sx_a > sx_b) {
                        need_swap = true;
                    }
                }

                if (need_swap) {
                    // Skip face/coface pairs (invalid swap for filtration)
                    bool is_face_coface = false;
                    if (dims[sx_a] != dims[sx_b]) {
                        int lo = (dims[sx_a] < dims[sx_b]) ? sx_a : sx_b;
                        int hi = (dims[sx_a] < dims[sx_b]) ? sx_b : sx_a;
                        for (unsigned int f : boundaries[hi]) {
                            if ((int)f == lo) { is_face_coface = true; break; }
                        }
                    }

                    if (!is_face_coface) {
                        matrix.vine_swap(pos);
                        std::swap(current_order[pos], current_order[pos + 1]);
                        did_swap = true;
                    }
                }
            }
        }

        all_barcodes.append(get_barcode_as_filt(current_order, filt));
    }

    return all_barcodes;
}


PYBIND11_MODULE(vineyard_ext, m) {
    m.doc() = "GUDHI-based vineyard computation using the persistence chain matrix with vine swaps";
    m.def("compute_vineyard", &compute_vineyard,
          py::arg("simplex_list"),
          py::arg("filtration_per_frame"),
          R"doc(
Compute a vineyard over a fixed simplicial complex with varying filtration values.

Parameters
----------
simplex_list : list of list[int]
    All simplices as lists of vertex indices.
filtration_per_frame : list of list[float]
    filtration_per_frame[t][i] = filtration value of simplex_list[i] at frame t.

Returns
-------
list of list of tuple(dim, birth_value, death_value)
    One barcode per frame.
)doc");
}
