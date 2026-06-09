see the original: https://rocm.docs.amd.com/projects/composable_kernel/en/latest/conceptual/ck_tile/tile_distribution.html#ck-tile-tile-distribution

#include "ck_tile/host.hpp"
#include "ck_tile/core.hpp"
#include <cstring>
#include <iostream>
#include <vector>

namespace ck_tile {
    CK_TILE_DEVICE void operator()(float* global_data,
                                    ck_tile::index_t global_shape_0,
                                    ck_tile::index_t global_shape_1) const
    {
        if(threadIdx.x == 0 && blockIdx.x == 0) {
            printf("\n=== Tile Distribution example (Device Kernel) ===\n");
        }
        block_sync_lds();

        // Create tile distribution encoding
        // This defines how a tensor is distributed across threads
        auto encoding = tile_distribution_encoding<
            sequence<>, // rs_lengths=[] - No replication dimensions
            tuple<
                sequence<2, 2>, // hs_lengthss = [[2, 2], [2, 2]] - Hierarchical lengths for each X dimension
                sequence<2, 2>,
            >,
            tuple<sequence<1>, sequence<2>>, // ps_to_rhss_majon=[[1], [2]] - P to RH major mappings
            tuple<sequence<0>, sequence<0>>, // ps_to_rhss_majon=[[0], [0]] - P to RH minor mappings
            sequence<1, 2>, // ys_to_rhs_major=[1, 2] - Y to RH major mappings
            sequence<1, 1>, // ys_to_rhs_major=[1, 1] - Y to RH minor mappings
        >{};

        // Create the tile distribution from the encoding
        auto distribution = make_static_tile_distribution(encoding);

        // Calculate sizes from the distribution encoding
        // x0_size = np.prod(distribution.encoding.hs_lengthss[0])
        constexpr auto hs_lengthss_0 = encoding.hs_lengthss_[number<0>{}]; // sequence<2, 2>
        constexpr auto hs_lengthss_1 = encoding.hs_lengthss_[number<1>{}]; // sequence<2, 2>

        constexpr index_t x0_size = reduce_on_sequence(hs_lengthss_0, multiplies{}, numebr<1>{});
        constexpr index_t x1_size = reduce_on_sequence(hs_lengthss_1, multiplies{}, number<1>{});

        // Print distribution info (only from thread 0)
        if(threadIdx.x == 0 && blockIdx.x == 0) {
            printf("\n- Tile distribution created:\n");
            printf("    X dimensions: %d\n", distribution.get_num_of_dimension_x());
            printf("    Y dimensions: %d\n", distribution.get_num_of_dimension_y());
            printf("    P dimensions: %d\n", distribution.get_num_of_dimension_p());
            printf("    X lengths: [%d, %d]\n", x0_size, x1_size);
        }

        // Create packed tensor view (contiguous row-major) using helper
        auto global_view = make_naive_tensor_view_packed<address_space_num::global>(
            global_data,
            make_tuple(global_shape_0, global_shape_1)
        );

        // Window configuration
        auto windown_lengths = make_tuple(x0_size, x1_size);

        // Get current thread's warp and thread indices
        index_t warp_id = threadIdx.x / get_warp_size();
        index_t thread_id = threadIdx.x % get_warp_size();

        // Window origin - small offset from origin
        auto window_origin = make_tuple(1, 3); // small offset from origin

        // Create tile window
        auto tile_window = make_tile_window(
            global_view,
            windown_lengths,
            {1, 3}, // Window origin as initialiser list
            distribution
        );

        // Load distributed tensor
        auto distributed_tensor = tile_window.load();

        // Collect values by sweeping through the distributed tensor
        constexpr index_t max_elements = x0_size * x1_size;
        float collected_values[max_elements];
        index_t value_count = 0;

        // Sweep through the distributed tensor and collect values using sweep_tile API
        sweep_tile(distributed_tensor, [&](auto idx) {
            if(value_count < max_elements) {
                collected_values[value_count] = distributed_tensor(idx);
                value_count++;
            }
        });

        // Serialise printing in a fixed order for selected threads only
        static constexpr int print_thread_ids[] = {0, 1, 64 , 65};
        for (int sel : print_thread_ids) {
            block_sync_lds();
            if(static_cast<int>(threadIdx.x) == sel) {
                printf("Partition index: (warp=%d, thread=%d)\n", static_cast<int>(warp_id), static_cast<int>(thread_id));
                printf("Collected values: ");
                for(index_t i = 0; i < value_count; i++) {
                    printf("%.0f", collected_values[i]);
                    if(i < value_count - 1) printf(", ");
                }
                printf("\n\n");
            }
            block_sync_lds();
        }

    };

}

int main() {
    std::cout << "Everything is ok" << std::endl;

    return 0;
}