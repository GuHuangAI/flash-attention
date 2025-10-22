/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once
#include "namespace_config.h"

#include <cute/tensor.hpp>
#include <cstdint>

namespace FLASH_NAMESPACE {

using namespace cute;

template <typename Engine, typename Layout>
__forceinline__ __device__ void apply_mask(Tensor<Engine, Layout> &tensor, const int max_seqlen_k,
                                  const int col_idx_offset_ = 0) {
    // tensor has shape (nrow=(2, MMA_M), ncol=(2, MMA_N))
    static_assert(Layout::rank == 2, "Only support 2D Tensor");
    const int lane_id = threadIdx.x % 32;
    const int col_idx_offset = col_idx_offset_ + (lane_id % 4) * 2;
    #pragma unroll
    for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
        const int col_idx_base = col_idx_offset + nj * 8;
        #pragma unroll
        for (int j = 0; j < size<1, 0>(tensor); ++j) {
            const int col_idx = col_idx_base + j;
            if (col_idx >= max_seqlen_k) {
                // Without the "make_coord" we get wrong results
                #pragma unroll
                for (int mi = 0; mi < size<0>(tensor); ++mi) {
                    tensor(mi, make_coord(j, nj)) = -INFINITY;
                }
            }
        }
    }
}

template <bool HasWSLeft=true, typename Engine, typename Layout>
__forceinline__ __device__ void apply_mask_local(Tensor<Engine, Layout> &tensor, const int col_idx_offset_,
                                        const int max_seqlen_k, const int row_idx_offset,
                                        const int max_seqlen_q, const int warp_row_stride,
                                        const int window_size_left, const int window_size_right) {
    // tensor has shape (nrow=(2, MMA_M), ncol=(2, MMA_N))
    static_assert(Layout::rank == 2, "Only support 2D Tensor");
    const int lane_id = threadIdx.x % 32;
    const int col_idx_offset = col_idx_offset_ + (lane_id % 4) * 2;
    #pragma unroll
    for (int mi = 0; mi < size<0, 1>(tensor); ++mi) {
        const int row_idx_base = row_idx_offset + mi * warp_row_stride;
        #pragma unroll
        for (int i = 0; i < size<0, 0>(tensor); ++i) {
            const int row_idx = row_idx_base + i * 8;
            const int col_idx_limit_left = std::max(0, row_idx + max_seqlen_k - max_seqlen_q - window_size_left);
            const int col_idx_limit_right = std::min(max_seqlen_k, row_idx + 1 + max_seqlen_k - max_seqlen_q + window_size_right);
            #pragma unroll
            for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
                const int col_idx_base = col_idx_offset + nj * 8;
                #pragma unroll
                for (int j = 0; j < size<1, 0>(tensor); ++j) {
                    const int col_idx = col_idx_base + j;
                    if (col_idx >= col_idx_limit_right || (HasWSLeft && col_idx < col_idx_limit_left)) {
                        tensor(make_coord(i, mi), make_coord(j, nj)) = -INFINITY;
                    }
                }
            }
            // if (cute::thread0()) {
            //     printf("mi = %d, i = %d, row_idx = %d, max_seqlen_k = %d\n", mi, i, row_idx, max_seqlen_k);
            //     print(tensor(make_coord(i, mi), _));
            //     // print(tensor(_, j + nj * size<1, 0>(tensor)));
            // }
        }
    }
}

template <typename Engine, typename Layout>
__forceinline__ __device__ void apply_mask_causal(Tensor<Engine, Layout> &tensor, const int col_idx_offset_,
                                         const int max_seqlen_k, const int row_idx_offset,
                                         const int max_seqlen_q, const int warp_row_stride) {
    // Causal masking is equivalent to local masking with window_size_left = infinity and window_size_right = 0
    apply_mask_local</*HasWSLeft=*/false>(tensor, col_idx_offset_, max_seqlen_k, row_idx_offset,
                                          max_seqlen_q, warp_row_stride, -1, 0);
}

template <typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__forceinline__ __device__ void apply_mask_causal_w_idx(
    Tensor<Engine0, Layout0> &tensor, Tensor<Engine1, Layout1> const &idx_rowcol,
    const int col_idx_offset_, const int max_seqlen_k, const int row_idx_offset)
{
    // tensor has shape (nrow=(2, MMA_M), ncol=(2, MMA_N))
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 2, "Only support 2D Tensor");
    CUTE_STATIC_ASSERT_V(size<0>(tensor) == size<0>(idx_rowcol));
    CUTE_STATIC_ASSERT_V(size<1>(tensor) == size<1>(idx_rowcol));
    #pragma unroll
    for (int mi = 0; mi < size<0>(tensor); ++mi) {
        const int col_idx_limit = std::min(max_seqlen_k, 1 + row_idx_offset + get<0>(idx_rowcol(mi, 0)));
        #pragma unroll
        for (int ni = 0; ni < size<1, 1>(tensor); ++ni) {
            if (col_idx_offset_ + get<1>(idx_rowcol(0, ni)) >= col_idx_limit) {
                tensor(mi, ni) = -INFINITY;
            }
        }
        // if (cute::thread0()) {
        //     printf("ni = %d, j = %d, col_idx = %d, max_seqlen_k = %d\n", ni, j, col_idx, max_seqlen_k);
        //     print(tensor(_, make_coord(j, ni)));
        //     // print(tensor(_, j + ni * size<1, 0>(tensor)));
        // }
    }
}

template <bool Is_causal, bool Is_local, bool Has_alibi>
struct Mask {

    const int max_seqlen_k, max_seqlen_q;
    const int window_size_left, window_size_right;
    const float alibi_slope;
    const bool has_custom_mask;
    const bool custom_mask_is_additive;
    const bool custom_mask_is_bool;
    const int attn_mask_elem_size;
    const int attn_mask_seqlen_q;
    const int attn_mask_seqlen_k;
    const int64_t attn_mask_row_stride;
    const int64_t attn_mask_col_stride;
    const uint8_t *bool_mask_base;
    const float *float_mask_base;

    __forceinline__ __device__ Mask(const int max_seqlen_k, const int max_seqlen_q,
                                    const int window_size_left, const int window_size_right,
                                    const float alibi_slope=0.f,
                                    const void *attn_mask_ptr=nullptr,
                                    const int64_t attn_mask_batch_stride=0,
                                    const int64_t attn_mask_head_stride=0,
                                    const int64_t attn_mask_row_stride=0,
                                    const int64_t attn_mask_col_stride=0,
                                    const int attn_mask_elem_size=0,
                                    const int attn_mask_seqlen_q=0,
                                    const int attn_mask_seqlen_k=0,
                                    const bool attn_mask_is_additive=false,
                                    const bool attn_mask_is_bool=false,
                                    const int bidb=0,
                                    const int bidh=0)
        : max_seqlen_k(max_seqlen_k)
        , max_seqlen_q(max_seqlen_q)
        , window_size_left(window_size_left)
        , window_size_right(window_size_right)
        , alibi_slope(!Has_alibi ? 0.0 : alibi_slope)
        , has_custom_mask(attn_mask_ptr != nullptr)
        , custom_mask_is_additive(attn_mask_ptr != nullptr && attn_mask_is_additive)
        , custom_mask_is_bool(attn_mask_ptr != nullptr && attn_mask_is_bool)
        , attn_mask_elem_size(attn_mask_ptr == nullptr ? 0 : attn_mask_elem_size)
        , attn_mask_seqlen_q(attn_mask_ptr == nullptr ? 0 : attn_mask_seqlen_q)
        , attn_mask_seqlen_k(attn_mask_ptr == nullptr ? 0 : attn_mask_seqlen_k)
        , attn_mask_row_stride(attn_mask_ptr == nullptr ? 0 : attn_mask_row_stride)
        , attn_mask_col_stride(attn_mask_ptr == nullptr ? 0 : attn_mask_col_stride)
        , bool_mask_base(nullptr)
        , float_mask_base(nullptr) {
        if (has_custom_mask) {
            const char *base_ptr = reinterpret_cast<const char *>(attn_mask_ptr);
            const int64_t base_offset = static_cast<int64_t>(bidb) * attn_mask_batch_stride
                + static_cast<int64_t>(bidh) * attn_mask_head_stride;
            base_ptr += base_offset * attn_mask_elem_size;
            if (custom_mask_is_bool) {
                bool_mask_base = reinterpret_cast<const uint8_t *>(base_ptr);
            } else {
                float_mask_base = reinterpret_cast<const float *>(base_ptr);
            }
        }
    };

    template <typename T>
    __forceinline__ __device__ T load_mask_value(const T *row_ptr, const int64_t index) const {
        return row_ptr[index];
    }

    // Causal_mask: whether this particular iteration needs causal masking
    template <bool Causal_mask=false, bool Is_even_MN=true, typename Engine, typename Layout>
    __forceinline__ __device__ void apply_mask(Tensor<Engine, Layout> &tensor_,
                                               const int col_idx_offset_,
                                               const int row_idx_offset,
                                               const int warp_row_stride) {
        static_assert(!(Causal_mask && Is_local), "Cannot be both causal and local");
        static_assert(Layout::rank == 3, "Only support 3D Tensor");
        static_assert(decltype(size<0>(tensor_))::value == 4, "First dimension must be 4");
        const bool need_custom_mask = has_custom_mask;
        const bool need_masking = need_custom_mask || Has_alibi || Causal_mask || Is_local || !Is_even_MN;
        if (need_masking) {
            // Reshape tensor_ from (MMA=4, MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, MMA_N))
            Tensor tensor = make_tensor(tensor_.data(), FLASH_NAMESPACE::convert_layout_acc_rowcol(tensor_.layout()));
            // Do we need both row and column indices, or just column incides?
            const bool use_col_idx_only = !(Has_alibi && !Is_causal) && !Is_local && !Causal_mask && !need_custom_mask;
            const int lane_id = threadIdx.x % 32;
            const int col_idx_offset = col_idx_offset_ + (lane_id % 4) * 2;
            if (use_col_idx_only) {
                #pragma unroll
                for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
                    const int col_idx_base = col_idx_offset + nj * 8;
                    #pragma unroll
                    for (int j = 0; j < size<1, 0>(tensor); ++j) {
                        const int col_idx = col_idx_base + j;
                        #pragma unroll
                        for (int mi = 0; mi < size<0>(tensor); ++mi) {
                            // No causal, no local
                            if constexpr (Has_alibi) {
                                tensor(mi, make_coord(j, nj)) += alibi_slope * col_idx;
                            }
                            if constexpr (!Is_even_MN) {
                                if (col_idx >= max_seqlen_k) { tensor(mi, make_coord(j, nj)) = -INFINITY; }
                            }
                            if (need_custom_mask) {
                                if (col_idx >= attn_mask_seqlen_k) {
                                    tensor(mi, make_coord(j, nj)) = -INFINITY;
                                } else if (custom_mask_is_bool) {
                                    const uint8_t mask_val = bool_mask_base[col_idx * attn_mask_col_stride];
                                    if (!mask_val) {
                                        tensor(mi, make_coord(j, nj)) = -INFINITY;
                                    }
                                } else if (custom_mask_is_additive) {
                                    const float mask_val = float_mask_base[col_idx * attn_mask_col_stride];
                                    tensor(mi, make_coord(j, nj)) += mask_val;
                                }
                            }
                        }
                    }
                }
            } else {
                #pragma unroll
                for (int mi = 0; mi < size<0, 1>(tensor); ++mi) {
                    const int row_idx_base = row_idx_offset + mi * warp_row_stride;
                    #pragma unroll
                    for (int i = 0; i < size<0, 0>(tensor); ++i) {
                        const int row_idx = row_idx_base + i * 8;
                        const int col_idx_limit_left = std::max(0, row_idx + max_seqlen_k - max_seqlen_q - window_size_left);
                        const int col_idx_limit_right = std::min(max_seqlen_k, row_idx + 1 + max_seqlen_k - max_seqlen_q + window_size_right);
                        const uint8_t *bool_row_ptr = custom_mask_is_bool && need_custom_mask
                            ? bool_mask_base + static_cast<int64_t>(row_idx) * attn_mask_row_stride
                            : nullptr;
                        const float *float_row_ptr = !custom_mask_is_bool && need_custom_mask
                            ? float_mask_base + static_cast<int64_t>(row_idx) * attn_mask_row_stride
                            : nullptr;
                        #pragma unroll
                        for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
                            const int col_idx_base = col_idx_offset + nj * 8;
                            #pragma unroll
                            for (int j = 0; j < size<1, 0>(tensor); ++j) {
                                const int col_idx = col_idx_base + j;
                                if constexpr (Has_alibi) {
                                    if constexpr (Is_causal) {
                                        tensor(make_coord(i, mi), make_coord(j, nj)) += alibi_slope * col_idx;
                                    } else {
                                        tensor(make_coord(i, mi), make_coord(j, nj)) -= alibi_slope * abs(row_idx + max_seqlen_k - max_seqlen_q - col_idx);

                                    }
                                }
                                if constexpr (Causal_mask) {
                                    if (col_idx >= col_idx_limit_right) {
                                        tensor(make_coord(i, mi), make_coord(j, nj)) = -INFINITY;
                                    }
                                }
                                if constexpr (Is_local) {
                                    if (col_idx >= col_idx_limit_right || col_idx < col_idx_limit_left) {
                                        tensor(make_coord(i, mi), make_coord(j, nj)) = -INFINITY;
                                    }
                                }
                                if constexpr (!Causal_mask && !Is_local && !Is_even_MN) {
                                    // Causal and Local already handles MN masking
                                    if (col_idx >= max_seqlen_k) {
                                        tensor(make_coord(i, mi), make_coord(j, nj)) = -INFINITY;
                                    }
                                }
                                if (need_custom_mask) {
                                    if (row_idx >= attn_mask_seqlen_q || col_idx >= attn_mask_seqlen_k) {
                                        tensor(make_coord(i, mi), make_coord(j, nj)) = -INFINITY;
                                    } else if (custom_mask_is_bool) {
                                        const uint8_t mask_val = bool_row_ptr[col_idx * attn_mask_col_stride];
                                        if (!mask_val) {
                                            tensor(make_coord(i, mi), make_coord(j, nj)) = -INFINITY;
                                        }
                                    } else if (custom_mask_is_additive) {
                                        const float mask_val = float_row_ptr[col_idx * attn_mask_col_stride];
                                        tensor(make_coord(i, mi), make_coord(j, nj)) += mask_val;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    };

    template <typename Engine, typename Layout>
    __forceinline__ __device__ void apply_custom_mask_only(
        Tensor<Engine, Layout> &tensor_, const int col_idx_offset_, const int row_idx_offset,
        const int warp_row_stride) const {
        if (!has_custom_mask) { return; }
        Tensor tensor = make_tensor(tensor_.data(), FLASH_NAMESPACE::convert_layout_acc_rowcol(tensor_.layout()));
        const int lane_id = threadIdx.x % 32;
        const int col_idx_offset = col_idx_offset_ + (lane_id % 4) * 2;
        #pragma unroll
        for (int mi = 0; mi < size<0, 1>(tensor); ++mi) {
            const int row_idx_base = row_idx_offset + mi * warp_row_stride;
            #pragma unroll
            for (int i = 0; i < size<0, 0>(tensor); ++i) {
                const int row_idx = row_idx_base + i * 8;
                const uint8_t *bool_row_ptr = custom_mask_is_bool
                    ? bool_mask_base + static_cast<int64_t>(row_idx) * attn_mask_row_stride
                    : nullptr;
                const float *float_row_ptr = !custom_mask_is_bool
                    ? float_mask_base + static_cast<int64_t>(row_idx) * attn_mask_row_stride
                    : nullptr;
                #pragma unroll
                for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
                    const int col_idx_base = col_idx_offset + nj * 8;
                    #pragma unroll
                    for (int j = 0; j < size<1, 0>(tensor); ++j) {
                        const int col_idx = col_idx_base + j;
                        if (row_idx >= attn_mask_seqlen_q || col_idx >= attn_mask_seqlen_k) {
                            tensor(make_coord(i, mi), make_coord(j, nj)) = -INFINITY;
                        } else if (custom_mask_is_bool) {
                            const uint8_t mask_val = bool_row_ptr[col_idx * attn_mask_col_stride];
                            if (!mask_val) {
                                tensor(make_coord(i, mi), make_coord(j, nj)) = -INFINITY;
                            }
                        } else if (custom_mask_is_additive) {
                            const float mask_val = float_row_ptr[col_idx * attn_mask_col_stride];
                            tensor(make_coord(i, mi), make_coord(j, nj)) += mask_val;
                        }
                    }
                }
            }
        }
    }

};

} // namespace FLASH_NAMESPACE
