



# 第34章 精密机床加工仓库：GGML 张量计算框架的核心架构与执行引擎深度解析

## 学习目标

- 掌握 GGML 张量（`ggml_tensor`）的核心结构定义：类型、形状、步长、操作类型、源张量、视图关系
- 掌握张量属性查询函数族：`is_scalar`、`is_vector`、`is_matrix`、`is_3d`、`is_contiguous`、`is_transposed`、`is_permuted` 等判别逻辑
- 掌握 GGML 算子的统一构建模式：`dup_tensor`/`view_tensor` → 设置 `op_params` → 链接 `src`
- 掌握元素级一元算子（abs/neg/sin/cos/tanh）、二元算术（add/sub/mul/div）、归约（sum/mean/argmax）的实现机制
- 掌握 in-place 版本的实现原理：通过 `ggml_view_tensor` 共享内存并标记为 `GGML_OP_*_INPLACE`
- 掌握 GLU 门控机制的全部变体：ReGLU、GEGLU、SwiGLU 及其 split/oai/swapped 版本
- 掌握卷积、池化、填充、插值等 CV 相关算子的构建逻辑
- 掌握 FlashAttention、RoPE 等注意力机制相关核心算子
- 掌握图形化反向传播：`build_forward_expand`、`build_backward_expand` 的梯度图构建
- 掌握内存管理基础设施：`aligned_malloc`、动态分配器、图分配器、缓冲区抽象
- 掌握 GGUF 文件格式的读写：元数据 KV、张量信息、序列化与反序列化

## 34.1 生活类比：精密机床加工仓库

想象 GGML 是一个**精密机床加工仓库**，负责将原始材料（张量数据）加工成最终的复杂工件（计算结果）：

**仓库与图纸（`ggml_cgraph`）**：每张图纸标注了加工步骤（op）、源材料位置（src）、目标形状（ne）。图纸不是立即执行的工单，而是设计蓝图——延迟计算的核心优势。

**两台关键机床类型**：
- **VIP 加急机床（In-place）**：直接改造原材料，无需额外空间 → `ggml_view_tensor`
- **标准复制机床**：将材料完整加工到新产品 → `ggml_dup_tensor`

**精密量具（属性检查）**：
- 卡尺（`ggml_is_contiguous`）：检查材料是否连续排列
- 角度仪（`ggml_is_transposed`）：检查是否旋转了 90°
- 三维扫描仪（`ggml_n_dims`）：测量有多少个维度

**模具家族（激活函数）**：不同形状的模具可压制出相同原料的不同形状。GELU 有三种模具：精确版（Slow）、近似版（Fast）、ERF 版。SwiGLU 是双通道模具：一部分决定开关（sigmoid）、一部分是通道。

**专家分配系统（MoE）**：`ggml_mul_mat_id` 根据客户需求（ids）把工单分配给不同车间，每个车间（Expert）有自己的工艺，最后合并结果。

**质量检测（反向传播）**：`ggml_build_backward_expand` 反向检查工件，计算每个工序的误差，从最终产品逐步回溯到原材料。

**仓储管理系统**：
- `ggml_tallocr`：简单的线性货架，适合预先知道所有商品大小的场景
- `ggml_dyn_tallocr`：动态段落式货架，支持多个货架段，按需借用
- `ggml_gallocr`：智能管理员，分析整个仓库的最佳布局，支持跨架利用

**集装箱格式（GGUF）**：国际通用集装箱：头部有规格说明 + 货物清单 + 实际商品。任何港口（CPU/GPU）都能识别和处理这种集装箱。

**总控中心（调度器）**：`ggml_backend_sched` 多车间协调中心，将工单分配到不同车间，平衡负载、优化传输、支持并行生产。

## 34.2 源码地图

```
ggml/include/ggml.h
├── 核心数据结构
│   ├── ggml_init_params        # 上下文初始化参数
│   ├── ggml_tensor             # 张量：type, ne[], nb[], op, src[], view_src, data
│   ├── ggml_cgraph             # 计算图：nodes[], leafs[], grads[]
│   └── ggml_type_traits        # 类型特征：blck_size, type_size, to_float, from_float
│
├── 张量生命周期
│   ├── ggml_init                   # 创建上下文
│   ├── ggml_reset                  # 重置上下文
│   ├── ggml_free                   # 释放上下文
│   ├── ggml_new_tensor_*d          # 创建1/2/3/4维张量
│   ├── ggml_dup_tensor            # 复制张量结构（不复制数据）
│   ├── ggml_view_tensor           # 创建视图（共享数据）
│   ├── ggml_get_first_tensor      # 遍历上下文张量
│   ├── ggml_set_input/output/param/loss  # 设置张量标志
│   └── ggml_set_name/format_name  # 命名与格式化
│
├── 张量属性查询
│   ├── ggml_nelements / ggml_nrows / ggml_nbytes / ggml_nbytes_pad
│   ├── ggml_blck_size / ggml_type_size / ggml_row_size / ggml_element_size
│   ├── ggml_is_quantized / ggml_type_name
│   ├── ggml_is_scalar / is_vector / is_matrix / is_3d / n_dims
│   ├── ggml_is_contiguous / is_contiguous_0/1/2 / is_contiguous_rows
│   ├── ggml_is_transposed / is_permuted / is_view / is_empty
│   ├── ggml_can_repeat / are_same_shape / are_same_stride
│   ├── ggml_op_name / ggml_op_symbol / ggml_op_desc / ggml_unary_op_name / ggml_glu_op_name
│   └── ggml_ftype_to_ggml_type
│
├── 基础算术与一元运算
│   ├── ggml_add / ggml_sub / ggml_mul / ggml_div 及 inplace 变体
│   ├── ggml_sqr / ggml_sqrt / ggml_log / ggml_exp / ggml_expm1
│   ├── ggml_sin / ggml_cos / ggml_tanh
│   ├── ggml_abs / ggml_sgn / ggml_neg / ggml_step
│   ├── ggml_floor / ggml_ceil / ggml_round / ggml_trunc
│   ├── ggml_scale / ggml_scale_bias（s*x + b）
│   └── ggml_repeat / ggml_concat / ggml_argsort / ggml_top_k / ggml_arange
│
├── 激活函数家族
│   ├── ggml_relu / ggml_leaky_relu / ggml_silu / ggml_sigmoid
│   ├── ggml_gelu / ggml_gelu_erf / ggml_gelu_quick 三个 GELU 变体
│   ├── ggml_hardswish / ggml_hardsigmoid
│   ├── ggml_elu / ggml_softplus / ggml_xielu
│   └── ggml_silu_back （反向传播）
│
├── GLU 门控线性单元家族
│   ├── ggml_glu / ggml_glu_split（通用接口）
│   ├── ggml_reglu / reglu_split / reglu_swapped
│   ├── ggml_geglu / geglu_split / geglu_swapped / geglu_erf
│   ├── ggml_swiglu / swiglu_split / swiglu_swapped
│   └── ggml_swiglu_oai（带 alpha/limit 参数）
│
├── 归一化与归约
│   ├── ggml_norm / ggml_rms_norm / ggml_l2_norm（含 inplace 变体）
│   ├── ggml_group_norm（Stable Diffusion 用）
│   ├── ggml_rms_norm_back
│   ├── ggml_sum / ggml_sum_rows / ggml_mean / ggml_argmax / ggml_count_equal
│   └── ggml_cumsum（累加和）
│
├── 取行与矩阵运算
│   ├── ggml_get_rows / ggml_get_rows_back / ggml_set_rows
│   ├── ggml_mul_mat / ggml_mul_mat_set_prec（矩阵乘法）
│   ├── ggml_mul_mat_id（稀疏专家路由）
│   ├── ggml_out_prod（外积）
│   └── ggml_solve_tri（三角方程求解）
│
├── 卷积与图像相关
│   ├── ggml_conv_1d / conv_1d_ph / conv_1d_dw / conv_1d_dw_ph
│   ├── ggml_conv_2d / conv_2d_dw / conv_2d_direct / conv_2d_dw_direct
│   ├── ggml_conv_transpose_1d / conv_transpose_2d_p0
│   ├── ggml_conv_3d / conv_3d_direct
│   ├── ggml_im2col / ggml_im2col_back / im2col_3d
│   ├── ggml_pool_1d / ggml_pool_2d / pool_2d_back
│   ├── ggml_upscale / upscale_ext / interpolate（上采样与插值）
│   ├── ggml_pad / pad_ext / pad_circular / pad_ext_circular / pad_reflect_1d
│   ├── ggml_roll / ggml_tri（滚动与三角矩阵）
│   └── ggml_win_part / ggml_win_unpart（窗口划分）
│
├── 注意力与位置编码
│   ├── ggml_soft_max / soft_max_inplace
│   ├── ggml_soft_max_ext / soft_max_ext_inplace（带 mask、scale、ALiBi）
│   ├── ggml_soft_max_add_sinks（Flash Attention sink 支持）
│   ├── ggml_soft_max_ext_back（反向传播）
│   ├── ggml_flash_attn_ext / flash_attn_ext_set_prec/get_prec
│   ├── ggml_flash_attn_ext_add_sinks / flash_attn_back
│   ├── ggml_rope / rope_inplace / rope_ext / rope_ext_inplace
│   ├── ggml_rope_multi / rope_multi_inplace
│   ├── ggml_rope_ext_back / rope_multi_back
│   ├── ggml_rope_yarn_corr_dims（计算 YARN 维度修正）
│   ├── ggml_clamp
│   ├── ggml_get_rel_pos / add_rel_pos（相对位置）
│   └── ggml_timestep_embedding（时间步嵌入）
│
├── 状态空间与序列模型
│   ├── ggml_ssm_conv / ssm_scan
│   ├── ggml_rwkv_wkv6 / rwkv_wkv7
│   ├── ggml_gated_linear_attn
│   ├── ggml_gated_delta_net
│   └── ggml_fill / fill_inplace（填充）
│
├── 形状变换与视图
│   ├── ggml_cont / cont_*d（连续化）
│   ├── ggml_reshape / reshape_1d/2d/3d/4d
│   ├── ggml_view_1d/2d/3d/4d
│   ├── ggml_permute / transpose
│   ├── ggml_set / set_inplace / set_1d/2d、set_1d/2d_inplace
│   ├── ggml_cpy（复制）/ ggml_cast（类型转换）
│   └── ggml_diag / diag_mask_inf/zero（对角矩阵与 mask）
│
├── 反向传播与图构建
│   ├── ggml_build_forward_expand / build_backward_expand / build_forward_select
│   ├── ggml_new_graph / new_graph_custom / graph_dup / graph_cpy
│   ├── ggml_graph_reset / graph_clear
│   ├── ggml_graph_size / graph_node / graph_nodes / graph_n_nodes
│   ├── ggml_graph_get_tensor / get_grad / get_grad_acc
│   ├── ggml_graph_print / graph_dump_dot（图可视化）
│   └── ggml_set_zero（清零）
│
├── 自定义算子与优化器
│   ├── ggml_map_custom1/2/3 与 inplace 变体、ggml_custom_4d、ggml_custom_inplace
│   ├── ggml_cross_entropy_loss / cross_entropy_loss_back
│   └── ggml_opt_step_adamw / ggml_opt_step_sgd（优化器步骤）
│
├── 日志与工具
│   ├── ggml_log_get / ggml_log_set
│   ├── ggml_set_abort_callback / ggml_abort
│   ├── ggml_status_to_string
│   ├── ggml_time_init / ggml_time_ms / ggml_time_us / ggml_cycles / ggml_cycles_per_ms
│   ├── ggml_fopen
│   ├── ggml_fp16_to_fp32 / fp32_to_fp16
│   ├── ggml_bf16_to_fp32 / fp32_to_bf16 及行版本
│   └── ggml_threadpool_params_default/init/match
│
└── 量化与类型转换
    ├── ggml_quantize_init / quantize_free / quantize_requires_imatrix / quantize_chunk
    └── ggml_version / ggml_commit

ggml/include/ggml-cpu.h
└── CPU 特定 API
    ├── ggml_cpu_has_* 系列（CPU 特性检测）
    ├── ggml_threadpool_new / free / get_n_threads / pause / resume
    └── ggml_graph_plan / graph_compute（CPU 计算图执行）

ggml/include/gguf.h
├── GGUF 上下文
│   ├── gguf_init_empty / gguf_init_from_file / gguf_init_from_file_ptr / gguf_free
│   └── gguf_get_version / get_alignment / get_data_offset
│
├── KV 读取
│   ├── gguf_get_n_kv / gguf_find_key / gguf_get_key
│   ├── gguf_get_kv_type / get_arr_type
│   ├── gguf_get_val_u8/i8/u16/i16/u32/i32/f32/u64/i64/f64/bool/str/data
│   └── gguf_get_arr_n / get_arr_data / get_arr_str
│
├── 张量信息读取与设置
│   ├── gguf_get_n_tensors / find_tensor / get_tensor_offset/name/type/size
│   ├── gguf_remove_key
│   ├── gguf_set_val_*（所有类型）
│   ├── gguf_set_arr_data / set_arr_str / set_kv
│   ├── gguf_add_tensor / set_tensor_type / set_tensor_data
│   └── gguf_write_to_file / write_to_file_ptr / get_meta_size / get_meta_data

ggml/include/ggml-backend.h
├── 缓冲与缓冲区类型
│   ├── ggml_backend_buft_*（name/alloc_buffer/get_alignment/get_max_size/get_alloc_size/is_host/get_device）
│   └── ggml_backend_buffer_*（name/free/get_base/get_size/init_tensor/clear 等）
│
├── 设备、后端、注册表
│   ├── ggml_backend_dev_*（name/description/memory/type/init 支持 op 等）
│   ├── ggml_backend_*（guid/name/free/get_device 等）
│   ├── ggml_backend_reg_*（name/dev_count/dev_get/reg_by_name）
│   └── ggml_backend_event_*（new/free/record/synchronize/wait）
│
├── 调度器
│   ├── ggml_backend_sched_new / free / reserve / alloc_graph / graph_compute
│   ├── ggml_backend_sched_reset / synchronize / split_graph
│   ├── ggml_backend_sched_get_n_backends/backends/buffer_type/buffer_size
│   └── ggml_backend_sched_set/get_tensor_backend
│
└── 工具函数
    ├── ggml_backend_tensor_copy / copy_async
    ├── ggml_backend_tensor_set/get/memset 及 async 版本
    └── ggml_backend_compare_graph_backend

ggml/include/ggml-cpp.h
└── C++ RAII 封装的智能指针删除器
    ├── ggml_context_deleter / gguf_context_deleter
    ├── ggml_gallocr_deleter
    ├── ggml_backend_deleter / ggml_backend_buffer_deleter
    ├── ggml_backend_event_deleter / ggml_backend_sched_deleter

ggml/include/ggml-alloc.h
├── 图分配器
│   ├── ggml_tallocr_new / ggml_tallocr_alloc
│   ├── ggml_gallocr_new / gallocr_new_n / gallocr_free
│   ├── ggml_gallocr_reserve / gallocr_reserve_n
│   ├── ggml_gallocr_alloc_graph / gallocr_get_buffer_size
│   └── ggml_gallocr_reserve_n_size
└── 张量上下文缓冲区分配
    ├── ggml_backend_alloc_ctx_tensors_from_buft_size
    ├── ggml_backend_alloc_ctx_tensors_from_buft
    └── ggml_backend_alloc_ctx_tensors

ggml/include/ggml-opt.h
├── 数据集管理（ggml_opt_dataset）
│   ├── ggml_opt_dataset_init / free / ndata / data / labels
│   └── ggml_opt_dataset_shuffle / get_batch / get_batch_host
│
├── 优化器上下文（ggml_opt_context）
│   ├── ggml_opt_init / free / reset / static_graphs / inputs / outputs / labels / loss / pred / ncorrect
│   └── ggml_opt_grad_acc / prepare_alloc / alloc / eval
│
├── 优化器参数与结果
│   ├── ggml_opt_default_params / get_optimizer_params / get_default_optimizer_params
│   └── ggml_opt_result_init/free/reset/ndata/loss/pred/accuracy
│
└── 高层训练 API
    ├── ggml_opt_epoch / epoch_callback / epoch_callback_progress_bar
    ├── ggml_opt_fit（完整训练循环）
    └── ggml_opt_optimizer_name

ggml/src/ggml.c
├── 初始化与内存管理
│   ├── ggml_init / ggml_new_object / ggml_reset
│   ├── ggml_print_object / ggml_print_objects
│   ├── ggml_aligned_malloc / ggml_aligned_free / ggml_malloc / ggml_calloc
│   ├── ggml_fopen / ggml_mbstowcs
│   └── ggml_set_abort_callback / ggml_abort / ggml_print_backtrace / backtrace_symbols
│
├── 日志与回调
│   ├── ggml_log_internal / ggml_log_callback_default / ggml_log_get / ggml_log_set
│   └── ggml_log_internal_v
│
├── 线程池与调试工具
│   ├── ggml_threadpool_params_init / default / match
│   └── ggml_print_backtrace_symbols / unwind_callback
│
├── 时间工具
│   ├── ggml_time_init / time_ms / time_us / cycles / cycles_per_ms / version / commit
│   └── ggml_guid_matches
│
├── 张量属性查询实现
│   ├── ggml_nelements / nrows / nbytes / nbytes_pad / blck_size / type_size / row_size
│   ├── ggml_is_contiguous / is_contiguous_0/1/2 / is_contiguously_allocated / is_contiguous_channels / is_contiguous_rows
│   ├── ggml_is_scalar / is_vector / is_matrix / is_3d / n_dims / is_transposed / is_permuted / is_empty
│   ├── ggml_is_quantized / type_name / op_name / op_symbol / unary_op_name / glu_op_name
│   └── ggml_can_repeat / are_same_shape / are_same_stride / element_size
│
├── 张量创建与视图
│   ├── ggml_new_tensor_impl / new_tensor / new_tensor_1d-4d / new_buffer
│   ├── ggml_dup_tensor / view_tensor / get_first_tensor / get_next_tensor / get_tensor
│   └── ggml_get_unary_op / get_glu_op / get_data / get_data_f32 / get_name
│
├── 核心数学运算实现
│   ├── ggml_unary_impl / unary / unary_inplace
│   ├── ggml_sin/cos/tanh/abs/sgn/neg/step/relu/leaky_relu/silu/gelu 等激活
│   ├── ggml_norm_impl / norm / norm_inplace
│   ├── ggml_rms_norm_impl / rms_norm / rms_norm_inplace / rms_norm_back
│   ├── ggml_l2_norm_impl / l2_norm / l2_norm_inplace
│   ├── ggml_group_norm_impl / group_norm 变体
│   ├── ggml_scale_impl / scale 系列
│   ├── ggml_sum / sum_rows / mean / argmax / count_equal / cumsum
│   ├── ggml_repeat / repeat_4d / repeat_back / concat
│   └── ggml_mul_mat / mul_mat_id / out_prod
│
├── 门控线性单元实现
│   ├── ggml_glu_impl（核心实现）
│   ├── ggml_glu / glu_split（通用接口）
│   ├── ggml_reglu/geglu/swiglu 及其 split/swapped 版本
│   └── ggml_swiglu_oai
│
├── 形状变换与复制
│   ├── ggml_cont_impl / cont / cont_*d
│   ├── ggml_reshape / reshape_1d-4d
│   ├── ggml_view_impl / view_1d-4d
│   ├── ggml_permute / transpose
│   ├── ggml_set_impl / set 变体 / set_rows / get_rows / get_rows_back
│   ├── ggml_cpy_impl / cpy / cast
│   └── ggml_diag / diag_mask_inf_impl / diag_mask_* 系列
│
├── 卷积与图像操作
│   ├── ggml_conv_1d / conv_1d_ph / conv_1d_dw / conv_1d_dw_ph / conv_transpose_1d
│   ├── ggml_conv_2d / conv_2d_dw / conv_2d_direct / conv_2d_dw_direct / conv_2d_sk_p0 / conv_2d_s1_ph
│   ├── ggml_conv_3d / conv_3d_direct
│   ├── ggml_im2col / im2col_back / im2col_3d
│   ├── ggml_pool_1d / pool_2d / pool_2d_back
│   ├── ggml_upscale / upscale_ext / interpolate
│   ├── ggml_pad 系列 / roll / tri / fill 系列
│   └── ggml_win_part / win_unpart
│
├── 注意力与位置编码
│   ├── ggml_soft_max_impl / soft_max / soft_max_inplace
│   ├── ggml_soft_max_ext / soft_max_ext_inplace / soft_max_add_sinks
│   ├── ggml_soft_max_ext_back_impl / soft_max_ext_back
│   ├── ggml_flash_attn_ext / flash_attn_back / flash_attn_ext_*
│   ├── ggml_rope_impl / rope 系列 / rope_multi 系列 / rope_ext 系列
│   ├── ggml_rope_yarn_corr_dims / corr_dim
│   ├── ggml_rope_*back（反向）
│   ├── ggml_clamp / get_rel_pos / add_rel_pos
│   └── ggml_timestep_embedding
│
├── SSM 与 RWKV
│   ├── ggml_ssm_conv / ssm_scan
│   ├── ggml_rwkv_wkv6 / rwkv_wkv7
│   ├── ggml_gated_linear_attn / gated_delta_net
│   └── ggml_solve_tri
│
├── 反向传播引擎
│   ├── ggml_visit_parents_graph（图遍历）
│   ├── ggml_build_forward_impl / build_forward_expand / build_forward_select
│   ├── ggml_compute_backward（各种 op 的反向传播逻辑）
│   ├── ggml_add_or_set / add1_or_set / sub_or_set / acc_or_set
│   ├── ggml_graph_nbytes / graph_overhead / graph_overhead_custom
│   └── ggml_new_graph_custom / graph_dup / graph_cpy / graph_view / graph_clear/reset/size/node/n_nodes
│
├── 图打印与调试
│   ├── ggml_graph_get_tensor / get_grad / get_grad_acc / print / dump_dot / find / get_parent
│   ├── ggml_graph_dump_dot_node_edge / dump_dot_leaf_edge
│   ├── ggml_can_fuse_subgraph_ext / node_list_find_tensor
│   └── ggml_set_zero
│
└── 量化接口
    ├── ggml_quantize_init / quantize_free / quantize_requires_imatrix / quantize_chunk
    └── ggml_fp16/bf16 转换函数（*_row）

ggml/src/ggml.cpp
└── terminate_handler 安装
    └── ggml_uncaught_exception / __main__（静态初始化）

ggml/src/ggml-alloc.c
├── 图分配器实现
│   ├── ggml_op_can_inplace
│   ├── ggml_tallocr_new/alloc
│   ├── ggml_dyn_tallocr_*（动态分配器）
│   ├── vbuffer_*（虚拟缓冲区）
│   ├── ggml_gallocr_*（核心图分配器实现）
│   └── alloc_tensor_range / ggml_backend_alloc_ctx_tensors_*
│
└── 内部工具
    ├── aligned_offset / buffer_address_less
    ├── add_allocated_tensor / remove_allocated_tensor
    ├── get_node_buffer_id / free_buffers

ggml/src/ggml-backend.cpp
├── 后端基础设施实现
│   ├── ggml_backend_buffer_init / buffer_*
│   ├── ggml_backend_dev_*
│   ├── ggml_backend_tensor_copy/copy_async/set/get/memset 系列
│   ├── ggml_backend_graph_plan_* / graph_compute*
│   ├── ggml_backend_event_*
│   ├── ggml_backend_multi_buffer_*
│   └── ggml_backend_graph_optimize
│
├── 调度器核心
│   ├── ggml_backend_sched_backend_id / backend_id_from_cur / backend_from_buffer / buffer_supported / set_if_supported
│   ├── ggml_backend_sched_split_graph
│   ├── ggml_backend_sched_reserve / alloc_graph / graph_compute 系列
│   ├── ggml_backend_sched_reset / synchronize / get_* 系列
│   └── ggml_backend_sched_print_assignments
│
└── 图复制与验证
    ├── ggml_backend_view_init / tensor_alloc
    ├── graph_copy_dup_tensor / graph_copy_init_tensor
    ├── ggml_backend_graph_copy / graph_copy_free
    └── ggml_backend_compare_graph_backend

ggml/src/ggml-backend-reg.cpp
├── 后端注册表管理
│   ├── ggml_backend_registry（构造/析构/register_backend/register_device/load_backend/unload_backend）
│   ├── get_reg / path_str / path_str / striequals
│   ├── ggml_backend_init_by_name / by_type / init_best
│   ├── ggml_backend_register / device_register / load / unload
│   ├── get_executable_path / backend_filename_prefix/extension
│   └── ggml_backend_load_best / load_all / load_all_from_path
│
└── 动态加载封装
    └── （path 字符串和文件探测）

ggml/src/ggml-backend-dl.cpp
└── 动态链接库跨平台支持
    ├── dl_load_library / dl_get_sym / dl_error
    └── （Windows/POSIX 条件编译）

ggml/src/ggml-opt.cpp
├── 训练数据集实现
│   ├── ggml_opt_dataset_init / free / shuffle
│   └── ggml_opt_dataset_get_batch / get_batch_host
│
├── 优化器上下文
│   ├── ggml_opt_build（构建优化图）
│   ├── ggml_opt_init / free
│   ├── map_tensor / dup_graph（图复制工具）
│   └── ggml_opt_reset
│
├── 优化器属性
│   └── ggml_opt_context_optimizer_type / optimizer_name
│
├── 结果追踪
│   ├── ggml_opt_result_init / free / reset / ndata / loss / pred / accuracy
│   └── ggml_opt_prepare_alloc / alloc / eval
│
└── 高层 API
    ├── ggml_opt_epoch / epoch_callback_progress_bar
    └── ggml_opt_fit

ggml/src/gguf.cpp
├── 文件读取
│   ├── gguf_reader（完整读取器类）
│   ├── gguf_init_from_file_ptr / gguf_init_from_file
│   └── gguf_read_emplace_helper
│
├── 文件写入
│   ├── gguf_writer_base（抽象基类）
│   ├── gguf_writer_buf（内存缓冲区写入）
│   ├── gguf_writer_file（文件写入）
│   ├── gguf_write_out
│   ├── gguf_write_to_file / write_to_file_ptr
│   └── gguf_get_meta_size / get_meta_data
│
├── KV 与张量管理
│   ├── gguf_kv（KV 对象类）
│   ├── gguf_tensor_info
│   ├── gguf_find_key / remove_key / set_val_* 系列
│   ├── gguf_set_arr_data / set_arr_str / set_kv
│   ├── gguf_add_tensor / set_tensor_type / set_tensor_data
│   └── gguf_find_tensor / get_tensor_* 系列
│
└── 元数据查询与版本
    ├── gguf_type_name / gguf_type_size
    ├── gguf_get_version / get_alignment / get_data_offset / get_n_kv
    └── gguf_get_kv_type / get_arr_type / get_val_* 系列

ggml/src/ggml-threading.cpp
└── 全局临界区
    ├── ggml_critical_section_start
    └── ggml_critical_section_end

```

## 34.3 仓库图纸与乐高积木：ggml_tensor 核心结构与计算图拓扑

GGML 的核心设计哲学是**延迟执行**：你并不直接计算张量，而是通过 API 构建一张「计算图蓝图」（`ggml_cgraph`），最后再由后端统一执行。这种设计将「定义计算」与「执行计算」解耦，使得跨后端调度、图优化、内存规划成为可能。

### 34.3.1 ggml_tensor 核心结构定义：乐高积木的属性与连接点

`ggml_tensor` 是 GGML 中的「乐高积木」，每个积木不仅知道自己的形状，还记录了自己是如何由其他积木拼装而成的。

**源码路径**：`ggml/include/ggml.h`（约第 180-250 行）

```c
// 类型即图纸，先理解数据结构
struct ggml_tensor {
    // 1. 基础属性
    enum ggml_type type;        // 数据类型：F32, F16, Q4_0, Q8_0 等
    int64_t ne[GGML_MAX_DIMS];  // 形状：[ne0, ne1, ne2, ne3]，ne0 是最内层维度
    size_t  nb[GGML_MAX_DIMS];  // 步长：每维跨度（字节），nb[0] 通常等于 type_size

    // 2. 计算图拓扑
    enum ggml_op op;            // 操作类型：ADD, MUL_MAT, ROPE, SOFT_MAX 等
    struct ggml_tensor * src[GGML_MAX_SRC]; // 源张量指针数组（最多 10 个输入）
    struct ggml_tensor * view_src;          // 视图源张量（非 NULL 表示这是个 view）
    size_t view_offs;           // 相对 view_src 的字节偏移

    // 3. 数据与标志
    void * data;                // 数据指针（由后端分配，或外部提供）
    struct ggml_backend_buffer * buffer;    // 所属后端缓冲区
    uint32_t flags;             // 标志位：INPUT, OUTPUT, PARAM, LOSS, COMPUTE 等

    // 4. 算子参数存储（紧凑数组，避免额外分配）
    int8_t op_params[GGML_MAX_OP_PARAMS];   // 存放算子特有参数（如 RoPE 的频率、norm 的 eps）
    char name[GGML_MAX_NAME];   // 调试用名称
};
```

**关键设计点**：
- **步长 `nb` 而非 strides**：以字节为单位，天然支持量化类型（如 Q4_0 每 32 个元素占 18 字节）。
- **`view_src` 实现零拷贝视图**：`reshape`、`permute`、`transpose` 等形状变换操作不复制数据，只是创建新张量指向同一 `data`，并修改 `ne`/`nb`/`view_offs`。
- **`op_params` 内联存储**：避免为每个算子单独分配参数结构体，所有参数压扁在 64 字节数组中，通过 `ggml_set_op_params_i32/f32` 序列化读写。

这段代码定义了 GGML 核心数据结构 `ggml_tensor`，它不仅包含张量的数据类型、形状、步长等基础属性，还内嵌了计算图拓扑信息（操作类型、源张量、视图关系）和算子参数存储。这种设计使得张量本身就是计算图的节点，天然支持延迟执行和自动微分。

### 34.3.2 ggml_cgraph 计算图结构定义：蓝图的拓扑骨架

**源码路径**：`ggml/include/ggml.h`（约第 250-300 行）

```c
struct ggml_cgraph {
    size_t size;                    // 节点数组容量
    int n_nodes;                    // 当前计算节点数
    int n_leafs;                    // 叶子节点数（输入/权重）
    struct ggml_tensor ** nodes;    // 计算节点数组（拓扑序）
    struct ggml_tensor ** leafs;    // 叶子节点数组
    struct ggml_tensor ** grads;    // 梯度张量（反向传播用）
    struct ggml_tensor ** grad_accs;// 梯度累加器（大 batch 累积用）
    int32_t * use_counts;           // 引用计数：每个节点被多少后续节点依赖
    struct ggml_hash_set visited_hash_set; // 访问哈希表（去重用）
    enum ggml_cgraph_eval_order order;     // 执行顺序策略
};
```

### 34.3.3 计算图构建流程图

```mermaid
graph TD
    A[输出张量] --> B[ggml_build_forward_expand]
    B --> C[ggml_visit_parents_graph DFS 回溯]
    C --> D[visited_hash_set 去重]
    D --> E[use_counts 引用计数]
    E --> F[nodes[] 拓扑序数组]
    F --> G[ggml_build_backward_expand]
    G --> H[grad_accs 梯度累加器]
    H --> I[反向传播子图]
```

### 34.3.4 张量生命周期管理 API：从原材料到成品的工序单

**源码路径**：`ggml/include/ggml.h`（第 680-760 行）、`ggml/src/ggml.c`（`ggml_init` 等实现）

```c
// 创建上下文：分配元数据池（约 10MB 默认），不分配张量数据内存
struct ggml_context * ggml_init(struct ggml_init_params params);

// 创建张量（仅分配元数据，数据由后端稍后分配）
struct ggml_tensor * ggml_new_tensor_1d(struct ggml_context * ctx, enum ggml_type type, int64_t ne0);
struct ggml_tensor * ggml_new_tensor_2d(struct ggml_context * ctx, enum ggml_type type, int64_t ne0, int64_t ne1);
struct ggml_tensor * ggml_new_tensor_3d(struct ggml_context * ctx, enum ggml_type type, int64_t ne0, int64_t ne1, int64_t ne2);
struct ggml_tensor * ggml_new_tensor_4d(struct ggml_context * ctx, enum ggml_type type, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3);

// 核心构建模板：dup_tensor/view_tensor → 设置 op_params → 链接 src
struct ggml_tensor * ggml_dup_tensor(const struct ggml_context * ctx, const struct ggml_tensor * src);  // 复制元数据，独立内存
struct ggml_tensor * ggml_view_tensor(struct ggml_context * ctx, struct ggml_tensor * src);            // 共享数据，零拷贝

// 张量角色标记（影响图构建与内存规划）
void ggml_set_input(struct ggml_tensor * tensor);   // 图输入
void ggml_set_output(struct ggml_tensor * tensor);  // 图输出
void ggml_set_param(struct ggml_tensor * tensor);   // 可训练参数
void ggml_set_loss(struct ggml_tensor * tensor);    // 损失节点
```

这一段代码定义了 GGML 的核心数据结构与张量生命周期管理 API。`ggml_context` 是内存池容器，`ggml_tensor` 是计算图节点，`ggml_cgraph` 是拓扑排序后的执行计划。`dup_tensor` 用于需要独立内存的场景（如 `add` 结果），`view_tensor` 用于零拷贝形状变换与 in-place 操作基础。

## 34.4 量具与检测仪：张量属性查询与诊断 —— 张量形状检查大全

GGML 提供了一套完整的**张量体检工具**，后端在执行前会通过这些函数判断张量布局，选择最优内核（如连续内存走 GEMM，非连续走逐元素内核）。

### 34.4.1 形状判定函数：尺寸量具

**源码路径**：`ggml/include/ggml.h`（第 647-651 行）、`ggml/src/ggml.c`（第 1380-1420 行）

```c
// 逐段检测维度是否符合 [1,1,1,1] → [1,1,n,m] → [1,n,m,x] 等
bool ggml_is_scalar(const struct ggml_tensor * tensor);  // 所有维度为 1
bool ggml_is_vector(const struct ggml_tensor * tensor);  // ne[1..3] == 1
bool ggml_is_matrix(const struct ggml_tensor * tensor);  // ne[2..3] == 1
bool ggml_is_3d(const struct ggml_tensor * tensor);      // ne[3] == 1
int  ggml_n_dims(const struct ggml_tensor * tensor);     // 实际维度数（去除尾随的 1）
```

### 34.4.2 内存结构判定 —— 连续性与转置检测：精密卡尺与角度仪

**源码路径**：`ggml/include/ggml.h`（第 654-666 行）、`ggml/src/ggml.c`（第 1474-1560 行）

```c
// 标准连续：nb[i] == nb[i-1] * ne[i-1]（行主序）
bool ggml_is_contiguous(const struct ggml_tensor * tensor);
bool ggml_is_contiguous_0(const struct ggml_tensor * tensor); // 同 contiguous
bool ggml_is_contiguous_1(const struct ggml_tensor * tensor); // 从第 1 维起连续
bool ggml_is_contiguous_2(const struct ggml_tensor * tensor); // 从第 2 维起连续
bool ggml_is_contiguous_rows(const struct ggml_tensor * tensor); // 第 0 维连续或单块

// 转置检测：nb[0] > nb[1] 说明前两维交换过
bool ggml_is_transposed(const struct ggml_tensor * tensor);

// 任意置换检测：步长不再单调递增
bool ggml_is_permuted(const struct ggml_tensor * tensor);

// 其他实用判定
bool ggml_is_empty(const struct ggml_tensor * tensor);      // 任一维度为 0
bool ggml_is_view(const struct ggml_tensor * tensor);       // view_src != NULL
bool ggml_can_repeat(const struct ggml_tensor * t0, const struct ggml_tensor * t1); // 广播兼容
bool ggml_are_same_shape(const struct ggml_tensor * t0, const struct ggml_tensor * t1);
bool ggml_are_same_stride(const struct ggml_tensor * t0, const struct ggml_tensor * t1);
```

**连续性判定的核心逻辑**（`ggml/src/ggml.c:1474`）：

```c
// ### ggml_is_contiguous_n: 连续性判定核心实现
// 从指定维度开始检查步长是否符合连续内存布局
bool ggml_is_contiguous_n(const struct ggml_tensor * tensor, int start_dim) {
    for (int i = start_dim; i < GGML_MAX_DIMS - 1; i++) {
        if (tensor->ne[i] > 1 && tensor->nb[i+1] != tensor->nb[i] * tensor->ne[i]) {
            return false; // 步长不匹配元素大小 × 个数
        }
    }
    return true;
}
```

### 34.4.3 尺寸计算与类型查询：计量单位换算表

**源码路径**：`ggml/include/ggml.h`（第 615-641 行）、`ggml/src/ggml.c` 对应实现

```c
int64_t ggml_nelements(const struct ggml_tensor * tensor);   // 总元素数 = Π ne[i]
size_t  ggml_nbytes(const struct ggml_tensor * tensor);      // 实际占用字节（考虑量化块与步长）
size_t  ggml_nbytes_pad(const struct ggml_tensor * tensor);  // 对齐到 64 字节
int64_t ggml_blck_size(enum ggml_type type);                 // 量化块大小（Q4_0=32, F32=1）
size_t  ggml_type_size(enum ggml_type type);                 // 一个块的字节数（Q4_0=18, F32=4）
size_t  ggml_row_size(enum ggml_type type, int64_t ne);      // 指定列数的一行字节数
const char * ggml_type_name(enum ggml_type type);            // 类型名字符串
const char * ggml_op_name(enum ggml_op op);                  // 算子名字符串
bool ggml_is_quantized(enum ggml_type type);                 // 是否为量化类型
```

这组函数是后端内核选择、内存分配、量化反量化的基础判据。

## 34.5 数学工具箱：基础算术与一元激活函数 —— 张量加工的「数学工具箱」

GGML 的算子构建遵循统一模板：**创建结果张量 → 填充 `op` 与 `op_params` → 挂载 `src`**。这种设计使得新增算子只需关注数学定义，自动获得图构建、反向传播、多后端分发能力。

### 34.5.1 统一实现：ggml_unary_impl 一元算子工厂

**源码路径**：`ggml/src/ggml.c`（第 5886-5901 行）

```c
// ### ggml_unary_impl: 一元算子的统一实现
// 所有激活函数、三角函数等都通过该函数创建
static struct ggml_tensor * ggml_unary_impl(
    struct ggml_context * ctx,
    struct ggml_tensor  * a,
    enum ggml_unary_op   op,
    bool                 inplace
) {
    // ① 分配结果张量：inplace 用 view 共享内存，否则 dup 独立元数据
    struct ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    // ② 设置操作类型与参数
    ggml_set_op_params_i32(result, 0, (int32_t) op); // op_params[0] 存 unary_op 枚举
    result->op     = GGML_OP_UNARY;                   // 统一归类为 UNARY
    result->src[0] = a;                               // 链接源张量

    return result;
}
```

**深度解析**：`ggml_unary_impl` 是所有一元算子的「统一模具」。它通过 `inplace` 参数决定使用「VIP 加急机床」（`view_tensor`，共享内存）还是「标准复制机床」（`dup_tensor`，独立元数据）。`op_params[0]` 存储具体的 `ggml_unary_op` 枚举值（如 `GGML_UNARY_OP_RELU`、`GGML_UNARY_OP_GELU`），后端内核在执行时通过 `switch(result->op_params[0])` 分发到具体数学函数。这种「枚举驱动」设计消除了虚函数调用开销，实现了零开销抽象。

### 34.5.2 二元算术运算：双工位加工台（支持广播与 in-place）

**源码路径**：`ggml/include/ggml.h`（第 900-980 行）、`ggml/src/ggml.c` 对应实现

```c
// 二元算术：逐元素操作，形状必须完全相同（或通过广播）
struct ggml_tensor * ggml_add(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b);
struct ggml_tensor * ggml_add_inplace(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b); // view(a) + b → 覆盖 a
struct ggml_tensor * ggml_sub(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b);
struct ggml_tensor * ggml_mul(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b);
struct ggml_tensor * ggml_div(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b);
```

### 34.5.3 一元逐元素操作完整名单：各类专用模具

**源码路径**：`ggml/include/ggml.h`（第 1120-1340 行）

```c
// 符号与绝对值
struct ggml_tensor * ggml_abs(struct ggml_context * ctx, struct ggml_tensor * a);
struct ggml_tensor * ggml_sgn(struct ggml_context * ctx, struct ggml_tensor * a);
struct ggml_tensor * ggml_neg(struct ggml_context * ctx, struct ggml_tensor * a);

// 幂与开方
struct ggml_tensor * ggml_sqr(struct ggml_context * ctx, struct ggml_tensor * a);
struct ggml_tensor * ggml_sqrt(struct ggml_context * ctx, struct ggml_tensor * a);

// 对数与指数
struct ggml_tensor * ggml_log(struct ggml_context * ctx, struct ggml_tensor * a);
struct ggml_tensor * ggml_exp(struct ggml_context * ctx, struct ggml_tensor * a);
struct ggml_tensor * ggml_expm1(struct ggml_context * ctx, struct ggml_tensor * a); // exp(x) - 1，数值稳定

// 三角函数
struct ggml_tensor * ggml_sin(struct ggml_context * ctx, struct ggml_tensor * a);
struct ggml_tensor * ggml_cos(struct ggml_context * ctx, struct ggml_tensor * a);
struct ggml_tensor * ggml_tanh(struct ggml_context * ctx, struct ggml_tensor * a);

// 阶跃与取整
struct ggml_tensor * ggml_step(struct ggml_context * ctx, struct ggml_tensor * a);   // x > 0 ? 1 : 0
struct ggml_tensor * ggml_floor(struct ggml_context * ctx, struct ggml_tensor * a);
struct ggml_tensor * ggml_ceil(struct ggml_context * ctx, struct ggml_tensor * a);
struct ggml_tensor * ggml_round(struct ggml_context * ctx, struct ggml_tensor * a);
struct ggml_tensor * ggml_trunc(struct ggml_context * ctx, struct ggml_tensor * a); // 向零取整
```

### 34.5.4 激活函数大家族：非线性模具库

**源码路径**：`ggml/include/ggml.h`（第 1180-1340 行）、`ggml/src/ggml.c`（第 2727-2820 行）

```c
// ReLU 族
struct ggml_tensor * ggml_relu(struct ggml_context * ctx, struct ggml_tensor * a);
struct ggml_tensor * ggml_leaky_relu(struct ggml_context * ctx, struct ggml_tensor * a, float negative_slope, bool inplace);

// Sigmoid 族
struct ggml_tensor * ggml_sigmoid(struct ggml_context * ctx, struct ggml_tensor * a);
struct ggml_tensor * ggml_hardsigmoid(struct ggml_context * ctx, struct ggml_tensor * a);

// Swish/SiLU 族
struct ggml_tensor * ggml_silu(struct ggml_context * ctx, struct ggml_tensor * a);    // x * sigmoid(x)
struct ggml_tensor * ggml_hardswish(struct ggml_context * ctx, struct ggml_tensor * a);

// ELU/Softplus
struct ggml_tensor * ggml_elu(struct ggml_context * ctx, struct ggml_tensor * a);
struct ggml_tensor * ggml_softplus(struct ggml_context * ctx, struct ggml_tensor * a); // log(1+exp(x))

// GELU 三版本：精度与速度的权衡
struct ggml_tensor * ggml_gelu(struct ggml_context * ctx, struct ggml_tensor * a);        // 快速近似
struct ggml_tensor * ggml_gelu_erf(struct ggml_context * ctx, struct ggml_tensor * a);    // 精确 ERF 版
struct ggml_tensor * ggml_gelu_quick(struct ggml_context * ctx, struct ggml_tensor * a);  // 极快近似

// 反向传播专用
struct ggml_tensor * ggml_silu_back(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b); // a=x, b=dy
```

**GELU 三版本的工程取舍**：

| 版本 | 实现方式 | 精度 | 速度 | 适用场景 |
|------|----------|------|------|----------|
| `gelu` | `0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))` | 高 | 中 | 默认推荐 |
| `gelu_erf` | `0.5 * x * (1 + erf(x/√2))` | 最高 | 慢 | 精度敏感场景 |
| `gelu_quick` | `x * sigmoid(1.702 * x)` | 中 | 最快 | 移动端/量化推理 |

### 34.5.5 融合缩放与偏置：ggml_scale_impl 熔铸炉

**源码路径**：`ggml/include/ggml.h`（第 1552-1579 行）、`ggml/src/ggml.c`（第 3289-3323 行）

```c
// 融合算子：x = s * a + b，常用于 LayerNorm 后的缩放平移
struct ggml_tensor * ggml_scale(struct ggml_context * ctx, struct ggml_tensor * a, float s);
struct ggml_tensor * ggml_scale_inplace(struct ggml_context * ctx, struct ggml_tensor * a, float s);
struct ggml_tensor * ggml_scale_bias(struct ggml_context * ctx, struct ggml_tensor * a, float s, float b);
struct ggml_tensor * ggml_scale_bias_inplace(struct ggml_context * ctx, struct ggml_tensor * a, float s, float b);
```

**内部实现**：`op_params` 存储 `[s, b]` 两个 float，后端内核融合乘加，减少内存带宽。

## 34.6 计算核心：归约、归一化与矩阵运算 —— 深度学习的「计算核心」

### 34.6.1 归约操作：压缩机

**源码路径**：`ggml/include/ggml.h`（第 1051-1084 行）、`ggml/src/ggml.c`（第 2471-2500 行）

```c
struct ggml_tensor * ggml_sum(struct ggml_context * ctx, struct ggml_tensor * a);       // 全求和 → 标量
struct ggml_tensor * ggml_sum_rows(struct ggml_context * ctx, struct ggml_tensor * a);   // 按行求和 [a,b,c,d]→[1,b,c,d]
struct ggml_tensor * ggml_mean(struct ggml_context * ctx, struct ggml_tensor * a);       // 行均值
struct ggml_tensor * ggml_argmax(struct ggml_context * ctx, struct ggml_tensor * a);     // 行最大值索引
struct ggml_tensor * ggml_count_equal(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b); // 相等计数
struct ggml_tensor * ggml_cumsum(struct ggml_context * ctx, struct ggml_tensor * a);     // 前缀和
```

### 34.6.2 归一化三大剑客：ggml_norm_impl 等精密平衡器

**源码路径**：`ggml/include/ggml.h`（第 1450-1510 行）、`ggml/src/ggml.c`（第 3047-3180 行）

```c
// L2 归一化：x / sqrt(sum(x^2) + eps)
struct ggml_tensor * ggml_norm(struct ggml_context * ctx, struct ggml_tensor * a, float eps);
struct ggml_tensor * ggml_norm_inplace(struct ggml_context * ctx, struct ggml_tensor * a, float eps);

// RMSNorm（LLaMA 首选）：x / sqrt(mean(x^2) + eps)
struct ggml_tensor * ggml_rms_norm(struct ggml_context * ctx, struct ggml_tensor * a, float eps);
struct ggml_tensor * ggml_rms_norm_inplace(struct ggml_context * ctx, struct ggml_tensor * a, float eps);
struct ggml_tensor * ggml_rms_norm_back(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, float eps); // 反向传播

// L2 范数归一化（RWKV v7 用）
struct ggml_tensor * ggml_l2_norm(struct ggml_context * ctx, struct ggml_tensor * a, float eps);

// GroupNorm（Stable Diffusion）
struct ggml_tensor * ggml_group_norm(struct ggml_context * ctx, struct ggml_tensor * a, int n_groups, float eps);
```

**RMSNorm vs LayerNorm**：RMSNorm 去除了均值减法，仅保留均方根缩放，计算量减半且无需减均值的内存访问，成为 LLaMA 等模型的标配。

### 34.6.3 矩阵运算核心：ggml_mul_mat 等重型机床

**源码路径**：`ggml/include/ggml.h`（第 1517-1546 行）、`ggml/src/ggml.c`（第 3199-3283 行）

```c
// 标准矩阵乘法：A[m,k] @ B[k,n] → C[m,n]，支持广播
struct ggml_tensor * ggml_mul_mat(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b);
void ggml_mul_mat_set_prec(struct ggml_tensor * a, enum ggml_prec prec); // 设置精度（如 phi-2 需 F32 累加）

// MoE 专家路由：根据 ids 将输入路由到对应专家矩阵
struct ggml_tensor * ggml_mul_mat_id(struct ggml_context * ctx, struct ggml_tensor * as, struct ggml_tensor * b, struct ggml_tensor * ids);

// 外积：向量外积 A[m,1] @ B[1,n] → C[m,n]
struct ggml_tensor * ggml_out_prod(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b);

// 三角方程求解（RWKV 等用）
struct ggml_tensor * ggml_solve_tri(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, bool left, bool lower, bool uni);
```

**`ggml_mul_mat_id` 稀疏专家路由原理**：
- `as` 形状 `[experts, k, n]`，每个专家一组权重
- `ids` 形状 `[batch, n_tokens]`，指定每个 token 走哪个专家
- 后端内核根据 `ids` 分散读取 `as` 对应切片，完成 `b @ as[expert]^T`

### 34.6.4 门控线性单元家族：ggml_glu_impl 核心实现

**源码路径**：`ggml/include/ggml.h`（第 1351-1447 行）、`ggml/src/ggml.c`（第 2897-2990 行）

```c
// 通用 GLU 接口：单输入版本（沿通道切半）与双输入版本
struct ggml_tensor * ggml_glu(struct ggml_context * ctx, struct ggml_tensor * a, enum ggml_glu_op op, bool swapped);
struct ggml_tensor * ggml_glu_split(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, enum ggml_glu_op op);

// 三大主流变体
struct ggml_tensor * ggml_reglu(struct ggml_context * ctx, struct ggml_tensor * a);  // gate=ReLU(x), value=x
struct ggml_tensor * ggml_geglu(struct ggml_context * ctx, struct ggml_tensor * a);  // gate=GELU(x), value=x
struct ggml_tensor * ggml_swiglu(struct ggml_context * ctx, struct ggml_tensor * a); // gate=SiLU(x), value=x

// Swapped 版本：交换 gate 与 value 的位置
struct ggml_tensor * ggml_reglu_swapped(struct ggml_context * ctx, struct ggml_tensor * a);
// ... geglu_swapped, swiglu_swapped ...

// OAI 变体：带 alpha 和 limit 参数的 SwiGLU
struct ggml_tensor * ggml_swiglu_oai(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, float alpha, float limit);
```

**GLU 核心计算公式**：`output = gate_fn(gate) * value`，其中 `gate` 与 `value` 要么来自同一张量沿通道切半，要么由 `split` 版本分别提供。`op_params` 存储 `[glu_op枚举, swapped标志]`。

## 34.7 重塑师：形状变换、视图与数据移动 —— 乐高积木的「重塑师」

### 34.7.1 连续化与重塑：ggml_cont_impl 等整形机

**源码路径**：`ggml/include/ggml.h`（第 1656-1736 行）、`ggml/src/ggml.c`（第 3468-3702 行）

```c
// 使张量内存连续（必要时复制数据）
struct ggml_tensor * ggml_cont(struct ggml_context * ctx, struct ggml_tensor * a);
struct ggml_tensor * ggml_cont_2d(struct ggml_context * ctx, struct ggml_tensor * a, int64_t ne0, int64_t ne1);
struct ggml_tensor * ggml_cont_4d(struct ggml_context * ctx, struct ggml_tensor * a, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3);

// 重塑形状（必须连续且总元素数不变），返回 view
struct ggml_tensor * ggml_reshape(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b); // b 指定新形状
struct ggml_tensor * ggml_reshape_2d(struct ggml_context * ctx, struct ggml_tensor * a, int64_t ne0, int64_t ne1);
struct ggml_tensor * ggml_reshape_4d(struct ggml_context * ctx, struct ggml_tensor * a, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3);
```

### 34.7.2 零拷贝形状手术：ggml_view_impl 等视图变换

**源码路径**：`ggml/include/ggml.h`（第 1739-1780 行）、`ggml/src/ggml.c`（第 3704-3769 行）

```c
// 创建视图：指定偏移和步长，数据共享
struct ggml_tensor * ggml_view_1d(struct ggml_context * ctx, struct ggml_tensor * a, int64_t ne0, size_t offset);
struct ggml_tensor * ggml_view_2d(struct ggml_context * ctx, struct ggml_tensor * a, int64_t ne0, int64_t ne1, size_t nb1, size_t offset);
struct ggml_tensor * ggml_view_3d(struct ggml_context * ctx, struct ggml_tensor * a, int64_t ne0, int64_t ne1, int64_t ne2, size_t nb1, size_t nb2, size_t offset);
struct ggml_tensor * ggml_view_4d(struct ggml_context * ctx, struct ggml_tensor * a, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, size_t nb1, size_t nb2, size_t nb3, size_t offset);

// 维度置换：[a,b,c,d] → [a,d,c,b] 等
struct ggml_tensor * ggml_permute(struct ggml_context * ctx, struct ggml_tensor * a, int axis0, int axis1, int axis2, int axis3);
// 转置：swap dim 0/1（二维矩阵转置的特殊 permute）
struct ggml_tensor * ggml_transpose(struct ggml_context * ctx, struct ggml_tensor * a);
```

### 34.7.3 数据设置与复制：ggml_set_impl 等镶嵌工具

**源码路径**：`ggml/include/ggml.h`（第 1582-1653 行）、`ggml/src/ggml.c`（第 3337-3462 行）

```c
// 将 b 的值嵌入到 a 的指定位置（nb1/nb2/nb3 为步长，offset 字节偏移）
struct ggml_tensor * ggml_set(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, size_t nb1, size_t nb2, size_t nb3, size_t offset);
struct ggml_tensor * ggml_set_inplace(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, size_t nb1, size_t nb2, size_t nb3, size_t offset);
struct ggml_tensor * ggml_set_1d(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, size_t offset);
struct ggml_tensor * ggml_set_2d(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, size_t nb1, size_t offset);

// 复制与类型转换
struct ggml_tensor * ggml_cpy(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b); // a → b
struct ggml_tensor * ggml_cast(struct ggml_context * ctx, struct ggml_tensor * a, enum ggml_type type);  // 类型转换

// 特殊视图操作
struct ggml_tensor * ggml_diag(struct ggml_context * ctx, struct ggml_tensor * a);           // 向量→对角矩阵
struct ggml_tensor * ggml_diag_mask_inf(struct ggml_context * ctx, struct ggml_tensor * a, int n_past); // 上三角置 -INF
struct ggml_tensor * ggml_get_rows(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b); // 行收集
struct ggml_tensor * ggml_set_rows(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, struct ggml_tensor * c); // 行散射
```

**view 与 dup 的工程权衡**：

| 操作 | 内存 | 速度 | 适用场景 |
|------|------|------|----------|
| `view` | 零拷贝、共享数据 | 极快 | reshape、permute、transpose、in-place 算子基础 |
| `dup` | 复制元数据+分配新内存 | 较慢 | 需要独立副本、非连续转连续、图分支汇合点 |

## 34.8 CV任务的算子仓库：卷积、池化与图像处理

### 34.8.1 卷积运算家族：ggml_conv_1d 等滑动窗口机床

**源码路径**：`ggml/include/ggml.h`（第 1976-2170 行）、`ggml/src/ggml.c`（第 4467-4708 行）

```c
// 1D 卷积及其深度可分离版本
struct ggml_tensor * ggml_conv_1d(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, int s0, int p0, int d0);
struct ggml_tensor * ggml_conv_1d_ph(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, int s, int d); // padding = kernel/2
struct ggml_tensor * ggml_conv_1d_dw(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, int s0, int p0, int d0); // depthwise

// 2D 卷积
struct ggml_tensor * ggml_conv_2d(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, int s0, int s1, int p0, int p1, int d0, int d1);
struct ggml_tensor * ggml_conv_2d_dw(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, int s0, int s1, int p0, int p1, int d0, int d1); // depthwise
struct ggml_tensor * ggml_conv_2d_direct(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, int s0, int s1, int p0, int p1, int d0, int d1); // 直接实现

// 3D 卷积（视频/体素）
struct ggml_tensor * ggml_conv_3d(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, int64_t IC, int s0, int s1, int s2, int p0, int p1, int p2, int d0, int d1, int d2);

// 转置卷积（反卷积，用于上采样）
struct ggml_tensor * ggml_conv_transpose_1d(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, int s0, int p0, int d0);
struct ggml_tensor * ggml_conv_transpose_2d_p0(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, int stride);
```

### 34.8.2 卷积的矩阵化关键：ggml_im2col 展开台

**源码路径**：`ggml/include/ggml.h`（第 2119-2132 行）、`ggml/src/ggml.c`（第 4400-4442 行）

```c
// 将卷积窗口展开为矩阵，配合 mul_mat 实现高效卷积
struct ggml_tensor * ggml_im2col(
    struct ggml_context * ctx,
    struct ggml_tensor  * a,  // 卷积核
    struct ggml_tensor  * b,  // 输入数据
    int s0, int s1,           // 步幅
    int p0, int p1,           // 填充
    int d0, int d1,           // 膨胀
    bool is_2D,
    enum ggml_type dst_type   // 输出类型（通常 F16/F32）
);
```

**原理**：将输入图像按滑动窗口展开为 `[KH*KW*IC, OH*OW]` 矩阵，卷积核展平为 `[OC, KH*KW*IC]`，卷积即 `mul_mat(kernel, im2col(input))`。深度卷积每个通道独立，参数量大幅减少。

### 34.8.3 池化、上采样与填充：降采样与修边机

**源码路径**：`ggml/include/ggml.h`（第 2171-2340 行）、`ggml/src/ggml.c` 对应实现

```c
// 池化
struct ggml_tensor * ggml_pool_1d(struct ggml_context * ctx, struct ggml_tensor * a, enum ggml_op_pool op, int k0, int s0, int p0);
struct ggml_tensor * ggml_pool_2d(struct ggml_context * ctx, struct ggml_tensor * a, enum ggml_op_pool op, int k0, int k1, int s0, int s1, float p0, float p1);
struct ggml_tensor * ggml_pool_2d_back(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * af, enum ggml_op_pool op, int k0, int k1, int s0, int s1, float p0, float p1); // 反向

// 上采样/插值
struct ggml_tensor * ggml_upscale(struct ggml_context * ctx, struct ggml_tensor * a, int scale_factor, enum ggml_scale_mode mode);
struct ggml_tensor * ggml_interpolate(struct ggml_context * ctx, struct ggml_tensor * a, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, uint32_t mode);

// 填充
struct ggml_tensor * ggml_pad(struct ggml_context * ctx, struct ggml_tensor * a, int p0, int p1, int p2, int p3);
struct ggml_tensor * ggml_pad_ext(struct ggml_context * ctx, struct ggml_tensor * a, int lp0, int rp0, int lp1, int rp1, int lp2, int rp2, int lp3, int rp3);
struct ggml_tensor * ggml_pad_circular(struct ggml_context * ctx, struct ggml_tensor * a, int p0, int p1, int p2, int p3); // 环形填充
struct ggml_tensor * ggml_pad_reflect_1d(struct ggml_context * ctx, struct ggml_tensor * a, int p0, int p1); // 反射填充

// 其他
struct ggml_tensor * ggml_roll(struct ggml_context * ctx, struct ggml_tensor * a, int shift0, int shift1, int shift2, int shift3); // 循环移位
struct ggml_tensor * ggml_tri(struct ggml_context * ctx, struct ggml_tensor * a, enum ggml_tri_type type); // 三角矩阵 mask
```

## 34.9 注意力之链：注意力机制与位置编码 —— 序列模型的「注意力之链」

### 34.9.1 Softmax 及其扩展：ggml_soft_max_impl 等归一化熔炉

**源码路径**：`ggml/include/ggml.h`（第 1868-1919 行）、`ggml/src/ggml.c`（第 4017-4068 行）

```c
// 基础 Softmax
struct ggml_tensor * ggml_soft_max(struct ggml_context * ctx, struct ggml_tensor * a);
struct ggml_tensor * ggml_soft_max_inplace(struct ggml_context * ctx, struct ggml_tensor * a);

// 扩展版：支持 mask、scale、ALiBi 最大偏置
struct ggml_tensor * ggml_soft_max_ext(
    struct ggml_context * ctx,
    struct ggml_tensor * a,
    struct ggml_tensor * mask,  // 可选注意力 mask
    float scale,                // logit 缩放（通常 1/sqrt(d_head)）
    float max_bias              // ALiBi 线性偏置最大值
);
struct ggml_tensor * ggml_soft_max_ext_back(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, float scale, float max_bias); // 反向

// Attention Sinks 支持（为首个 token 保留注意力）
void ggml_soft_max_add_sinks(struct ggml_tensor * a, struct ggml_tensor * sinks);
```

### 34.9.2 融合算子：ggml_flash_attn_ext Flash Attention

**源码路径**：`ggml/include/ggml.h`（第 2412-2431 行）、`ggml/src/ggml.c`（第 5404-5442 行）

```c
// 融合 Q@K^T、softmax、@V 三步，避免中间大矩阵落内存
struct ggml_tensor * ggml_flash_attn_ext(
    struct ggml_context * ctx,
    struct ggml_tensor * q,           // [n_head, d_head, n_seq, n_batch]
    struct ggml_tensor * k,           // 同形状
    struct ggml_tensor * v,           // 同形状
    struct ggml_tensor * mask,        // 可选 mask
    float scale,                      // logit 缩放
    float max_bias,                   // ALiBi 偏置
    float logit_softcap               // logit 软上限（如 Gemma 用 30.0）
);
void ggml_flash_attn_ext_set_prec(struct ggml_tensor * a, enum ggml_prec prec); // 设置计算精度
void ggml_flash_attn_ext_add_sinks(struct ggml_tensor * a, struct ggml_tensor * sinks);
struct ggml_tensor * ggml_flash_attn_back(...); // 反向传播
```

**Flash Attention 的工程价值**：将 `O(N^2)` 显存占用的注意力矩阵拆分为块，在 SRAM 内完成 softmax，实现显存与算力的双重节省。

### 34.9.3 旋转位置编码：ggml_rope_impl 等 RoPE

**源码路径**：`ggml/include/ggml.h`（第 1926-2061 行）、`ggml/src/ggml.c`（第 4138-4264 行）

```c
// 基础 RoPE：仅频率参数
struct ggml_tensor * ggml_rope(
    struct ggml_context * ctx,
    struct ggml_tensor * a,           // [n_head, d_head/2, n_seq, n_batch] 复数实部虚部交错
    struct ggml_tensor * b,           // 位置索引 [n_seq]
    int n_dims,                       // 旋转维度数（通常 d_head）
    int mode                          // 模式位：NEOX 风格等
);

// 扩展版：支持 YaRN 外推、频率因子
struct ggml_tensor * ggml_rope_ext(
    struct ggml_context * ctx,
    struct ggml_tensor * a,
    struct ggml_tensor * b,           // 位置
    struct ggml_tensor * c,           // 频率因子（可选，如 phi3-128k）
    int n_dims, int mode,
    int n_ctx_orig,                   // 原始训练上下文长度
    float freq_base, float freq_scale, float ext_factor, float attn_factor, float beta_fast, float beta_slow
);

// 多维 RoPE（多模态场景，如图像 patch 位置）
struct ggml_tensor * ggml_rope_multi(
    struct ggml_context * ctx,
    struct ggml_tensor * a,
    struct ggml_tensor * b,
    struct ggml_tensor * c,
    int n_dims,
    int sections[GGML_MROPE_SECTIONS], // 各段维度划分
    int mode, int n_ctx_orig,
    float freq_base, float freq_scale, float ext_factor, float attn_factor, float beta_fast, float beta_slow
);

// YARN 维度修正计算
void ggml_rope_yarn_corr_dims(int n_dims, int n_ctx_orig, float freq_base, float beta_fast, float beta_slow, float dims[2]);
```

**RoPE 的数学本质**：将位置编码融入 Query/Key 的复数旋转，`q_k = q * exp(i * pos * freq)`，实现相对位置感知且支持外推。

## 34.10 状态追踪：状态空间与循环模型 —— RNN和Mamba的「状态追踪」

### 34.10.1 状态空间模型（SSM）

**源码路径**：`ggml/include/ggml.h`（第 2445-2459 行）、`ggml/src/ggml.c`（第 5559-5646 行）

```c
// SSM 卷积：状态空间模型的卷积计算
struct ggml_tensor * ggml_ssm_conv(struct ggml_context * ctx, struct ggml_tensor * sx, struct ggml_tensor * c);

// SSM Scan：并行扫描操作，用于训练时的并行化
struct ggml_tensor * ggml_ssm_scan(
    struct ggml_context * ctx,
    struct ggml_tensor * s,       // 初始状态
    struct ggml_tensor * x,       // 输入序列
    struct ggml_tensor * dt,      // 时间步缩放
    struct ggml_tensor * A,       // 状态转移矩阵
    struct ggml_tensor * B,       // 输入投影
    struct ggml_tensor * C,       // 输出投影
    struct ggml_tensor * ids      // 专家 ID（MoE 场景）
);
```

### 34.10.2 RWKV 家族

**源码路径**：`ggml/include/ggml.h`（第 2518-2548 行）、`ggml/src/ggml.c`（第 5767-5882 行）

```c
// RWKV-v6 WKV 计算
struct ggml_tensor * ggml_rwkv_wkv6(
    struct ggml_context * ctx,
    struct ggml_tensor * k, struct ggml_tensor * v, struct ggml_tensor * r,
    struct ggml_tensor * tf, struct ggml_tensor * td, struct ggml_tensor * state
);

// RWKV-v7 WKV 计算
struct ggml_tensor * ggml_rwkv_wkv7(
    struct ggml_context * ctx,
    struct ggml_tensor * r, struct ggml_tensor * w, struct ggml_tensor * k,
    struct ggml_tensor * v, struct ggml_tensor * a, struct ggml_tensor * b,
    struct ggml_tensor * state
);
```

### 34.10.3 门控线性注意力与 DeltaNet

**源码路径**：`ggml/include/ggml.h`（第 2528-2584 行）、`ggml/src/ggml.c` 对应实现

```c
// 门控线性注意力
struct ggml_tensor * ggml_gated_linear_attn(
    struct ggml_context * ctx,
    struct ggml_tensor * k, struct ggml_tensor * v, struct ggml_tensor * q,
    struct ggml_tensor * g, struct ggml_tensor * state, float scale
);

// Gated DeltaNet：DeltaNet 变体与门控结合
struct ggml_tensor * ggml_gated_delta_net(
    struct ggml_context * ctx,
    struct ggml_tensor * q, struct ggml_tensor * k, struct ggml_tensor * v,
    struct ggml_tensor * g, struct ggml_tensor * beta, struct ggml_tensor * state
);
```

**共性特征**：这些算子通常接受 `state` 张量作为输入，输出新 `state`，允许跨序列递归处理长时间依赖，无需保留全部历史（对比标准 Transformer 的 KV Cache）。

## 34.11 研发工坊：自定义算子与高层训练框架 —— 扩展与优化的「研发工坊」

### 34.11.1 突破内置算子限制：ggml_map_custom1_impl 等自定义算子

**源码路径**：`ggml/include/ggml.h`（第 2598-2682 行）、`ggml/src/ggml.c`（第 5915-6106 行）

```c
// 一元自定义算子：对每个元素应用用户回调
struct ggml_tensor * ggml_map_custom1(
    struct ggml_context * ctx,
    struct ggml_tensor * a,
    ggml_custom1_op_t fun,    // 用户回调：void fun(float *out, const float *in, int n, void *userdata)
    int n_tasks,              // 并行任务数
    void * userdata
);
struct ggml_tensor * ggml_map_custom1_inplace(...); // 原地版本

// 二元/三元自定义算子
struct ggml_tensor * ggml_map_custom2(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, ggml_custom2_op_t fun, int n_tasks, void * userdata);
struct ggml_tensor * ggml_map_custom3(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, struct ggml_tensor * c, ggml_custom3_op_t fun, int n_tasks, void * userdata);

// 最高自由度：自定义 4D 算子，指定输出形状、任意输入数量
struct ggml_tensor * ggml_custom_4d(
    struct ggml_context * ctx,
    enum ggml_type type,
    int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3,
    struct ggml_tensor ** args, int n_args,
    ggml_custom_op_t fun, int n_tasks, void * userdata
);

// 原地自定义算子
struct ggml_tensor * ggml_custom_inplace(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor ** args, int n_args, ggml_custom_op_t fun, int n_tasks, void * userdata);
```

### 34.11.2 训练框架：ggml_opt_init 等自动化流水线

**源码路径**：`ggml/include/ggml-opt.h`、`ggml/src/ggml-opt.cpp`

```c
// 数据集管理
ggml_opt_dataset_t ggml_opt_dataset_init(enum ggml_type type_data, enum ggml_type type_label, int64_t ne_datapoint, int64_t ne_label, int64_t ndata, int64_t ndata_shard);
void ggml_opt_dataset_shuffle(ggml_opt_context_t opt_ctx, ggml_opt_dataset_t dataset, int64_t idata);
void ggml_opt_dataset_get_batch(ggml_opt_dataset_t dataset, struct ggml_tensor * data_batch, struct ggml_tensor * labels_batch, int64_t ibatch);

// 优化上下文
ggml_opt_context_t ggml_opt_init(struct ggml_opt_params params);
void ggml_opt_free(ggml_opt_context_t opt_ctx);
void ggml_opt_reset(ggml_opt_context_t opt_ctx, bool optimizer);
void ggml_opt_build(ggml_opt_context_t opt_ctx); // 构建前向/反向/优化图

// 优化器选择
enum ggml_opt_optimizer_type { GGML_OPT_OPTIMIZER_TYPE_ADAMW, GGML_OPT_OPTIMIZER_TYPE_SGD };

// 高层训练循环
void ggml_opt_epoch(ggml_opt_context_t opt_ctx, ggml_opt_dataset_t dataset, ggml_opt_result_t result_train, ggml_opt_result_t result_eval, int64_t idata_split, ggml_opt_epoch_callback callback_train, ggml_opt_epoch_callback callback_eval);
void ggml_opt_fit(ggml_backend_sched_t backend_sched, struct ggml_context * ctx_compute, struct ggml_tensor * inputs, struct ggml_tensor * outputs, ggml_opt_dataset_t dataset, enum ggml_opt_loss_type loss_type, enum ggml_opt_optimizer_type optimizer, ggml_opt_get_optimizer_params get_opt_pars, int64_t nepoch, int64_t nbatch_logical, float val_split, bool silent);
```

**训练流程图解**：

```mermaid
graph LR
    A[前向图] --> B[损失计算]
    B --> C[build_backward_expand]
    C --> D[反向图]
    D --> E[optimizer_step(AdamW/SGD)]
    E --> F[参数更新]
    F -.->|grad_accs 累积梯度| A
```

## 34.12 仓库与运输系统：内存管理与缓冲区抽象 —— 高效的「仓库与运输系统」

### 34.12.1 分配器三个等级：从线性货架到智能仓储

**源码路径**：`ggml/include/ggml-alloc.h`、`ggml/src/ggml-alloc.c`

```c
// Level 1: 简单线性分配器
struct ggml_tallocr ggml_tallocr_new(ggml_backend_buffer_t buffer);
enum ggml_status ggml_tallocr_alloc(struct ggml_tallocr * talloc, struct ggml_tensor * tensor);

// Level 2: 动态张量分配器（多 chunk、最佳适配）
struct ggml_dyn_tallocr * ggml_dyn_tallocr_new(size_t alignment, size_t max_buffer_size);
struct buffer_address ggml_dyn_tallocr_alloc(struct ggml_dyn_tallocr * alloc, size_t size, const struct ggml_tensor * tensor);
void ggml_dyn_tallocr_free_bytes(struct ggml_dyn_tallocr * alloc, struct buffer_address addr, size_t size);

// Level 3: 图级分配器（理解整个计算图，智能复用内存）
ggml_gallocr_t ggml_gallocr_new(ggml_backend_buffer_type_t buft);
ggml_gallocr_t ggml_gallocr_new_n(ggml_backend_buffer_type_t * bufts, int n_bufs);
bool ggml_gallocr_reserve(ggml_gallocr_t galloc, struct ggml_cgraph * graph);
bool ggml_gallocr_alloc_graph(ggml_gallocr_t galloc, struct ggml_cgraph * graph);
size_t ggml_gallocr_get_buffer_size(ggml_gallocr_t galloc, int buffer_id);
```

### 34.12.2 核心算法：vbuffer 与 ggml_gallocr 智能管理员

**源码路径**：`ggml/src/ggml-alloc.c`（`ggml_gallocr_allocate_node` 第 550-623 行、`ggml_gallocr_free_node` 第 625-645 行）

```c
// ### vbuffer: 虚拟缓冲区结构
// 最多 16 个 chunk 组合的虚拟连续内存空间
struct vbuffer {
    ggml_backend_buffer_t chunks[GGML_VBUFFER_MAX_CHUNKS];
};

// ### ggml_gallocr_allocate_node: 核心分配逻辑
// 生命周期分析 + in-place 复用 + view 共享
static void ggml_gallocr_allocate_node(ggml_gallocr_t galloc, struct ggml_tensor * node, int buffer_id) {
    // 1. 已分配或为 view → 跳过
    // 2. 尝试 in-place 复用父张量内存：
    //    - op 支持 in-place (ggml_op_can_inplace)
    //    - 形状布局兼容 (ggml_are_same_layout)
    //    - 父张量仅有一个子节点且无 view 引用
    // 3. 否则从动态分配器申请新内存
}

// ### ggml_gallocr_free_node: 释放逻辑
// 引用计数归零 + 无 view 引用 → 归还内存池
static void ggml_gallocr_free_node(ggml_gallocr_t galloc, struct ggml_tensor * node) {
    if (node->flags & GGML_TENSOR_FLAG_OUTPUT) return; // 输出张量不释放
    // 归还内存到对应 chunk 的空闲块链表，尝试合并相邻块
}
```

**in-place 复用条件**（`ggml/src/ggml-alloc.c:26`）：

```c
// ### ggml_op_can_inplace: 可原位操作判定
bool ggml_op_can_inplace(enum ggml_op op) {
    switch (op) {
        case GGML_OP_FILL: case GGML_OP_SCALE: case GGML_OP_ADD: case GGML_OP_MUL:
        case GGML_OP_RMS_NORM: case GGML_OP_SOFT_MAX: case GGML_OP_ROPE:
        case GGML_OP_SILU_BACK: ... // 共 20+ 个可原位操作
            return true;
        default: return false;
    }
}
```

### 34.12.3 缓冲区抽象与跨后端复制：通用集装箱接口

**源码路径**：`ggml/include/ggml-backend.h`、`ggml/src/ggml-backend.cpp`

```c
// 缓冲区类型抽象：定义内存特性
const char * ggml_backend_buft_name(ggml_backend_buffer_type_t buft);
size_t ggml_backend_buft_get_alignment(ggml_backend_buffer_type_t buft);
size_t ggml_backend_buft_get_alloc_size(ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor);
bool ggml_backend_buft_is_host(ggml_backend_buffer_type_t buft); // 是否主机内存

// 物理缓冲区操作
ggml_backend_buffer_t ggml_backend_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size);
enum ggml_status ggml_backend_buffer_init_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor);

// 跨后端张量复制：同后端 memcpy，跨后端经主机中转
void ggml_backend_tensor_copy(struct ggml_tensor * src, struct ggml_tensor * dst);
void ggml_backend_tensor_copy_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, struct ggml_tensor * src, struct ggml_tensor * dst);
```

## 34.13 万能转换器与智能配送系统：后端抽象与多设备调度 —— 「万能转换器」和「智能配送系统」

### 34.13.1 三层抽象体系：设备、后端、缓冲区类型

**源码路径**：`ggml/include/ggml-backend.h`

```
ggml_backend_buffer_type_t  →  缓冲区类型（CPU内存、GPU显存、NPU内存等）
       ↑
ggml_backend_dev_t          →  物理设备（RTX 4090、Apple M2 NPU 等）
       ↑
ggml_backend_t              →  后端实例（具体实现的句柄，含函数表指针）
```

### 34.13.2 三阶段图拆分：ggml_backend_sched_split_graph 调度器

**源码路径**：`ggml/include/ggml-backend.h`（第 317-352 行）、`ggml/src/ggml-backend.cpp`（第 750-830 行）

```c
// 创建调度器：多后端、缓冲区类型、图大小、并行/卸载开关
ggml_backend_sched_t ggml_backend_sched_new(
    ggml_backend_t * backends,
    ggml_backend_buffer_type_t * bufts,
    int n_backends,
    size_t graph_size,
    bool parallel,      // 并行执行分割后的子图
    bool op_offload     // 不支持的 op 卸载到其他后端
);

// 三阶段图拆分算法
void ggml_backend_sched_split_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    // Phase 1: 根据权重位置、输入标记、用户指定分配初始后端 ID
    // Phase 2: 向下扩展 —— 让相邻节点倾向于同一后端（GPU 优先）
    // Phase 3: 向上扩展 —— 向输入节点传播后端分配
}

// 执行流程
bool ggml_backend_sched_alloc_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph); // 分配跨设备内存
enum ggml_status ggml_backend_sched_graph_compute(ggml_backend_sched_t sched, struct ggml_cgraph * graph); // 同步执行
```

**调度决策逻辑**（`ggml_backend_sched_backend_id_from_cur`）：
1. 预分配张量 → 必须在其所在后端
2. 图输入 → 最后一个后端（通常 CPU）
3. 权重张量 → 权重所在后端（优先 GPU）
4. 操作卸载 → 高优先级后端主动接管
5. 相邻节点 → 同后端（减少跨设备拷贝）

```mermaid
flowchart TD
    A[Phase 1: 预分配/输入/权重/卸载] --> B[Phase 2: 向下扩展 GPU优先]
    B --> C[Phase 3: 向上传播 统一后端]
    C --> D[生成 splits[] 子图]
    D --> E[ggml_gallocr_reserve_n 内存预留]
    E --> F[ggml_backend_sched_alloc_graph 分配]
    F --> G[ggml_backend_sched_graph_compute 执行]
```

## 34.14 海关集装箱：GGUF 文件格式 —— 模型的「海关集装箱」

### 34.14.1 GGUF 文件组成

```
GGUF 文件结构：
┌─────────────────────┐
│ Header (Magic + Ver)│  ← "GGUF", version, tensor_count, kv_count
├─────────────────────┤
│ Metadata KV Pairs   │  ← 模型超参数、词表大小、RoPE 配置、聊天模板等
├─────────────────────┤
│ Tensor Info Array   │  ← 每个张量：name, shape, type, offset
├─────────────────────┤
│ Tensor Data         │  ← 实际权重数据（按 alignment 对齐）
└─────────────────────┘
```

### 34.14.2 核心读写 API

**源码路径**：`ggml/include/gguf.h`、`ggml/src/gguf.cpp`

```c
// 读取
struct gguf_context * gguf_init_from_file(const char * fname, struct gguf_init_params params);
uint32_t gguf_get_version(const struct gguf_context * ctx);
int64_t  gguf_get_n_kv(const struct gguf_context * ctx);
int64_t  gguf_find_key(const struct gguf_context * ctx, const char * key);
// 类型安全读取
uint32_t gguf_get_val_u32(const struct gguf_context * ctx, int64_t key_id);
const char * gguf_get_val_str(const struct gguf_context * ctx, int64_t key_id);
float gguf_get_val_f32(const struct gguf_context * ctx, int64_t key_id);

// 张量查询
int64_t gguf_get_n_tensors(const struct gguf_context * ctx);
int64_t gguf_find_tensor(const struct gguf_context * ctx, const char * name);
enum ggml_type gguf_get_tensor_type(const struct gguf_context * ctx, int64_t tensor_id);
size_t gguf_get_tensor_offset(const struct gguf_context * ctx, int64_t tensor_id);
size_t gguf_get_tensor_size(const struct gguf_context * ctx, int64_t tensor_id);

// 写入
struct gguf_context * gguf_init_empty(void);
void gguf_set_val_u32(struct gguf_context * ctx, const char * key, uint32_t val);
void gguf_set_val_str(struct gguf_context * ctx, const char * key, const char * val);
void gguf_add_tensor(struct gguf_context * ctx, const struct ggml_tensor * tensor);
void gguf_set_tensor_type(struct gguf_context * ctx, const char * name, enum ggml_type type);
void gguf_set_tensor_data(struct gguf_context * ctx, const char * name, const void * data);
bool gguf_write_to_file(const struct gguf_context * ctx, const char * fname, bool only_meta);
```

### 34.14.3 内部实现：gguf_reader/gguf_writer 类

**源码路径**：`ggml/src/gguf.cpp`（`gguf_reader` 第 509-808 行、`gguf_writer_buf/file` 第 178-320 行）

```c
// 序列化写出流程
static void gguf_write_out(const struct gguf_context * ctx, writer_t & gw, bool only_meta) {
    // 1. 写文件头
    gw.write(GGUF_MAGIC[0..3]); gw.write(ctx->version); gw.write(n_tensors); gw.write(n_kv);
    // 2. 写所有 KV
    for (kv : ctx->kv) gw.write(kv);
    // 3. 写张量元信息
    for (info : ctx->info) gw.write_tensor_meta(info);
    // 4. 对齐到 alignment
    gw.pad(ctx->alignment);
    // 5. 写张量数据（only_meta 时跳过）
    if (!only_meta) for (info : ctx->info) gw.write_tensor_data(info, offset_data, alignment);
}
```

## 34.15 通用工具箱：基础设施与工具集 —— 跨平台的「通用工具箱」

### 34.15.1 跨平台内存与文件

**源码路径**：`ggml/src/ggml.c`（`ggml_aligned_malloc` 第 326-373 行、`ggml_fopen` 第 607-627 行）

```c
// 64/128/256 字节对齐分配（Windows _aligned_malloc, POSIX posix_memalign, macOS vm_allocate）
void * ggml_aligned_malloc(size_t size);
void ggml_aligned_free(void * ptr, size_t size);

// UTF-8 路径跨平台打开
FILE * ggml_fopen(const char * fname, const char * mode);
```

### 34.15.2 错误处理与调试

**源码路径**：`ggml/src/ggml.c`（`ggml_abort` 第 245-263 行、`ggml_print_backtrace` 第 151-230 行）

```c
// 致命错误回调
typedef void (*ggml_abort_callback_t)(const char * message);
ggml_abort_callback_t ggml_set_abort_callback(ggml_abort_callback_t callback);
GGML_NORETURN void ggml_abort(const char * file, int line, const char * fmt, ...);

// 调用栈打印：Linux 用 gdb/lldb attach，macOS 需 GGML_BACKTRACE_LLDB，Android 用 _Unwind_Backtrace
void ggml_print_backtrace(void);
```

### 34.15.3 时间测量与版本信息

**源码路径**：`ggml/include/ggml.h`（第 603-607 行）

```c
void ggml_time_init(void);          // 初始化高精度计时器
int64_t ggml_time_ms(void);         // 毫秒时间戳
int64_t ggml_time_us(void);         // 微秒时间戳
int64_t ggml_cycles(void);          // CPU 周期计数
int64_t ggml_cycles_per_ms(void);   // 每毫秒周期数（用于换算）
const char * ggml_version(void);    // 版本字符串
const char * ggml_commit(void);     // Git commit hash
```

## 34.16 并发与轻量化保障：线程同步、量化与性能优化

### 34.16.1 全局临界区：ggml_critical_section_start

**源码路径**：`ggml/src/ggml-threading.cpp`（第 5-11 行）

```c
static std::mutex ggml_critical_section_mutex;
void ggml_critical_section_start(void) { ggml_critical_section_mutex.lock(); }
void ggml_critical_section_end(void)   { ggml_critical_section_mutex.unlock(); }
```
**用途**：量化表惰性初始化、全局状态保护（如 `ggml_quantize_init` 内部调用）。

### 34.16.2 量化系统接口

**源码路径**：`ggml/include/ggml.h`、`ggml/src/ggml.c`（`ggml_quantize_chunk` 约第 7566-7811 行）

```c
void ggml_quantize_init(enum ggml_type type);      // 初始化 IQ 系列查找表
void ggml_quantize_free(void);                     // 释放量化状态
bool ggml_quantize_requires_imatrix(enum ggml_type type); // 是否需重要性矩阵
size_t ggml_quantize_chunk(enum ggml_type type, const float * src, void * dst, int64_t start, int64_t nrows, int64_t n_per_row, const float * imatrix); // 核心量化分发
```

**量化类型全景**：

| 家族 | 代表类型 | 块大小 | 特点 |
|------|----------|--------|------|
| 线性量化 | Q4_0, Q4_1, Q5_0, Q5_1, Q8_0 | 32 | 简单缩放+量化 |
| K-quant | Q2_K ~ Q6_K | 256 | 多级缩放、子块量化 |
| IQ 系列 | IQ2_XXS, IQ3_S, IQ4_NL | 可变 | 查找表非精确量化 |
| 新锐 | MXFP4, NVFP4, TQ1_0 | 32/64 | 硬件原生 FP4/三元量化 |

### 34.16.3 类型特征表：ggml_type_traits

**源码路径**：`ggml/include/ggml.h`（约第 1930-1950 行）、`ggml/src/ggml.c` `type_traits` 静态数组

```c
struct ggml_type_traits {
    const char * type_name;           // "F32", "Q4_0"
    size_t blck_size;                 // 块元素数
    size_t type_size;                 // 块字节数
    bool is_quantized;                // 是否量化
    ggml_to_float_t to_float;         // 反量化回调
    ggml_from_float_t from_float;     // 量化回调
};
```
每种类型（F32/F16/BF16/Q4_0/Q8_0/...）在编译期注册特征，运行时通过 `ggml_get_type_traits(type)` 查询，实现量化/反量化的多态分发。

## 34.17 设计中的取舍

### 34.17.1 为什么不用统一的算子基类 + 虚函数分发？

**答案**：性能与零开销抽象。
- C 风格的 `op` 枚举 + `op_params` 字节数组避免了虚函数表查找、动态分配、指针追踪。
- 后端内核通过 `switch(op)` 直接分发，编译器可内联优化。
- `op_params` 紧凑存储（64 字节）消除了每个算子单独分配参数结构的开销。

### 34.17.2 为什么用 `view_tensor` 实现 in-place 而不是单独的 in-place 标记？

**答案**：统一内存模型。
- `view` 天然表达「共享数据、不同形状/偏移」，in-place 本质是「输出 view 输入」。
- 内存分配器（gallocr）只需检查 `view_src` 与 `n_children==0` 即可判断可否复用，无需额外标记。
- 反向传播时 `view` 的梯度自然累加到源张量，符合自动微分链式法则。

### 34.17.3 GGUF 为什么要用「键值对 + 张量数组」而非扁平结构？

**答案**：可扩展性与流式读取。
- KV 区与张量区分离，允许只读元数据（`only_meta=true`）而不加载权重。
- 新增字段不破坏旧版本解析器（忽略未知 key）。
- 张量信息数组支持 `mmap` 直接映射，`ggml_backend_tensor_alloc` 时按 offset 定位数据。

### 34.17.4 为什么 `ggml_tensor` 用 `nb` 字节步长而非 `strides` 元素步长？

**答案**：原生支持量化类型。量化类型（如 Q4_0）每 32 个元素占 18 字节，非整数倍元素大小，字节步长 `nb` 可精确描述任意量化布局，元素步长则无法表示。

### 34.17.5 为什么 `op_params` 用字节数组而非结构体？

**答案**：避免每个算子单独 `malloc` 参数结构，消除内存碎片与分配开销。所有参数序列化压扁在 64 字节定长数组，`ggml_set_op_params_i32/f32` 统一读写，缓存友好且零开销。

### 34.17.6 `ggml_gallocr` 为何采用生命周期分析而非简单引用计数？

**答案**：支持 in-place 内存复用。引用计数无法感知「父张量仅被一个子节点使用且无 view 引用」这一 in-place 条件。gallocr 通过 `n_children`/`n_views` 精确建模张量生命周期，实现激进内存复用。

## 34.18 动手练习

1. **阅读 GGML 算子实现：统一模板分析**
   选择三个算子进行分析：
   1. `ggml_add`（二元算术）及 `ggml_add_inplace`
   2. `ggml_silu`（一元激活）及 `ggml_silu_inplace`
   3. `ggml_reshape_2d`（形状变换）
   
   需要回答：
   - 三类算子的共同点：都是如何创建结果张量的？（dup_tensor/view_tensor）当使用 `inplace` 版本时，张量的 op 变成什么？
   - 简述 `op_params` 数组在每类算子中的作用
   - 画一个 3 步计算例子：`a → silu(a) → reshape(2,x) → add(b,.)`，说明每一步返回张量的类型和关系

2. **理解反向传播的梯度计算**
   `ggml_build_backward_expand` 是训练流程的核心。请阅读其实现，重点理解：
   1. 如何生成 `ggml_tensor` 的 `grad_accs`（梯度累积器）
   2. `ggml_compute_backward` 中各算子分支的梯度计算公式
   3. 对比前向图与反向图的创建差异
   
   实践任务：
   - 手写一个简单的 2 层感知机（`x→silu→matmul→silu`），构建其前向图与反向图
   - 手动计算每个参数的梯度，检验与代码逻辑是否一致
   - 思考题：为什么 softmax 的反向传播这么复杂？梯度如何分解为两部分？

3. **图分配器内存优化**
   `ggml_gallocr` 是 GGML 中的图级内存分配器，负责分析计算图并优化内存使用。
   1. 阅读 `ggml/src/ggml-alloc.c` 中的 `ggml_gallocr_allocate_node` 函数
   2. 理解 in-place 复用的条件（`n_children==0` + layout 匹配 + not_output）
   3. 学习视图张量（view）的内存共享如何不额外分配内存
   
   任务：
   - 对比两个图：一个使用 `ggml_mul + ggml_mul`（创建新内存）
   - 另外一个：使用 `ggml_mul_inplace`
   - 在两种情况下用 `ggml_gallocr_get_buffer_size` 统计需要的缓冲区大小
   - 输出比较报告，比较内存差异和第 2 个图可进行 in-place 复用的位置

4. **GGUF 文件格式：读写与遍历**
   该练习聚焦 GGUF 文件的实际操作：
   1. 读取：用 `gguf_init_from_file` + 遍历 `gguf_get_n_kv` / `gguf_get_val_*` 读取模型元数据
   2. 写入：用 `gguf_init_empty` + `gguf_set_val_*` 构建最小化 GGUF 文件并保存
   3. 修改：读取现有 GGUF，改动某个关键参数（如 `rope_freq_base`），再写回
   
   思考问题：
   - GGUF 文件头的四个字节是什么？
   - 对齐参数(alignment)为何通常为 32 或 64？
   - KV 分区与张量数据区的分界是什么标记？
   - 如何在写入、追加权限之外加入密码或版本控制？

5. **多后端调度器实战**
   研究 `ggml_backend_sched_split_graph` 的三阶段拆分逻辑：
   1. Phase 1: 从缓冲区到节点分配后端（权重通常在 GPU）
   2. Phase 2: 从当前后端向下传播（让附近节点在同后端）
   3. Phase 3: 从当前后端向上传播（向用户节点传递）
   
   实践任务：
   - 构造一个混合计算图：一个在 GPU 上的 `mul_mat` → 节点 B（不指定后端）→ softmax
   - 使用 `ggml_backend_sched_print_assignments` 打印每个节点的后端分配
   - 手动调整开关：只用 GPU for matmul，其他全部 CPU，重新编译并重新打印分配
   - 根据结果分析调度决策逻辑是否合理，尤其是当某个算子仅 CPU 支持时会发生甚么？

6. **量化与精度：从 FP32 到 Q4_0**
   GGML 的核心之一是量化技术，支持从 FP32 到各种量化格式：
   1. 阅读 `ggml_quantize_chunk` 支持的全部类型
   2. 理解量化块（block）的结构：缩放因子(d)、量化值、最小值(m)
   3. 阅读量化函数（如 `quantize_q4_0`）的实现原理
   
   实践：
   - 写一个测试脚本：创建一个 `[128]` 的随机 FP32 张量，量化到 Q4_0 并再反量化
   - 计算量化误差：平均绝对误差（MAE）、信噪比（SNR）
   - 画出误差分布直方图并分析分量特点
   - 尝试 Q2_K、Q4_K、Q6_K、Q8_0，比较压缩比与精度平衡关系

7. **一个完整推理示例：LLaMA 文本生成**
   综合所学知识，走完一个完整的推理流程：
   1. 用 `ggml_init` 创建上下文
   2. 加载模型权重：从 GGUF 读取并构建 张量池
   3. 构建计算图：包含 embedding→transformer blocks→LM head
   4. 使用 `ggml_backend_sched` 进行多后端调度
   5. 执行 `ggml_backend_sched_graph_compute` 完成推理
   6. 提取 logits，用 `ggml_soft_max` 计算概率
   7. argmax 选择最优 token
   
   提示：可以参考 llama.cpp 的 simple infer 执行流程，理解上面这个宏观流程如何映射到具体 GGML 函数调用

## 34.19 本章小结

这一章中我们学习了 GGML 张量计算框架的核心基础设施。首先深入剖析了 `ggml_tensor` 与 `ggml_cgraph` 两大核心数据结构，理解了张量的类型、形状、步长、操作拓扑与视图关系，以及计算图的节点拓扑序、引用计数与梯度管理。其次系统梳理了张量属性查询家族，掌握了连续性、转置、维度数等判别逻辑对后端内核选择的决定性影响。接着详细讲解了算子构建的统一模板：从一元算子 `ggml_unary_impl` 到二元算术、激活函数、归一化、矩阵乘法、GLU 门控、形状变换、卷积、注意力、RoPE、SSM/RWKV 等全谱系算子的创建模式与参数编码。随后探讨了反向传播引擎的图构建机制，包括前向图扩展、反向图扩展、梯度累加器管理与各算子梯度公式分发。在内存管理层面，对比了三级分配器的设计演进，重点分析了 `ggml_gallocr` 的生命周期分析、in-place 复用条件与 view 共享机制。后端抽象部分揭示了缓冲区类型、设备、后端三层解耦与多后端调度器的三阶段图拆分算法。GGUF 文件格式解析了元数据 KV、张量信息与数据区的二进制布局及读写流程。最后覆盖了跨平台工具、量化系统、类型特征表等基础设施。

本章我们一起学习了以下概念：

| 概念 | 解释 |
|------|------|
| `ggml_tensor` | 张量核心结构：包含类型(type)、形状(ne[4])、步长(nb[4])、操作符(op)、源张量(src[])、视图关系(view_src)、数据指针(data) |
| `ggml_cgraph` | 计算图：包含节点(nodes)、叶子节点(leafs)、梯度(grads)、引用计数(use_counts)、访问哈希表(visited_hash_set) |
| `ggml_tensor` 构建模式 | 核心三部曲：分配 `dup_tensor`/`view_tensor` → 设置 `op_params` → 链接 `src` |
| `ggml_dup_tensor` vs `view_tensor` | dup 复制元数据但独立内存；view 共享内存（零拷贝），是 in-place 操作的基础 |
| `ggml_unary`/`unary_impl` | 一元算子的统一实现：所有激活函数、三角函数等都通过该函数创建 |
| `ggml_is_contiguous` 家族 | 连续性检查：影响后端优化路径选择，全流程的重要判别条件 |
| `ggml_is_transposed`/`is_permuted` | transposed 比较 `nb[0]>nb[1]`，permuted 更宽泛，影响形状变换策略 |
| in-place 算子 | 通过 `ggml_view_tensor` 标记，在 `ggml_op_can_inplace` 中列出可原位执行的操作 |
| GLU 家族 | ReGLU/GEGLU/SwiGLU 三大变体，分别使用 ReLU/GELU/SiLU 作为门控激活函数 |
| `ggml_mul_mat` | 矩阵乘法核心算子，`A[m,k] @ B[k,n] → C[m,n]`，支持广播 |
| `ggml_mul_mat_id` | MoE 专家路由算子，根据 ids 将输入路由到对应的专家矩阵 |
| RoPE | 旋转位置编码，通过 `ggml_rope` 及其多版本支持 YaRN 外推和多维度分节 |
| flash_attn_ext | 融合 Attention 计算，支持 mask、ALiBi、scale、softcap 等参数 |
| soft_max_ext | 可扩展 softmax，支持 mask、scale、ALiBi 最大偏置 |
| `ggml_conv` 系列 | 1D/2D/3D 卷积，深度卷积(dw)、转置卷积，基于 im2col+mul_mat 实现 |
| `ggml_im2col` | 图像块展开，将卷积转为矩阵乘法，深度卷积核心 |
| `ggml_tallocr`/`gallocr` | 张量及图级内存分配器，从简单线性分配到图感知分配 |
| `ggml_dyn_tallocr` | 动态张量分配器，支持多 chunk、最佳适配、内存合并 |
| vbuffer | 虚拟缓冲区，最多支持 16 个 chunk（多缓冲区）的组合管理 |
| `ggml_backend_buft`/`dev`/`backend` | 设备缓冲区类型/设备/后端三层抽象，屏蔽硬件差异 |
| `ggml_backend_sched` | 调度器，支持三阶段图拆分：预分配、向下扩展、向上传播 |
| `ggml_backend_multi_buffer` | 多缓冲区，由多个子缓冲区合并成的复合缓冲区 |
| `gguf_context` | GGUF 文件内存映像，包含 version、kv 列表、tensor 信息、对齐等 |
| `gguf_reader` | 低级文件读取器，处理对齐、长度前缀、字节序等细节 |
| `gguf_writer_buf/file` | 序列化写入器，支持内存缓冲区和文件两种写入目标 |
| `gguf_kv` | 键值对结构，支持所有标量、数组、字符串、混合类型 |
| IQ 系列量化 | Inexact 量化（IQ2_XXS 等），通过预定义的 `kdefaults_*` 查找表实现 |
| `ggml_build_forward_expand` | 从输出张量反向 DFS 遍历，将所有依赖的父节点加入计算图 |
| `ggml_build_backward_expand` | 生成反向传播子图，为每个可训练参数计算梯度 |
| `ggml_compute_backward` | 核心反向传播实现，按 op 类型分发到对应的梯度计算公式 |
| `ggml_add_or_set` | 梯度累加或初次设置，配合 `grad_accs` 实现跨 batch 梯度累积 |
| op_params 机制 | 每个算子可在 `op_params` 字段中嵌入参数，无需额外数据结构 |
| `ggml_opt` 系列 | 训练优化框架：数据集、优化器（AdamW/SGD）、训练循环 |
| `ggml_opt_dataset` | 训练数据集，支持分片打乱、批次提取 |
| `ggml_opt_step_adamw`/`sgd` | 优化器一步更新算子，直接修改模型参数的权重张量 |
| graph 反向传播 | `llama_opt_build` 的反向图构建：复制前向图 + `grad_accs` 管理 + `optimizer_step` 插入 |

下一章中，我们将学习 **CPU 后端引擎 —— x86/ARM/RISC-V/LoongArch/s390/WASM 的「百家争鸣」**，深入 ggml-cpu 后端的 SIMD 抽象、量化矩阵乘、AMX/KleidiAI 加速，以及各架构特性的检测与评分。



