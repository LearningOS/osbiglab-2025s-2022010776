#include <cviruntime.h>
#include <iostream>
#include <sstream>
#include <string>
#include <chrono>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

using dtype = float;
namespace py = pybind11;

constexpr int NUM_LAYERS = 6;
constexpr int NUM_HEADS = 2;
constexpr int NUM_Q_HEADS = 8;
constexpr int HEAD_DIM = 64;
constexpr int CAPACITY = 1024;
constexpr int BLOCK_LEN = 16;
constexpr int CONTEXT_LEN = 512;
constexpr int ATTN_GROUP_SIZE = NUM_Q_HEADS / NUM_HEADS;
constexpr int NUM_CONTEXT_PAGES = CONTEXT_LEN / BLOCK_LEN;

struct KVPage {
    dtype k[BLOCK_LEN][HEAD_DIM];
    dtype v[BLOCK_LEN][HEAD_DIM];
};

struct KVPageDigest {
    dtype k_max[HEAD_DIM];
    dtype k_min[HEAD_DIM];
};

struct KVPool {
    KVPage pages[CAPACITY][NUM_LAYERS][NUM_HEADS];
    KVPageDigest page_digests[CAPACITY][NUM_LAYERS][NUM_HEADS];
    bool page_valid[CAPACITY][NUM_LAYERS][NUM_HEADS];
};

template <typename T>
py::array create_array_from_tensor(CVI_TENSOR *tensor) {
    auto tensor_shape = CVI_NN_TensorShape(tensor);
    auto tensor_ptr = CVI_NN_TensorPtr(tensor);

    std::vector<ssize_t> shape;
    for (int i = 0; i < tensor_shape.dim_size; ++i) {
        shape.push_back(tensor_shape.dim[i]);
    }

    std::vector<ssize_t> strides;
    for (int i = 0; i < tensor_shape.dim_size; ++i) {
        int stride = sizeof(T);
        for (int j = i + 1; j < tensor_shape.dim_size; ++j) {
            stride *= tensor_shape.dim[j];
        }
        strides.push_back(stride);
    }

    py::array result(
        py::buffer_info(
            tensor_ptr,
            sizeof(T),
            py::format_descriptor<T>::format(),
            tensor_shape.dim_size,
            shape,
            strides
        )
    );
    return result;
}

enum KVMode {
    SLIDING_WINDOW,
    QUEST,
};

class KLM {
public:
    KVPool *kv_pool;
    int page_index = 0;
    int page_place = 0;
    py::function trace_tensor;
    py::function execute;
    KVMode mode = SLIDING_WINDOW;
    CVI_MODEL_HANDLE head;
    CVI_MODEL_HANDLE tail;
    CVI_MODEL_HANDLE segments[NUM_LAYERS - 1];

    explicit KLM(const std::string& path) {
        std::cerr << "Loading KLM model from " << path << std::endl;
        CVI_NN_RegisterModel((path + "/seg_head.cvimodel").c_str(), &head);
        CVI_NN_RegisterModel((path + "/seg_tail.cvimodel").c_str(), &tail);
        for (int i = 0; i < NUM_LAYERS - 1; i++) {
            CVI_NN_RegisterModel((path + "/seg_" + std::to_string(i) + ".cvimodel").c_str(), &segments[i]);
        }
        std::cerr << "Building KV Pool" << std::endl;
        kv_pool = new KVPool();
        std::cerr << "KV Pool built" << std::endl;
    }

    void pyTraceTensor(const py::function &trace_tensor_func, const py::function& execute_func) {
        trace_tensor = trace_tensor_func;
        execute = execute_func;
    }

    void pySetKVMode(const KVMode& kv_mode) {
        mode = kv_mode;
    }

    CVI_TENSOR *callHead(void *input_ids, void *rotary_pos_emb) const {
        const auto start = std::chrono::high_resolution_clock::now();
        CVI_TENSOR *input_tensors;
        CVI_TENSOR *output_tensors;
        int input_num;
        int output_num;
        CVI_NN_GetInputOutputTensors(head, &input_tensors, &input_num, &output_tensors, &output_num);

        CVI_TENSOR *input_ids_tensor = CVI_NN_GetTensorByName("input_ids*", input_tensors, input_num);
        CVI_TENSOR *rotary_pos_emb_tensor = CVI_NN_GetTensorByName("rotary_pos_emb*", input_tensors, input_num);

        CVI_NN_SetTensorPtr(input_ids_tensor, input_ids);
        CVI_NN_SetTensorPtr(rotary_pos_emb_tensor, rotary_pos_emb);

        CVI_NN_Forward(head, input_tensors, input_num, output_tensors, output_num);

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cerr << "Head forward time: " << duration.count() << " ms" << std::endl;
        return output_tensors;
    }

    CVI_TENSOR *callSegment(const CVI_MODEL_HANDLE *seg, void *residual, void *rotary_pos_emb, void *q, void *ks, void *vs) {
        const auto start = std::chrono::high_resolution_clock::now();
        CVI_TENSOR *input_tensors;
        CVI_TENSOR *output_tensors;
        int input_num;
        int output_num;
        CVI_NN_GetInputOutputTensors(*seg, &input_tensors, &input_num, &output_tensors, &output_num);

        CVI_TENSOR *residual_tensor = CVI_NN_GetTensorByName("residual*", input_tensors, input_num);
        CVI_TENSOR *q_tensor = CVI_NN_GetTensorByName("q*", input_tensors, input_num);
        CVI_TENSOR *ks_tensor = CVI_NN_GetTensorByName("ks*", input_tensors, input_num);
        CVI_TENSOR *vs_tensor = CVI_NN_GetTensorByName("vs*", input_tensors, input_num);

        CVI_NN_SetTensorPtr(residual_tensor, residual);
        CVI_NN_SetTensorPtr(q_tensor, q);
        CVI_NN_SetTensorPtr(ks_tensor, ks);
        CVI_NN_SetTensorPtr(vs_tensor, vs);

        if (input_num == 5) {
            CVI_TENSOR *rotary_pos_emb_tensor = CVI_NN_GetTensorByName("rotary_pos_emb*", input_tensors, input_num);
            CVI_NN_SetTensorPtr(rotary_pos_emb_tensor, rotary_pos_emb);
        }

        if (trace_tensor) {
            trace_tensor("ks", create_array_from_tensor<dtype>(ks_tensor));
            trace_tensor("vs", create_array_from_tensor<dtype>(vs_tensor));
        }

        CVI_NN_Forward(*seg, input_tensors, input_num, output_tensors, output_num);

        const auto end = std::chrono::high_resolution_clock::now();
        const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cerr << "Segment forward time: " << duration.count() << " ms" << std::endl;
        return output_tensors;
    }

    void pySetKVCache(const py::array_t<dtype>& past_ks, const py::array_t<dtype>& past_vs) {
        const auto info = past_ks.request();
        // shape (layer, bsz, n_heads, seq, head_dim)
        const int seq_len = info.shape[3];
        std::cerr << "setting kv cache seq_len=" << seq_len << std::endl;
        for (page_index = 0; page_index * BLOCK_LEN < seq_len; page_index++) {
            const auto length = std::min(seq_len - page_index * BLOCK_LEN, BLOCK_LEN);
            for (int layer = 0; layer < NUM_LAYERS; layer++) {
                for (int head = 0; head < NUM_HEADS; head++) {
                    auto &kv_page = kv_pool->pages[page_index][layer][head];
                    const auto k_ptr = past_ks.data() + layer * NUM_HEADS * seq_len * HEAD_DIM + head * seq_len * HEAD_DIM + page_index * BLOCK_LEN * HEAD_DIM;
                    const auto v_ptr = past_vs.data() + layer * NUM_HEADS * seq_len * HEAD_DIM + head * seq_len * HEAD_DIM + page_index * BLOCK_LEN * HEAD_DIM;
                    memcpy(kv_page.k, k_ptr, length * HEAD_DIM * sizeof(dtype));
                    memcpy(kv_page.v, v_ptr, length * HEAD_DIM * sizeof(dtype));
                }
            }
        }
        page_place = seq_len % BLOCK_LEN;
        if (page_place > 0) {
            page_index--;
        }
        std::cerr << "after setting kv cache, page_index=" << page_index << " page_place=" << page_place << std::endl;
    }

    void pyMoveContextPtr(const int ptr) {
        page_place = ptr % BLOCK_LEN;
        page_index = ptr / BLOCK_LEN;
    }

    std::tuple<py::array, py::array, py::array, py::array> pyParseSegmentOutput(CVI_TENSOR *output_tensors) {
        const auto residual_tensor = CVI_NN_GetTensorByName("residual*", output_tensors, 4);
        const auto q_tensor = CVI_NN_GetTensorByName("q*", output_tensors, 4);
        const auto ks_tensor = CVI_NN_GetTensorByName("k*", output_tensors, 4);
        const auto vs_tensor = CVI_NN_GetTensorByName("v*", output_tensors, 4);

        return std::make_tuple(
            create_array_from_tensor<dtype>(residual_tensor),
            create_array_from_tensor<dtype>(q_tensor),
            create_array_from_tensor<dtype>(ks_tensor),
            create_array_from_tensor<dtype>(vs_tensor)
        );
    }

    py::array pyParseTailOutput(CVI_TENSOR *output_tensors) {
        const auto logits_tensor = CVI_NN_GetTensorByName("logits*", output_tensors, 1);

        return create_array_from_tensor<dtype>(logits_tensor);
    }

    std::vector<int> slidingWindow(CVI_TENSOR *q_tensor, const int layer, const int head) const {
        std::vector<int> pages;
        std::cerr << "selectPages layer=" << layer << " pages=";
        for (int i = page_index - 1; i >= 0 && pages.size() < NUM_CONTEXT_PAGES; i--) {
            pages.push_back(i);
            std::cerr << i << " ";
        }
        std::cerr << "pages.size()=" << pages.size() << std::endl;
        return pages;
    }

    std::vector<int> quest(CVI_TENSOR *q_tensor, const int layer, const int head) const {
        std::vector<dtype> scores;
        dtype q[HEAD_DIM]{};
        const auto q_ptr = static_cast<dtype *>(CVI_NN_TensorPtr(q_tensor));
        for (int q_head = ATTN_GROUP_SIZE * head; q_head < ATTN_GROUP_SIZE * (head + 1); q_head++) {
            for (int i = 0; i < HEAD_DIM; i++) {
                q[i] += q_ptr[q_head * HEAD_DIM + i];
            }
        }
        for (int i = 0; i < HEAD_DIM; i++) {
            q[i] /= ATTN_GROUP_SIZE;
        }
        for (int i = 0; i < page_index; i++) {
            auto &digest = kv_pool->page_digests[i][layer][head];
            const auto &valid = kv_pool->page_valid[i][layer][head];
            if (!valid) {
                // make digest
                const auto &page = kv_pool->pages[i][layer][head];
                for (int j = 0; j < HEAD_DIM; j++) {
                    digest.k_max[j] = -1e6;
                    digest.k_min[j] = 1e6;
                }
                for (int j = 0; j < BLOCK_LEN; j++) {
                    for (int k = 0; k < HEAD_DIM; k++) {
                        digest.k_max[k] = std::max(digest.k_max[k], page.k[j][k]);
                        digest.k_min[k] = std::min(digest.k_min[k], page.k[j][k]);
                    }
                }
            }

            dtype score = 0;
            for (int j = 0; j < HEAD_DIM; j++) {
                score += q[i] > 0 ? digest.k_max[j] * q[j] : digest.k_min[j] * q[j];
            }
            scores.push_back(score);
        }
        // top NUM_CONTEXT_PAGES pages
        std::vector<int> pages;
        for (int i = 0; i < page_index; i++) pages.push_back(i);
        std::partial_sort(pages.begin(), pages.begin() + NUM_CONTEXT_PAGES, pages.end(), [&scores](int a, int b) {
            return scores[a] > scores[b];
        });
        pages.resize(NUM_CONTEXT_PAGES);
        // in order
        std::sort(pages.begin(), pages.end());
        return pages;
    }

    std::vector<int> selectPages(CVI_TENSOR *q_tensor, const int layer, const int head) const {
        if (mode == SLIDING_WINDOW) {
            return slidingWindow(q_tensor, layer, head);
        }
        if (mode == QUEST) {
            return quest(q_tensor, layer, head);
        }
    }

    py::array pyCallModel(const py::array_t<long>& input_ids, const py::array_t<dtype>& rotary_pos_emb) {
        const auto rotary_pos_emb_numpy_ptr = (void *) rotary_pos_emb.data();
        auto output_tensors = callHead((void *)input_ids.data(), rotary_pos_emb_numpy_ptr);
        py::object cm;

        for (int layer = 0; layer < NUM_LAYERS; layer++) {
            const auto residual_tensor = CVI_NN_GetTensorByName("residual*", output_tensors, 4);
            const auto q_tensor = CVI_NN_GetTensorByName("q*", output_tensors, 4);
            const auto k_tensor = CVI_NN_GetTensorByName("k*", output_tensors, 4);
            const auto v_tensor = CVI_NN_GetTensorByName("v*", output_tensors, 4);

            if (execute) {
                std::ostringstream oss;
                oss << "seg" << layer;
                cm = execute(oss.str());
                cm.attr("__enter__")();
            }
            if (trace_tensor) {
                trace_tensor("residual", create_array_from_tensor<dtype>(residual_tensor));
                trace_tensor("q", create_array_from_tensor<dtype>(q_tensor));
                trace_tensor("k", create_array_from_tensor<dtype>(k_tensor));
                trace_tensor("v", create_array_from_tensor<dtype>(v_tensor));
            }

            // store the kv pages
            // qkv shape (bsz, n_heads, seq=1, head_dim)
            const auto k_ptr = static_cast<dtype *>(CVI_NN_TensorPtr(k_tensor));
            const auto v_ptr = static_cast<dtype *>(CVI_NN_TensorPtr(v_tensor));
            for (int head = 0; head < NUM_HEADS; head++) {
                const auto kv_page = &kv_pool->pages[page_index][layer][head];
                memcpy(kv_page->k[page_place], k_ptr + head * HEAD_DIM, HEAD_DIM * sizeof(dtype));
                memcpy(kv_page->v[page_place], v_ptr + head * HEAD_DIM, HEAD_DIM * sizeof(dtype));
            }

            // retrieve past kv
            auto ks = new dtype[NUM_HEADS][CONTEXT_LEN - 1][HEAD_DIM];
            auto vs = new dtype[NUM_HEADS][CONTEXT_LEN - 1][HEAD_DIM];
            int place = CONTEXT_LEN - 1;
            // first put current page in
            place -= page_place + 1;
            for (int head = 0; head < NUM_HEADS; head++) {
                const auto kv_page = &kv_pool->pages[page_index][layer][head];
                memcpy(ks[head][place], kv_page->k, (page_place + 1) * HEAD_DIM * sizeof(dtype));
                memcpy(vs[head][place], kv_page->v, (page_place + 1) * HEAD_DIM * sizeof(dtype));
            }
            // then put selected pages in
            for (int head = 0; head < NUM_HEADS; head++) {
                auto pages = selectPages(q_tensor, layer, head);
                for (const auto page : pages) {
                    place -= BLOCK_LEN;
                    int len = BLOCK_LEN;
                    int offset = 0;
                    if (place < 0) {
                        len += place;
                        offset = -place;
                        place = 0;
                    }
                    const auto kv_page = &kv_pool->pages[page][layer][head];
                    memcpy(ks[head][place], kv_page->k[offset], len * HEAD_DIM * sizeof(dtype));
                    memcpy(vs[head][place], kv_page->v[offset], len * HEAD_DIM * sizeof(dtype));
                }
            }

            const auto seg = layer < NUM_LAYERS - 1 ? &segments[layer]: &tail;
            output_tensors = callSegment(seg, CVI_NN_TensorPtr(residual_tensor), rotary_pos_emb_numpy_ptr, CVI_NN_TensorPtr(q_tensor), ks, vs);
            delete[] ks;
            delete[] vs;
            if (execute) {
                cm.attr("__exit__")(py::none(), py::none(), py::none());
            }
        }

        page_place++;
        if (page_place == BLOCK_LEN) {
            page_place = 0;
            page_index++;
        }

        std::cerr << "page_index=" << page_index << " page_place=" << page_place << std::endl;

        return pyParseTailOutput(output_tensors);
    }

    std::tuple<py::array, py::array, py::array, py::array> pyCallHead(const py::array_t<long>& input_ids, const py::array_t<dtype>& rotary_pos_emb) {
        const auto out = callHead((void *)input_ids.data(), (void *)rotary_pos_emb.data());
        return pyParseSegmentOutput(out);
    }

    std::tuple<py::array, py::array, py::array, py::array> pyCallSegment(
        const int layer,
        const py::array_t<dtype>& residual,
        const py::array_t<dtype>& rotary_pos_emb,
        const py::array_t<dtype>& q,
        const py::array_t<dtype>& ks,
        const py::array_t<dtype>& vs
    ) {
        auto out = callSegment(&segments[layer], (void *)residual.data(), (void *)rotary_pos_emb.data(), (void *)q.data(), (void *)ks.data(), (void *)vs.data());
        return pyParseSegmentOutput(out);
    }

    py::array pyCallTail(
        const py::array_t<dtype>& residual,
        const py::array_t<dtype>& q,
        const py::array_t<dtype>& ks,
        const py::array_t<dtype>& vs
    ) {
        auto out = callSegment(&tail, (void *)residual.data(), nullptr, (void *)q.data(),(void *) ks.data(), (void *)vs.data());
        return pyParseTailOutput(out);
    }
};

PYBIND11_MODULE(cvi_klm, m) {
    m.doc() = "CVI KLM module";
    py::class_<KLM>(m, "KLM")
        .def(py::init<const std::string &>())
        .def("call_head", &KLM::pyCallHead)
        .def("call_segment", &KLM::pyCallSegment)
        .def("call_tail", &KLM::pyCallTail)
        .def("call_model", &KLM::pyCallModel)
        .def("set_kv_cache", &KLM::pySetKVCache)
        .def("move_context_ptr", &KLM::pyMoveContextPtr)
        .def("trace", &KLM::pyTraceTensor)
    ;
    py::enum_<KVMode>(m, "KVMode")
        .value("SLIDING_WINDOW", KVMode::SLIDING_WINDOW)
        .value("QUEST", KVMode::QUEST)
        .export_values();
}
