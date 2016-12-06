#include <cstddef>
#include <iostream>
#include <algorithm>

#include <ctc.h>

#include "detail/cpu_ctc.h"
#ifdef __CUDACC__
    #include "detail/gpu_ctc.h"
#endif


extern "C" {

const char* ctcGetStatusString(ctcStatus_t status) {
    switch (status) {
    case CTC_STATUS_SUCCESS:
        return "no error";
    case CTC_STATUS_MEMOPS_FAILED:
        return "cuda memcpy or memset failed";
    case CTC_STATUS_INVALID_VALUE:
        return "invalid value";
    case CTC_STATUS_EXECUTION_FAILED:
        return "execution failed";

    case CTC_STATUS_UNKNOWN_ERROR:
    default:
        return "unknown error";

    }

}

inline void throw_on_error(ctcStatus_t status, const char* message) {
    if (status != CTC_STATUS_SUCCESS) {
        throw std::runtime_error(message + (", stat = " + 
                                            std::string(ctcGetStatusString(status))));
    }
}

int compute_ctc_gpu(const float* const activations,
                             float* gradients,
                             const int* const flat_labels,
                             const int* const label_lengths,
                             const int* const input_lengths,
                             int alphabet_size,
                             int minibatch,
                             float *costs,
                             cudaStream_t stream,
                             char *ctc_gpu_workspace){

    if (activations == nullptr ||
        flat_labels == nullptr ||
        label_lengths == nullptr ||
        input_lengths == nullptr ||
        costs == nullptr ||
        ctc_gpu_workspace == nullptr ||
        alphabet_size <= 0 ||
        minibatch <= 0)
        return CTC_STATUS_INVALID_VALUE;

#ifdef __CUDACC__
        GpuCTC<float> ctc(alphabet_size, minibatch, ctc_gpu_workspace, stream);

        if (gradients != NULL){
            ctcStatus_t status;
            status = ctc.cost_and_grad(activations, gradients, costs,
                                     flat_labels, label_lengths,
                                     input_lengths);
            return int(status);
        }
        else{
            return ctc.score_forward(activations, costs, flat_labels,
                                     label_lengths, input_lengths);
        }
#else
        std::cerr << "GPU execution requested, but not compiled with GPU support" << std::endl;
        return int(CTC_STATUS_EXECUTION_FAILED);
#endif
}


ctcStatus_t compute_ctc_loss(const float* const activations,
                             float* gradients,
                             const int* const flat_labels,
                             const int* const label_lengths,
                             const int* const input_lengths,
                             int alphabet_size,
                             int minibatch,
                             float *costs,
                             void *workspace,
                             ctcComputeInfo info) {

    if (activations == nullptr ||
        flat_labels == nullptr ||
        label_lengths == nullptr ||
        input_lengths == nullptr ||
        costs == nullptr ||
        workspace == nullptr ||
        alphabet_size <= 0 ||
        minibatch <= 0)
        return CTC_STATUS_INVALID_VALUE;

    if (info.loc == CTC_CPU) {
        CpuCTC<float> ctc(alphabet_size, minibatch, workspace, info.num_threads);

        if (gradients != NULL)
            return ctc.cost_and_grad(activations, gradients,
                                     costs,
                                     flat_labels, label_lengths,
                                     input_lengths);
        else
            return ctc.score_forward(activations, costs, flat_labels,
                                     label_lengths, input_lengths);
    } else if (info.loc == CTC_GPU) {
#ifdef __CUDACC__
        GpuCTC<float> ctc(alphabet_size, minibatch, workspace, info.stream);

        if (gradients != NULL)
            return ctc.cost_and_grad(activations, gradients, costs,
                                     flat_labels, label_lengths,
                                     input_lengths);
        else
            return ctc.score_forward(activations, costs, flat_labels,
                                     label_lengths, input_lengths);
#else
        std::cerr << "GPU execution requested, but not compiled with GPU support" << std::endl;
        return CTC_STATUS_EXECUTION_FAILED;
#endif
    } else {
        return CTC_STATUS_INVALID_VALUE;
    }
}


int get_workspace_size_gpu(int maxL, int maxT, int alphabet_size, int minibatch)
{
    //maxL: max label length
    //maxT: max input length
    if (maxL <= 0 ||
        maxT <= 0 ||
        alphabet_size <= 0 ||
        minibatch <= 0)
        return int(CTC_STATUS_INVALID_VALUE);

    const int S = 2 * maxL + 1;
    int size_bytes = 0;

    // GPU storage
    //nll_forward, nll_backward
    size_bytes += 2 * sizeof(float) * minibatch;

    //repeats
    size_bytes += sizeof(int) * minibatch;

    //label offsets
    size_bytes += sizeof(int) * minibatch;

    //utt_length
    size_bytes += sizeof(int) * minibatch;

    //label lengths
    size_bytes += sizeof(int) * minibatch;

    //labels without blanks - overallocate for now
    size_bytes += sizeof(int) * maxL * minibatch;

    //labels with blanks
    size_bytes += sizeof(int) * S * minibatch;

    //alphas
    size_bytes += sizeof(float) * S * maxT * minibatch;

    //denoms
    size_bytes += sizeof(float) * maxT * minibatch;

    //probs (since we will pass in activations)
    size_bytes += sizeof(float) * alphabet_size * maxT * minibatch;

    return size_bytes;
}


ctcStatus_t get_workspace_size(const int* const label_lengths,
                               const int* const input_lengths,
                               int alphabet_size, int minibatch,
                               ctcComputeInfo info,
                               size_t* size_bytes)
{
    if (label_lengths == nullptr ||
        input_lengths == nullptr ||
        size_bytes == nullptr ||
        alphabet_size <= 0 ||
        minibatch <= 0)
        return CTC_STATUS_INVALID_VALUE;

    // This is the max of all S and T for all examples in the minibatch.
    int maxL = *std::max_element(label_lengths, label_lengths + minibatch);
    int maxT = *std::max_element(input_lengths, input_lengths + minibatch);

    const int S = 2 * maxL + 1;

    *size_bytes = 0;

    if (info.loc == CTC_GPU) {
        // GPU storage
        //nll_forward, nll_backward
        *size_bytes += 2 * sizeof(float) * minibatch;

        //repeats
        *size_bytes += sizeof(int) * minibatch;

        //label offsets
        *size_bytes += sizeof(int) * minibatch;

        //utt_length
        *size_bytes += sizeof(int) * minibatch;

        //label lengths
        *size_bytes += sizeof(int) * minibatch;

        //labels without blanks - overallocate for now
        *size_bytes += sizeof(int) * maxL * minibatch;

        //labels with blanks
        *size_bytes += sizeof(int) * S * minibatch;

        //alphas
        *size_bytes += sizeof(float) * S * maxT * minibatch;

        //denoms
        *size_bytes += sizeof(float) * maxT * minibatch;

        //probs (since we will pass in activations)
        *size_bytes += sizeof(float) * alphabet_size * maxT * minibatch;

    } else {
        //cpu can eventually replace all minibatch with
        //max number of concurrent threads if memory is
        //really tight

        //per minibatch memory
        size_t per_minibatch_bytes = 0;

        //output
        per_minibatch_bytes += sizeof(float) * alphabet_size ;

        //alphas
        per_minibatch_bytes += sizeof(float) * S * maxT;

        //betas
        per_minibatch_bytes += sizeof(float) * S;

        //labels w/blanks, e_inc, s_inc
        per_minibatch_bytes += 3 * sizeof(int) * S;

        *size_bytes = per_minibatch_bytes * minibatch;

        //probs
        *size_bytes += sizeof(float) * alphabet_size * maxT * minibatch;
    }

    return CTC_STATUS_SUCCESS;
}

void cpu_ctc(float* acts, 
             float* grads,
             int* labels, 
             int* label_lengths,
             int* input_lengths,
             int alphabet_size, 
             int minibatch,
             float* cost,
             int num_threads)
{
    ctcComputeInfo info;
    info.loc = CTC_CPU;
    info.num_threads = num_threads;

    size_t cpu_alloc_bytes;
    get_workspace_size(label_lengths, input_lengths,
                       alphabet_size, minibatch, info,
                       &cpu_alloc_bytes);

    void* ctc_cpu_workspace = malloc(cpu_alloc_bytes);

    compute_ctc_loss(acts, grads,
                     labels, label_lengths,
                     input_lengths,
                     alphabet_size,
                     minibatch,
                     cost,
                     ctc_cpu_workspace,
                     info);

    free(ctc_cpu_workspace);
}

}
