// Copyright 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "triton/backend/backend_common.h"
#include "triton/backend/backend_input_collector.h"
#include "triton/backend/backend_model.h"
#include "triton/backend/backend_model_instance.h"
#include "triton/backend/backend_output_responder.h"
#include "triton/core/tritonbackend.h"
#include <iostream>
#include <numeric>
#include <memory>
#include <opencv2/opencv.hpp>

using std::cout;
using std::endl;


/////////////
//
// Coin: ModelEndPoint
//
// Wraps model end point information

struct InModelEndPoint {
  // Seems tis will take care of input buffer memory, so we just 
  // use the pointer without caring about memory release
  const char *buffer; 
  size_t buffer_byte_size;
  TRITONSERVER_MemoryType buffer_memory_type;
  int64_t buffer_memory_type_id;
};

struct OutModelEndPoint {
  // It is up to user to allocate output buffer memory, therefore
  // we should take care of memory release with shared_ptr
  std::shared_ptr<const char> buffer;
  size_t buffer_byte_size;
  TRITONSERVER_MemoryType buffer_memory_type;
  int64_t buffer_memory_type_id;
  std::vector<int64_t> shape;
};


//// call back function
// Input is raw bytes, which will be decoded into image tensors 
// It is not allowed to batch raw bytes inputs together to decode and preprocess
// Therefore we should not use the field of 'dynamic_batching' in the config.pbtxt
void callback_func(std::vector<InModelEndPoint>& inputs, 
		std::vector<OutModelEndPoint>& outputs) {

  // decode image and resize
  cv::Mat buffer = cv::Mat(cv::Size(1, inputs[0].buffer_byte_size), 
			  CV_8UC1, const_cast<char*>(inputs[0].buffer));
  cv::Mat im = cv::imdecode(buffer, cv::IMREAD_COLOR);
  cout << "image size: " << im.size() << endl;
  int ndims = outputs[0].shape.size();
  int H = outputs[0].shape[ndims - 2]; // last two dims as hw
  int W = outputs[0].shape[ndims - 1];
  cv::Mat imresized = im;
  if (im.rows != H or im.cols != W) {
    cv::resize(im, imresized, cv::Size(W, H), 0., 0., cv::INTER_CUBIC);
    cout << "resize image into: " << imresized.size() << endl;
  }

  // obtain mean/std
  std::vector<float> channel_mean(3), channel_std(3);
  for (int i{0}; i < 3; ++i) {
    channel_mean[i] = reinterpret_cast<float*>(
		    const_cast<char*>(inputs[1].buffer))[i];
    channel_std[i] = 1. / reinterpret_cast<float*>(
		    const_cast<char*>(inputs[2].buffer))[i];
  }
  float scale = 1. / 255;

  // allocate output buffer
  outputs[0].buffer = std::shared_ptr<const char>(
		  new char[outputs[0].buffer_byte_size],
         	  [](const char* p) {cout<<"release output memory\n"; delete[] p;});
  float* obuf = reinterpret_cast<float*>(const_cast<char*>(outputs[0].buffer.get()));

  // divide 255 and then normalize with channel mean/std
  for (int h{0}; h < H; ++h) {
    cv::Vec3b *ptr = imresized.ptr<cv::Vec3b>(h);
    for (int w{0}; w < W; ++w) {
      for (int c{0}; c < 3; ++c) {
        int ind = c * H * W + h * W + w;
	obuf[ind] = (ptr[w][2 - c] * scale - channel_mean[c]) * channel_std[c];
      }
    }
  }
}


namespace triton { namespace backend { namespace recommended {

//
// Backend that demonstrates the TRITONBACKEND API. This backend works
// for any model that has 1 input with any datatype and any shape and
// 1 output with the same shape and datatype as the input. The backend
// supports both batching and non-batching models.
//
// For each batch of requests, the backend returns the input tensor
// value in the output tensor.
//

/////////////

extern "C" {

// Triton calls TRITONBACKEND_Initialize when a backend is loaded into
// Triton to allow the backend to create and initialize any state that
// is intended to be shared across all models and model instances that
// use the backend. The backend should also verify version
// compatibility with Triton in this function.
//
TRITONSERVER_Error*
TRITONBACKEND_Initialize(TRITONBACKEND_Backend* backend)
{
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_BackendName(backend, &cname));
  std::string name(cname);

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("TRITONBACKEND_Initialize: ") + name).c_str());

  // Check the backend API version that Triton supports vs. what this
  // backend was compiled against. Make sure that the Triton major
  // version is the same and the minor version is >= what this backend
  // uses.
  uint32_t api_version_major, api_version_minor;
  RETURN_IF_ERROR(
      TRITONBACKEND_ApiVersion(&api_version_major, &api_version_minor));

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("Triton TRITONBACKEND API version: ") +
       std::to_string(api_version_major) + "." +
       std::to_string(api_version_minor))
          .c_str());
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("'") + name + "' TRITONBACKEND API version: " +
       std::to_string(TRITONBACKEND_API_VERSION_MAJOR) + "." +
       std::to_string(TRITONBACKEND_API_VERSION_MINOR))
          .c_str());

  if ((api_version_major != TRITONBACKEND_API_VERSION_MAJOR) ||
      (api_version_minor < TRITONBACKEND_API_VERSION_MINOR)) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_UNSUPPORTED,
        "triton backend API version does not support this backend");
  }

  // The backend configuration may contain information needed by the
  // backend, such as tritonserver command-line arguments. This
  // backend doesn't use any such configuration but for this example
  // print whatever is available.
  TRITONSERVER_Message* backend_config_message;
  RETURN_IF_ERROR(
      TRITONBACKEND_BackendConfig(backend, &backend_config_message));

  const char* buffer;
  size_t byte_size;
  RETURN_IF_ERROR(TRITONSERVER_MessageSerializeToJson(
      backend_config_message, &buffer, &byte_size));
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("backend configuration:\n") + buffer).c_str());

  // This backend does not require any "global" state but as an
  // example create a string to demonstrate.
  std::string* state = new std::string("backend state");
  RETURN_IF_ERROR(
      TRITONBACKEND_BackendSetState(backend, reinterpret_cast<void*>(state)));

  return nullptr;  // success
}

// Triton calls TRITONBACKEND_Finalize when a backend is no longer
// needed.
//
TRITONSERVER_Error*
TRITONBACKEND_Finalize(TRITONBACKEND_Backend* backend)
{
  // Delete the "global" state associated with the backend.
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_BackendState(backend, &vstate));
  std::string* state = reinterpret_cast<std::string*>(vstate);

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("TRITONBACKEND_Finalize: state is '") + *state + "'")
          .c_str());

  delete state;

  return nullptr;  // success
}

}  // extern "C"

/////////////

//
// ModelState
//
// State associated with a model that is using this backend. An object
// of this class is created and associated with each
// TRITONBACKEND_Model. ModelState is derived from BackendModel class
// provided in the backend utilities that provides many common
// functions.
//
class ModelState : public BackendModel {
 public:
  static TRITONSERVER_Error* Create(
      TRITONBACKEND_Model* triton_model, ModelState** state);
  virtual ~ModelState() = default;

  // Name of the input and output tensor
  const std::string& InputTensorName(int ind) const { return input_names_[ind]; }
  const std::string& OutputTensorName(int ind) const { return output_names_[ind]; }

  // Datatype of the input and output tensor
  /// TRITONSERVER_DataType TensorDataType() const { return datatype_; }
  TRITONSERVER_DataType InputTensorDataType(int ind) const { return inp_dtypes_[ind]; }
  TRITONSERVER_DataType OutputTensorDataType(int ind) const { return outp_dtypes_[ind]; }

  // Shape of the input and output tensor as given in the model
  // configuration file. This shape will not include the batch
  // dimension (if the model has one).
  const std::vector<int64_t>& InputTensorNonBatchShape(int ind) const { return inp_nb_shapes_[ind]; }
  const std::vector<int64_t>& OutputTensorNonBatchShape(int ind) const { return outp_nb_shapes_[ind]; }

  // Shape of the input and output tensor, including the batch
  // dimension (if the model has one). This method cannot be called
  // until the model is completely loaded and initialized, including
  // all instances of the model. In practice, this means that backend
  // should only call it in TRITONBACKEND_ModelInstanceExecute.
  TRITONSERVER_Error* InputTensorShape(std::vector<int64_t>& shape, int ind);
  TRITONSERVER_Error* OutputTensorShape(std::vector<int64_t>& shape, int ind);

  // Number of inputs and outputs
  int NumOfInputs() { return n_inps; }
  int NumOfOutputs() { return n_outps; }

  // Validate that this model is supported by this backend.
  TRITONSERVER_Error* ValidateModelConfig();

 private:
  ModelState(TRITONBACKEND_Model* triton_model);

  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;

  std::vector<TRITONSERVER_DataType> inp_dtypes_;
  std::vector<TRITONSERVER_DataType> outp_dtypes_;

  std::vector<bool> inp_shape_initialized_;
  std::vector<bool> outp_shape_initialized_;
  std::vector<std::vector<int64_t>> inp_nb_shapes_; // no-batch shape
  std::vector<std::vector<int64_t>> inp_shapes_;
  std::vector<std::vector<int64_t>> outp_nb_shapes_; // no-batch shape
  std::vector<std::vector<int64_t>> outp_shapes_;

  int n_inps;
  int n_outps;
};

ModelState::ModelState(TRITONBACKEND_Model* triton_model)
    : BackendModel(triton_model)
{
  // Validate that the model's configuration matches what is supported
  // by this backend.
  THROW_IF_BACKEND_MODEL_ERROR(ValidateModelConfig());
}

TRITONSERVER_Error*
ModelState::Create(TRITONBACKEND_Model* triton_model, ModelState** state)
{
  try {
    *state = new ModelState(triton_model);
  }
  catch (const BackendModelException& ex) {
    RETURN_ERROR_IF_TRUE(
        ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL,
        std::string("unexpected nullptr in BackendModelException"));
    RETURN_IF_ERROR(ex.err_);
  }

  return nullptr;  // success
}


TRITONSERVER_Error*
ModelState::InputTensorShape(std::vector<int64_t>& shape, int ind)
{
  // This backend supports models that batch along the first dimension
  // and those that don't batch. For non-batch models the output shape
  // will be the shape from the model configuration. For batch models
  // the output shape will be the shape from the model configuration
  // prepended with [ -1 ] to represent the batch dimension. The
  // backend "responder" utility used below will set the appropriate
  // batch dimension value for each response. The shape needs to be
  // initialized lazily because the SupportsFirstDimBatching function
  // cannot be used until the model is completely loaded.
  if (!inp_shape_initialized_[ind]) {
    bool supports_first_dim_batching;
    RETURN_IF_ERROR(SupportsFirstDimBatching(&supports_first_dim_batching));
    if (supports_first_dim_batching) {
      inp_shapes_[ind].push_back(-1);
    }

    inp_shapes_[ind].insert(
		    inp_shapes_[ind].end(), 
		    inp_nb_shapes_[ind].begin(), 
		    inp_nb_shapes_[ind].end());
    inp_shape_initialized_[ind] = true;
  }

  shape = inp_shapes_[ind];

  return nullptr;  // success
}


TRITONSERVER_Error*
ModelState::OutputTensorShape(std::vector<int64_t>& shape, int ind)
{
  // This backend supports models that batch along the first dimension
  // and those that don't batch. For non-batch models the output shape
  // will be the shape from the model configuration. For batch models
  // the output shape will be the shape from the model configuration
  // prepended with [ -1 ] to represent the batch dimension. The
  // backend "responder" utility used below will set the appropriate
  // batch dimension value for each response. The shape needs to be
  // initialized lazily because the SupportsFirstDimBatching function
  // cannot be used until the model is completely loaded.
  if (!outp_shape_initialized_[ind]) {
    bool supports_first_dim_batching;
    RETURN_IF_ERROR(SupportsFirstDimBatching(&supports_first_dim_batching));
    if (supports_first_dim_batching) {
      outp_shapes_[ind].push_back(-1);
    }

    outp_shapes_[ind].insert(
		    outp_shapes_[ind].end(), 
		    outp_nb_shapes_[ind].begin(), 
		    outp_nb_shapes_[ind].end());
    outp_shape_initialized_[ind] = true;
  }

  shape = outp_shapes_[ind];

  return nullptr;  // success
}


TRITONSERVER_Error*
ModelState::ValidateModelConfig()
{
  // If verbose logging is enabled, dump the model's configuration as
  // JSON into the console output.
  if (TRITONSERVER_LogIsEnabled(TRITONSERVER_LOG_VERBOSE)) {
    common::TritonJson::WriteBuffer buffer;
    RETURN_IF_ERROR(ModelConfig().PrettyWrite(&buffer));
    LOG_MESSAGE(
        TRITONSERVER_LOG_VERBOSE,
        (std::string("model configuration:\n") + buffer.Contents()).c_str());
  }

  // ModelConfig is the model configuration as a TritonJson
  // object. Use the TritonJson utilities to parse the JSON and
  // determine if the configuration is supported by this backend.
  common::TritonJson::Value inputs, outputs;
  RETURN_IF_ERROR(ModelConfig().MemberAsArray("input", &inputs));
  RETURN_IF_ERROR(ModelConfig().MemberAsArray("output", &outputs));


  //// input name, data_type and shape from config.pbtxt
  n_inps = inputs.ArraySize(); // num_inputs
  inp_shapes_.resize(n_inps);
  for (int i{0}; i < n_inps; ++i) {

    common::TritonJson::Value input;
    RETURN_IF_ERROR(inputs.IndexAsObject(i, &input));

    // name
    const char* input_name;
    size_t input_name_len;
    RETURN_IF_ERROR(input.MemberAsString("name", &input_name, &input_name_len));
    input_names_.push_back(std::string(input_name));

    // data type
    std::string input_dtype;
    RETURN_IF_ERROR(input.MemberAsString("data_type", &input_dtype));
    inp_dtypes_.push_back(ModelConfigDataTypeToTritonServerDataType(input_dtype));

    // dims and shape
    std::vector<int64_t> input_shape;
    RETURN_IF_ERROR(backend::ParseShape(input, "dims", &input_shape));
    inp_nb_shapes_.push_back(input_shape);
    inp_shape_initialized_.push_back(false);
  
  }

  //// out name, data_type and shape from config.pbtxt
  n_outps = outputs.ArraySize(); // num_outputs
  outp_shapes_.resize(n_outps);
  for (int i{0}; i < n_outps; ++i) {

    common::TritonJson::Value output;
    RETURN_IF_ERROR(outputs.IndexAsObject(i, &output));

    // name
    const char* output_name;
    size_t output_name_len;
    RETURN_IF_ERROR(
        output.MemberAsString("name", &output_name, &output_name_len));
    output_names_.push_back(std::string(output_name));

    // data type
    std::string output_dtype;
    RETURN_IF_ERROR(output.MemberAsString("data_type", &output_dtype));
    outp_dtypes_.push_back(ModelConfigDataTypeToTritonServerDataType(output_dtype));

    // dims and shape
    std::vector<int64_t> output_shape;
    RETURN_IF_ERROR(backend::ParseShape(output, "dims", &output_shape));
    outp_nb_shapes_.push_back(output_shape);
    outp_shape_initialized_.push_back(false);
  }

  return nullptr;  // success
}

extern "C" {

// Triton calls TRITONBACKEND_ModelInitialize when a model is loaded
// to allow the backend to create any state associated with the model,
// and to also examine the model configuration to determine if the
// configuration is suitable for the backend. Any errors reported by
// this function will prevent the model from loading.
//
TRITONSERVER_Error*
TRITONBACKEND_ModelInitialize(TRITONBACKEND_Model* model)
{
  // Create a ModelState object and associate it with the
  // TRITONBACKEND_Model. If anything goes wrong with initialization
  // of the model state then an error is returned and Triton will fail
  // to load the model.
  ModelState* model_state;
  RETURN_IF_ERROR(ModelState::Create(model, &model_state));
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelSetState(model, reinterpret_cast<void*>(model_state)));

  return nullptr;  // success
}

// Triton calls TRITONBACKEND_ModelFinalize when a model is no longer
// needed. The backend should cleanup any state associated with the
// model. This function will not be called until all model instances
// of the model have been finalized.
//
TRITONSERVER_Error*
TRITONBACKEND_ModelFinalize(TRITONBACKEND_Model* model)
{
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vstate));
  ModelState* model_state = reinterpret_cast<ModelState*>(vstate);
  delete model_state;

  return nullptr;  // success
}

}  // extern "C"


/////////////

//
// ModelInstanceState
//
// State associated with a model instance. An object of this class is
// created and associated with each
// TRITONBACKEND_ModelInstance. ModelInstanceState is derived from
// BackendModelInstance class provided in the backend utilities that
// provides many common functions.
//
class ModelInstanceState : public BackendModelInstance {
 public:
  static TRITONSERVER_Error* Create(
      ModelState* model_state,
      TRITONBACKEND_ModelInstance* triton_model_instance,
      ModelInstanceState** state);
  virtual ~ModelInstanceState() = default;

  // Get the state of the model that corresponds to this instance.
  ModelState* StateForModel() const { return model_state_; }

 private:
  ModelInstanceState(
      ModelState* model_state,
      TRITONBACKEND_ModelInstance* triton_model_instance)
      : BackendModelInstance(model_state, triton_model_instance),
        model_state_(model_state)
  {
  }

  ModelState* model_state_;
};

TRITONSERVER_Error*
ModelInstanceState::Create(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance,
    ModelInstanceState** state)
{
  try {
    *state = new ModelInstanceState(model_state, triton_model_instance);
  }
  catch (const BackendModelInstanceException& ex) {
    RETURN_ERROR_IF_TRUE(
        ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL,
        std::string("unexpected nullptr in BackendModelInstanceException"));
    RETURN_IF_ERROR(ex.err_);
  }

  return nullptr;  // success
}

extern "C" {

// Triton calls TRITONBACKEND_ModelInstanceInitialize when a model
// instance is created to allow the backend to initialize any state
// associated with the instance.
//
TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceInitialize(TRITONBACKEND_ModelInstance* instance)
{
  // Get the model state associated with this instance's model.
  TRITONBACKEND_Model* model;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceModel(instance, &model));

  void* vmodelstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vmodelstate));
  ModelState* model_state = reinterpret_cast<ModelState*>(vmodelstate);

  // Create a ModelInstanceState object and associate it with the
  // TRITONBACKEND_ModelInstance.
  ModelInstanceState* instance_state;
  RETURN_IF_ERROR(
      ModelInstanceState::Create(model_state, instance, &instance_state));
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceSetState(
      instance, reinterpret_cast<void*>(instance_state)));

  return nullptr;  // success
}

// Triton calls TRITONBACKEND_ModelInstanceFinalize when a model
// instance is no longer needed. The backend should cleanup any state
// associated with the model instance.
//
TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceFinalize(TRITONBACKEND_ModelInstance* instance)
{
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(instance, &vstate));
  ModelInstanceState* instance_state =
      reinterpret_cast<ModelInstanceState*>(vstate);
  delete instance_state;

  return nullptr;  // success
}

}  // extern "C"

/////////////

extern "C" {

// When Triton calls TRITONBACKEND_ModelInstanceExecute it is required
// that a backend create a response for each request in the batch. A
// response may be the output tensors required for that request or may
// be an error that is returned in the response.
//
TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceExecute(
    TRITONBACKEND_ModelInstance* instance, TRITONBACKEND_Request** requests,
    const uint32_t request_count)
{
  // Collect various timestamps during the execution of this batch or
  // requests. These values are reported below before returning from
  // the function.

  uint64_t exec_start_ns = 0;
  SET_TIMESTAMP(exec_start_ns);

  // Triton will not call this function simultaneously for the same
  // 'instance'. But since this backend could be used by multiple
  // instances from multiple models the implementation needs to handle
  // multiple calls to this function at the same time (with different
  // 'instance' objects). Best practice for a high-performance
  // implementation is to avoid introducing mutex/lock and instead use
  // only function-local and model-instance-specific state.
  ModelInstanceState* instance_state;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(
      instance, reinterpret_cast<void**>(&instance_state)));
  ModelState* model_state = instance_state->StateForModel();

  // 'responses' is initialized as a parallel array to 'requests',
  // with one TRITONBACKEND_Response object for each
  // TRITONBACKEND_Request object. If something goes wrong while
  // creating these response objects, the backend simply returns an
  // error from TRITONBACKEND_ModelInstanceExecute, indicating to
  // Triton that this backend did not create or send any responses and
  // so it is up to Triton to create and send an appropriate error
  // response for each request. RETURN_IF_ERROR is one of several
  // useful macros for error handling that can be found in
  // backend_common.h.

  std::vector<TRITONBACKEND_Response*> responses;
  responses.reserve(request_count);
  for (uint32_t r = 0; r < request_count; ++r) {
    TRITONBACKEND_Request* request = requests[r];
    TRITONBACKEND_Response* response;
    RETURN_IF_ERROR(TRITONBACKEND_ResponseNew(&response, request));
    responses.push_back(response);
  }

  // At this point, the backend takes ownership of 'requests', which
  // means that it is responsible for sending a response for every
  // request. From here, even if something goes wrong in processing,
  // the backend must return 'nullptr' from this function to indicate
  // success. Any errors and failures must be communicated via the
  // response objects.
  //
  // To simplify error handling, the backend utilities manage
  // 'responses' in a specific way and it is recommended that backends
  // follow this same pattern. When an error is detected in the
  // processing of a request, an appropriate error response is sent
  // and the corresponding TRITONBACKEND_Response object within
  // 'responses' is set to nullptr to indicate that the
  // request/response has already been handled and no futher processing
  // should be performed for that request. Even if all responses fail,
  // the backend still allows execution to flow to the end of the
  // function so that statistics are correctly reported by the calls
  // to TRITONBACKEND_ModelInstanceReportStatistics and
  // TRITONBACKEND_ModelInstanceReportBatchStatistics.
  // RESPOND_AND_SET_NULL_IF_ERROR, and
  // RESPOND_ALL_AND_SET_NULL_IF_ERROR are macros from
  // backend_common.h that assist in this management of response
  // objects.

  // The backend could iterate over the 'requests' and process each
  // one separately. But for performance reasons it is usually
  // preferred to create batched input tensors that are processed
  // simultaneously. This is especially true for devices like GPUs
  // that are capable of exploiting the large amount parallelism
  // exposed by larger data sets.
  //
  // The backend utilities provide a "collector" to facilitate this
  // batching process. The 'collector's ProcessTensor function will
  // combine a tensor's value from each request in the batch into a
  // single contiguous buffer. The buffer can be provided by the
  // backend or 'collector' can create and manage it. In this backend,
  // there is not a specific buffer into which the batch should be
  // created, so use ProcessTensor arguments that cause collector to
  // manage it.

  BackendInputCollector collector(
      requests, request_count, &responses, model_state->TritonMemoryManager(),
      false /* pinned_enabled */, nullptr /* stream*/);

  // To instruct ProcessTensor to "gather" the entire batch of input
  // tensors into a single contiguous buffer in CPU memory, set the
  // "allowed input types" to be the CPU ones (see tritonserver.h in
  // the triton-inference-server/core repo for allowed memory types).
  std::vector<std::pair<TRITONSERVER_MemoryType, int64_t>> allowed_input_types =
      {{TRITONSERVER_MEMORY_CPU_PINNED, 0}, {TRITONSERVER_MEMORY_CPU, 0}};

  /// Coin: Here we get input buffers and meta info, such that we can process 
  /// them and implement the logic of the backend
  int n_inps = model_state->NumOfInputs();
  std::vector<InModelEndPoint> inputs(n_inps);
  for (int i{0}; i < n_inps; ++i) {
    RESPOND_ALL_AND_SET_NULL_IF_ERROR(
      responses, request_count,
      collector.ProcessTensor(
        model_state->InputTensorName(i).c_str(), nullptr /* existing_buffer */,
        0 /* existing_buffer_byte_size */, allowed_input_types, 
        &inputs[i].buffer,
        &inputs[i].buffer_byte_size, 
        &inputs[i].buffer_memory_type,
        &inputs[i].buffer_memory_type_id));
  }

  int n_outps = model_state->NumOfOutputs();
  std::vector<OutModelEndPoint> outputs(n_outps);
  for (int i{0}; i < n_outps; ++i) {
    outputs[i].buffer_memory_type = TRITONSERVER_MEMORY_CPU;
    outputs[i].buffer_memory_type_id = 0;
    model_state->OutputTensorShape(outputs[i].shape, i);
    int64_t byte_size = TRITONSERVER_DataTypeByteSize(
		    model_state->OutputTensorDataType(i));
    int64_t n_pixels = std::accumulate(
		    outputs[i].shape.begin()+1, outputs[i].shape.end(), 
		    1, std::multiplies<int64_t>());
    outputs[i].buffer_byte_size = n_pixels * byte_size;

    /// print datatype 
    /// cout << TRITONSERVER_DataTypeString(model_state->OutputTensorDataType(i)) << endl;
  }

  //// Coin: here we implement our logic
  callback_func(inputs, outputs);


  // Finalize the collector. If 'true' is returned, 'input_buffer'
  // will not be valid until the backend synchronizes the CUDA
  // stream or event that was used when creating the collector. For
  // this backend, GPU is not supported and so no CUDA sync should
  // be needed; so if 'true' is returned simply log an error.
  const bool need_cuda_input_sync = collector.Finalize();
  if (need_cuda_input_sync) {
    LOG_MESSAGE(
        TRITONSERVER_LOG_ERROR,
        "'recommended' backend: unexpected CUDA sync required by collector");
  }

  // 'input_buffer' contains the batched input tensor. The backend can
  // implement whatever logic is necessary to produce the output
  // tensor. This backend simply logs the input tensor value and then
  // returns the input tensor value in the output tensor so no actual
  // computation is needed.

  uint64_t compute_start_ns = 0;
  SET_TIMESTAMP(compute_start_ns);

  /*
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("model ") + model_state->Name() + ": requests in batch " +
       std::to_string(request_count))
          .c_str());
  std::string tstr;
  IGNORE_ERROR(BufferAsTypedString(
      tstr, input_buffer, input_buffer_byte_size,
      model_state->TensorDataType()));
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("batched " + model_state->InputTensorName() + " value: ") +
       tstr)
          .c_str());
  */

  // const char* output_buffer = input_buffer;
  // TRITONSERVER_MemoryType output_buffer_memory_type = input_buffer_memory_type;
  // int64_t output_buffer_memory_type_id = input_buffer_memory_type_id;

  uint64_t compute_end_ns = 0;
  SET_TIMESTAMP(compute_end_ns);

  bool supports_first_dim_batching;
  RESPOND_ALL_AND_SET_NULL_IF_ERROR(
      responses, request_count,
      model_state->SupportsFirstDimBatching(&supports_first_dim_batching));

  std::vector<std::vector<int64_t>> outp_shapes(n_outps);
  for (int i{0}; i < n_outps; ++i) {
    RESPOND_ALL_AND_SET_NULL_IF_ERROR(
        responses, request_count, model_state->OutputTensorShape(
            outp_shapes[i], i));
  }

  // Because the output tensor values are concatenated into a single
  // contiguous 'output_buffer', the backend must "scatter" them out
  // to the individual response output tensors.  The backend utilities
  // provide a "responder" to facilitate this scattering process.

  // The 'responders's ProcessTensor function will copy the portion of
  // 'output_buffer' corresonding to each request's output into the
  // response for that request.

  BackendOutputResponder responder(
      requests, request_count, &responses, model_state->TritonMemoryManager(),
      supports_first_dim_batching, false /* pinned_enabled */,
      nullptr /* stream*/);

  for (int i{0}; i < n_outps; ++i) {
    const char* output_buffer = outputs[i].buffer.get();
    //const char* output_buffer = new char[16];
    responder.ProcessTensor(
        model_state->OutputTensorName(i).c_str(), model_state->OutputTensorDataType(i),
        outp_shapes[i], output_buffer, outputs[i].buffer_memory_type,
        outputs[i].buffer_memory_type_id);

  }

  // Finalize the responder. If 'true' is returned, the output
  // tensors' data will not be valid until the backend synchronizes
  // the CUDA stream or event that was used when creating the
  // responder. For this backend, GPU is not supported and so no CUDA
  // sync should be needed; so if 'true' is returned simply log an
  // error.
  const bool need_cuda_output_sync = responder.Finalize();
  if (need_cuda_output_sync) {
    LOG_MESSAGE(
        TRITONSERVER_LOG_ERROR,
        "'recommended' backend: unexpected CUDA sync required by responder");
  }

  // Send all the responses that haven't already been sent because of
  // an earlier error.
  for (auto& response : responses) {
    if (response != nullptr) {
      LOG_IF_ERROR(
          TRITONBACKEND_ResponseSend(
              response, TRITONSERVER_RESPONSE_COMPLETE_FINAL, nullptr),
          "failed to send response");
    }
  }

  uint64_t exec_end_ns = 0;
  SET_TIMESTAMP(exec_end_ns);

#ifdef TRITON_ENABLE_STATS
  // For batch statistics need to know the total batch size of the
  // requests. This is not necessarily just the number of requests,
  // because if the model supports batching then any request can be a
  // batched request itself.
  size_t total_batch_size = 0;
  if (!supports_first_dim_batching) {
    total_batch_size = request_count;
  } else {
    for (uint32_t r = 0; r < request_count; ++r) {
      auto& request = requests[r];
      TRITONBACKEND_Input* input = nullptr;
      LOG_IF_ERROR(
          TRITONBACKEND_RequestInputByIndex(request, 0 /* index */, &input),
          "failed getting request input");
      if (input != nullptr) {
        const int64_t* shape = nullptr;
        LOG_IF_ERROR(
            TRITONBACKEND_InputProperties(
                input, nullptr, nullptr, &shape, nullptr, nullptr, nullptr),
            "failed getting input properties");
        if (shape != nullptr) {
          total_batch_size += shape[0];
        }
      }
    }
  }
#else
  (void)exec_start_ns;
  (void)exec_end_ns;
  (void)compute_start_ns;
  (void)compute_end_ns;
#endif  // TRITON_ENABLE_STATS

  // Report statistics for each request, and then release the request.
  for (uint32_t r = 0; r < request_count; ++r) {
    auto& request = requests[r];

#ifdef TRITON_ENABLE_STATS
    LOG_IF_ERROR(
        TRITONBACKEND_ModelInstanceReportStatistics(
            instance_state->TritonModelInstance(), request,
            (responses[r] != nullptr) /* success */, exec_start_ns,
            compute_start_ns, compute_end_ns, exec_end_ns),
        "failed reporting request statistics");
#endif  // TRITON_ENABLE_STATS

    LOG_IF_ERROR(
        TRITONBACKEND_RequestRelease(request, TRITONSERVER_REQUEST_RELEASE_ALL),
        "failed releasing request");
  }

#ifdef TRITON_ENABLE_STATS
  // Report batch statistics.
  LOG_IF_ERROR(
      TRITONBACKEND_ModelInstanceReportBatchStatistics(
          instance_state->TritonModelInstance(), total_batch_size,
          exec_start_ns, compute_start_ns, compute_end_ns, exec_end_ns),
      "failed reporting batch request statistics");
#endif  // TRITON_ENABLE_STATS

  return nullptr;  // success
}

}  // extern "C"

}}}  // namespace triton::backend::recommended
