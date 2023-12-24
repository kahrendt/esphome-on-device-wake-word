#include "on_device_wake_word.h"

#include "esphome/core/hal.h"
#include "esphome/core/helpers.h"
#include "esphome/core/log.h"

#include <ringbuf.h>

#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

// #include "audio_preprocessor_int8_model_data.h"
#include "audio_preprocessor_float32_model_data.h"
#include "model.h"

#include <cmath>

namespace esphome {
namespace voice_assistant {

static int8_t convert_float_to_int8(float input, float scale, int zero_point) {
  float scaled = input/scale;
  float zeroed = scaled + zero_point;

  if (zeroed < zero_point) {
    return static_cast<int8_t> (zero_point);
  }
  else if (zeroed > 127) {
    return static_cast<int8_t> (127);
  }
  else {
    return static_cast<int8_t>(round(zeroed));
  }
}

bool OnDeviceWakeWord::intialize_models() {
  ExternalRAMAllocator<uint8_t> arena_allocator(ExternalRAMAllocator<uint8_t>::ALLOW_FAILURE);
  ExternalRAMAllocator<float> spectrogram_allocator(ExternalRAMAllocator<float>::ALLOW_FAILURE);
  ExternalRAMAllocator<int16_t> audio_samples_allocator(ExternalRAMAllocator<int16_t>::ALLOW_FAILURE);

  this->streaming_tensor_arena_ = arena_allocator.allocate(STREAMING_MODEL_ARENA_SIZE);
  if (this->streaming_tensor_arena_ == nullptr) {
    ESP_LOGE(TAG_LOCAL, "Could not allocate the streaming model's tensor arena.");
    return false;
  }

  this->streaming_var_arena_ = arena_allocator.allocate(STREAMING_MODEL_VARIABLE_ARENA_SIZE);
  if (this->streaming_var_arena_ == nullptr) {
    ESP_LOGE(TAG_LOCAL, "Could not allocate the streaming model variable's tensor arena.");
    return false;
  }

  this->nonstreaming_tensor_arena_ = arena_allocator.allocate(NONSTREAMING_MODEL_ARENA_SIZE);
  if (this->nonstreaming_tensor_arena_ == nullptr) {
    ESP_LOGE(TAG_LOCAL, "Could not allocate the nonstreaming model's tensor arena.");
    return false;
  }

  this->preprocessor_tensor_arena_ = arena_allocator.allocate(PREPROCESSOR_ARENA_SIZE);
  if (this->preprocessor_tensor_arena_ == nullptr) {
    ESP_LOGE(TAG_LOCAL, "Could not allocate the audio preprocessor model's tensor arena.");
    return false;
  }

  this->spectrogram_ = spectrogram_allocator.allocate(SPECTROGRAM_TOTAL_PIXELS);
  if (this->spectrogram_ == nullptr) {
    ESP_LOGE(TAG_LOCAL, "Could not allocate the audio features buffer.");
    return false;
  }

  this->preprocessor_audio_buffer_ = audio_samples_allocator.allocate(MAX_AUDIO_SAMPLE_SIZE * 32);
  if (this->preprocessor_audio_buffer_ == nullptr) {
    ESP_LOGE(TAG_LOCAL, "Could not allocate the audio preprocessor's buffer.");
    return false;
  }

  this->preprocessor_stride_buffer_ = audio_samples_allocator.allocate(HISTORY_SAMPLES_TO_KEEP);
  if (this->preprocessor_stride_buffer_ == nullptr) {
    ESP_LOGE(TAG_LOCAL, "Could not allocate the audio preprocessor's stride buffer.");
    return false;
  }

  this->preprocessor_model_ = tflite::GetModel(g_audio_preprocessor_float32_tflite);
  if (this->preprocessor_model_->version() != TFLITE_SCHEMA_VERSION) {
    ESP_LOGE(TAG_LOCAL, "Wake word's audio preprocessor model's schema is not supported");
    return false;
  }

  this->streaming_model_ = tflite::GetModel(streaming_model);
  if (this->streaming_model_->version() != TFLITE_SCHEMA_VERSION) {
    ESP_LOGE(TAG_LOCAL, "Wake word's streaming model's schema is not supported");
    return false;
  }
  this->nonstreaming_model_ = tflite::GetModel(nonstreaming_model);
  if (this->nonstreaming_model_->version() != TFLITE_SCHEMA_VERSION) {
    ESP_LOGE(TAG_LOCAL, "Wake word's nonstreaming model's schema is not supported");
    return false;
  }

  static tflite::MicroMutableOpResolver<18> preprocessor_op_resolver;
  static tflite::MicroMutableOpResolver<9> nonstreaming_op_resolver;
  static tflite::MicroMutableOpResolver<16> streaming_op_resolver;

  if (!this->register_preprocessor_ops_(preprocessor_op_resolver))
    return false;
  if (!this->register_nonstreaming_ops_(nonstreaming_op_resolver))
    return false;
  if (!this->register_streaming_ops_(streaming_op_resolver))
    return false;

  tflite::MicroAllocator *ma =
      tflite::MicroAllocator::Create(this->streaming_var_arena_, STREAMING_MODEL_VARIABLE_ARENA_SIZE);
  this->mrv_ = tflite::MicroResourceVariables::Create(ma, 8);

  static tflite::MicroInterpreter static_preprocessor_interpreter(
      this->preprocessor_model_, preprocessor_op_resolver, this->preprocessor_tensor_arena_, PREPROCESSOR_ARENA_SIZE);

  static tflite::MicroInterpreter static_streaming_interpreter(this->streaming_model_, streaming_op_resolver,
                                                               this->streaming_tensor_arena_,
                                                               STREAMING_MODEL_ARENA_SIZE, this->mrv_);

  static tflite::MicroInterpreter static_nonstreaming_interpreter(this->nonstreaming_model_, nonstreaming_op_resolver,
                                                                  this->nonstreaming_tensor_arena_,
                                                                  NONSTREAMING_MODEL_ARENA_SIZE);

  this->preprocessor_interperter_ = &static_preprocessor_interpreter;
  this->streaming_interpreter_ = &static_streaming_interpreter;
  this->nonstreaming_interpreter_ = &static_nonstreaming_interpreter;

  // Allocate tensors for each models.
  if (this->preprocessor_interperter_->AllocateTensors() != kTfLiteOk) {
    ESP_LOGE(TAG_LOCAL, "Failed to allocate tensors for the audio preprocessor");
    return false;
  }
  if (this->streaming_interpreter_->AllocateTensors() != kTfLiteOk) {
    ESP_LOGE(TAG_LOCAL, "Failed to allocate tensors for the streaming model");
    return false;
  }
  if (this->nonstreaming_interpreter_->AllocateTensors() != kTfLiteOk) {
    ESP_LOGE(TAG_LOCAL, "Failed to allocate tensors for the nonstreaming model");
    return false;
  }

  // Clear the spectrogram
  for (int n = 0; n < SPECTROGRAM_TOTAL_PIXELS; ++n) {
    // this->spectrogram_[n] = 0;
    this->spectrogram_[n] = 0.0;
  }

  for (int n = 0; n < 5; ++n) {
    this->last_5_probabilities_[n] = 0;
    this->recent_streaming_probabilities_[n] = 0.0;
  }

  return true;
}

bool OnDeviceWakeWord::update_spectrogram_(ringbuf_handle_t &ring_buffer) {
  if (!this->slice_available_(ring_buffer)) {
    return false;
  }

  // Shift over the all spectrogram feature slices by one
  for (int dest_slice = 0; dest_slice < PREPROCESSOR_FEATURE_COUNT - 1; ++dest_slice) {
    float *dest_slice_data = this->spectrogram_ + (dest_slice * PREPROCESSOR_FEATURE_SIZE);

    const int src_slice = dest_slice + 1; // Next slice
    const float *src_slice_data = this->spectrogram_ + (src_slice * PREPROCESSOR_FEATURE_SIZE);

    memcpy((void *) (dest_slice_data),
        (const void *) (src_slice_data), PREPROCESSOR_FEATURE_SIZE*sizeof(float));
  }

  // Retrieve strided audio samples
  int16_t *audio_samples = nullptr;
  if (!this->stride_audio_samples_(&audio_samples, ring_buffer)) {
    return false;
  }

  // Pointer to the last feature slice in the spectrogram
  float *new_slice_data = this->spectrogram_ + ((PREPROCESSOR_FEATURE_COUNT-1) * PREPROCESSOR_FEATURE_SIZE);

  // Compute the features for the newest audio samples and store them at the end of the spectrogram
  if (!this->generate_single_feature_(audio_samples, SAMPLE_DURATION_COUNT, new_slice_data)) {
    return false;
  }

  return true;
}

float OnDeviceWakeWord::perform_streaming_inference_() {
  TfLiteTensor *input = this->streaming_interpreter_->input(0);

  float input_scale = input->params.scale;
  int input_zero_point = input->params.zero_point;

  float *new_slice_data = this->spectrogram_ + ((PREPROCESSOR_FEATURE_COUNT-1) * PREPROCESSOR_FEATURE_SIZE);

  // Copy the newest slice's features as input into the streaming model after quantizing them
  for (int i = 0; i < PREPROCESSOR_FEATURE_SIZE; ++i) {
    int8_t converted_value = convert_float_to_int8(new_slice_data[i], input_scale, input_zero_point);
    input->data.int8[i] = converted_value;
  }

  uint32_t prior_invoke = millis();

  // Run the streaming model on only the newest slice's features
  TfLiteStatus invoke_status = this->streaming_interpreter_->Invoke();
  if (invoke_status != kTfLiteOk) {
    ESP_LOGW(TAG_LOCAL, "Streaming Interpreter Invoke failed");
    return false;
  }

  ESP_LOGV(TAG_LOCAL, "Streaming Inference Latency=%u ms", (millis() - prior_invoke));

  TfLiteTensor *output = this->streaming_interpreter_->output(0);

  return static_cast<float>(output->data.uint8[0])/255.0;
}

float OnDeviceWakeWord::perform_nonstreaming_inference_() {
  // Run the nonstreaming model on the entire spectrogram input and make sure it succeeds.
  TfLiteTensor *input_tensor = this->nonstreaming_interpreter_->input(0);

  size_t bytes_to_copy = input_tensor->bytes;

  memcpy((void *) (tflite::GetTensorData<float>(input_tensor)),
        (const void *) (this->spectrogram_), bytes_to_copy);

  uint32_t prior_invoke = millis();

  TfLiteStatus invoke_status = this->nonstreaming_interpreter_->Invoke();
  if (invoke_status != kTfLiteOk) {
    ESP_LOGD(TAG_LOCAL, "Nonstreaming model invoke failed");
    return false;
  }

  ESP_LOGI(TAG_LOCAL, "Nonstreaming inference latency=%u ms", (millis() - prior_invoke));

  TfLiteTensor *output = this->nonstreaming_interpreter_->output(0);

  return output->data.f[0];
}

bool OnDeviceWakeWord::detect_wakeword(ringbuf_handle_t &ring_buffer) {
  if (!this->update_spectrogram_(ring_buffer)) {
    return false;
  }

  float streaming_prob = this->perform_streaming_inference_();

  this->recent_streaming_probabilities_[this->last_5_index_] = streaming_prob;
  ++this->last_5_index_;
  if (this->last_5_index_ == 5)
    this->last_5_index_ = 0;

  float sum = 0.0;
  for (int i = 0; i < 5; ++i) {
    sum += this->recent_streaming_probabilities_[i];
  }

  float sliding_window_average = sum/5.0;

  if (sliding_window_average > 0.5) {
        ESP_LOGD(TAG_LOCAL, "streaming rolling wake word average=%.3f", sliding_window_average);
  }


  this->ignore_windows_ = std::min(this->ignore_windows_+1, 0);
  if (this->ignore_windows_ < 0) {
    return false;
  }

  // if (streaming_prob > STREAMING_MODEL_PROBABILITY_CUTOFF) {
  //   ++this->successive_wake_words_;
  // } else {
  //   this->successive_wake_words_ = std::max(this->successive_wake_words_-1,0);
  // }

  // if (this->successive_wake_words_ < 2) {
  //   return false;
  // }

  if (sliding_window_average > STREAMING_MODEL_PROBABILITY_CUTOFF) {
    this->ignore_windows_ = -PREPROCESSOR_FEATURE_COUNT*3/4;
    this->mrv_->ResetAll();
    for (int n = 0; n < 5; ++n) {
      this->last_5_probabilities_[n] = 0;
      this->recent_streaming_probabilities_[n] = 0.0;
    }
    return true;
  }

  // If the sliding window average is high enough, start slices countdown before nonstream execution
  // if ((sliding_window_average > STREAMING_MODEL_PROBABILITY_CUTOFF) || (this->countdown_to_nonstreaming_ != 5)){
  //   this->countdown_to_nonstreaming_ = std::max(this->countdown_to_nonstreaming_-1, 0);
  // }

  // Do not continue if countdown hasn't reached 0
  if (this->countdown_to_nonstreaming_ > 0) {
    return false;
  }

  float nonstream_prob = this->perform_nonstreaming_inference_();
  ESP_LOGD(TAG_LOCAL, "Nonstreaming wake word probability = %.3f", nonstream_prob);

  this->successive_wake_words_ = 0;

  // Reset countdown
  this->countdown_to_nonstreaming_ = 5;

  // Do not run nonstreaming model for several slices
  this->ignore_windows_ = -PREPROCESSOR_FEATURE_COUNT*3/4;

  return nonstream_prob > NONSTREAMING_MODEL_PROBABILITY_CUTOFF;
}


bool OnDeviceWakeWord::slice_available_(ringbuf_handle_t &ring_buffer) {
  // int slices_needed = rb_bytes_filled(ring_buffer) / (NEW_SAMPLES_TO_GET * sizeof(int16_t));
  if (rb_bytes_filled(ring_buffer) > NEW_SAMPLES_TO_GET*sizeof(int16_t)) {
    return true;
  }
  return false;
}

bool OnDeviceWakeWord::stride_audio_samples_(int16_t **audio_samples, ringbuf_handle_t &ring_buffer) {
  // Copy 320 bytes (160 samples over 10 ms) into preprocessor_audio_buffer_ from history in
  // preprocessor_stride_buffer_
  memcpy((void *) (this->preprocessor_audio_buffer_), (void *) (this->preprocessor_stride_buffer_),
         HISTORY_SAMPLES_TO_KEEP * sizeof(int16_t));

  if (rb_bytes_filled(ring_buffer) < NEW_SAMPLES_TO_GET * sizeof(int16_t)) {
    ESP_LOGD(TAG_LOCAL, "Audio Buffer not full enough");
    return false;
  }

  // Copy 640 bytes (320 samples over 20 ms) from the ring buffer
  // The first 320 bytes (160 samples over 10 ms) will be from history
  int bytes_read = rb_read(ring_buffer, ((char *) (this->preprocessor_audio_buffer_ + HISTORY_SAMPLES_TO_KEEP)),
                           NEW_SAMPLES_TO_GET * sizeof(int16_t), pdMS_TO_TICKS(200));

  if (bytes_read < 0) {
    ESP_LOGE(TAG_LOCAL, "Could not read data from Ring Buffer");
  } else if (bytes_read < NEW_SAMPLES_TO_GET * sizeof(int16_t)) {
    ESP_LOGD(TAG_LOCAL, "Partial Read of Data by Model");
    ESP_LOGD(TAG_LOCAL, "Could only read %d bytes when required %d bytes ", bytes_read,
             (int) (NEW_SAMPLES_TO_GET * sizeof(int16_t)));
    return false;
  }

  // Copy the last 320 bytes (160 samples over 10 ms) from the audio buffer into history stride buffer for the next
  // iteration
  memcpy((void *) (this->preprocessor_stride_buffer_), (void *) (this->preprocessor_audio_buffer_ + NEW_SAMPLES_TO_GET),
         HISTORY_SAMPLES_TO_KEEP * sizeof(int16_t));

  *audio_samples = this->preprocessor_audio_buffer_;
  return true;
}

bool OnDeviceWakeWord::generate_single_feature_(const int16_t *audio_data, const int audio_data_size,
                                                     float feature_output[PREPROCESSOR_FEATURE_SIZE]) {
  TfLiteTensor *input = this->preprocessor_interperter_->input(0);
  TfLiteTensor *output = this->preprocessor_interperter_->output(0);
  std::copy_n(audio_data, audio_data_size, tflite::GetTensorData<int16_t>(input));

  if (this->preprocessor_interperter_->Invoke() != kTfLiteOk) {
    ESP_LOGE(TAG_LOCAL, "Failed to preprocess audio for local wake word.");
    return false;
  }

  // std::memcpy(feature_output, tflite::GetTensorData<int8_t>(output), PREPROCESSOR_FEATURE_SIZE * sizeof(int8_t));
  std::memcpy(feature_output, tflite::GetTensorData<float>(output), PREPROCESSOR_FEATURE_SIZE * sizeof(float));

  return true;
}

bool OnDeviceWakeWord::register_preprocessor_ops_(tflite::MicroMutableOpResolver<18> &op_resolver) {
  if (op_resolver.AddReshape() != kTfLiteOk)
    return false;
  if (op_resolver.AddCast() != kTfLiteOk)
    return false;
  if (op_resolver.AddStridedSlice() != kTfLiteOk)
    return false;
  if (op_resolver.AddConcatenation() != kTfLiteOk)
    return false;
  if (op_resolver.AddMul() != kTfLiteOk)
    return false;
  if (op_resolver.AddAdd() != kTfLiteOk)
    return false;
  if (op_resolver.AddDiv() != kTfLiteOk)
    return false;
  if (op_resolver.AddMinimum() != kTfLiteOk)
    return false;
  if (op_resolver.AddMaximum() != kTfLiteOk)
    return false;
  if (op_resolver.AddWindow() != kTfLiteOk)
    return false;
  if (op_resolver.AddFftAutoScale() != kTfLiteOk)
    return false;
  if (op_resolver.AddRfft() != kTfLiteOk)
    return false;
  if (op_resolver.AddEnergy() != kTfLiteOk)
    return false;
  if (op_resolver.AddFilterBank() != kTfLiteOk)
    return false;
  if (op_resolver.AddFilterBankSquareRoot() != kTfLiteOk)
    return false;
  if (op_resolver.AddFilterBankSpectralSubtraction() != kTfLiteOk)
    return false;
  if (op_resolver.AddPCAN() != kTfLiteOk)
    return false;
  if (op_resolver.AddFilterBankLog() != kTfLiteOk)
    return false;

  return true;
}

bool OnDeviceWakeWord::register_streaming_ops_(tflite::MicroMutableOpResolver<16> &op_resolver) {
  if (op_resolver.AddCallOnce() != kTfLiteOk)
    return false;
  if (op_resolver.AddVarHandle() != kTfLiteOk)
    return false;
  if (op_resolver.AddReshape() != kTfLiteOk)
    return false;
  if (op_resolver.AddPad() != kTfLiteOk)
    return false;
  if (op_resolver.AddReadVariable() != kTfLiteOk)
    return false;
  if (op_resolver.AddStridedSlice() != kTfLiteOk)
    return false;
  if (op_resolver.AddConcatenation() != kTfLiteOk)
    return false;
  if (op_resolver.AddAssignVariable() != kTfLiteOk)
    return false;
  if (op_resolver.AddConv2D() != kTfLiteOk)
    return false;
  if (op_resolver.AddMul() != kTfLiteOk)
    return false;
  if (op_resolver.AddAdd() != kTfLiteOk)
    return false;
  if (op_resolver.AddDepthwiseConv2D() != kTfLiteOk)
    return false;
  if (op_resolver.AddAveragePool2D() != kTfLiteOk)
    return false;
  if (op_resolver.AddFullyConnected() != kTfLiteOk)
    return false;
  if (op_resolver.AddSoftmax() != kTfLiteOk)
    return false;
  if (op_resolver.AddQuantize() != kTfLiteOk)
    return false;

  return true;
}

bool OnDeviceWakeWord::register_nonstreaming_ops_(tflite::MicroMutableOpResolver<9> &op_resolver) {
  if (op_resolver.AddReshape())
    return false;
  if (op_resolver.AddPad())
    return false;
  if (op_resolver.AddConv2D())
    return false;
  if (op_resolver.AddDepthwiseConv2D())
    return false;
  if (op_resolver.AddMul())
    return false;
  if (op_resolver.AddAdd())
    return false;
  if (op_resolver.AddMean())
    return false;
  if (op_resolver.AddLogistic())
    return false;
  if (op_resolver.AddSoftmax())
    return false;

  return true;
}



// void OnDeviceWakeWord::copy_streaming_external_variables_() {
//   const size_t external_variables_count = this->streaming_interpreter_->inputs_size();

//   for (int i = 1; i < external_variables_count; ++i) {

//     TfLiteTensor *input_tensor = this->streaming_interpreter_->input(i);
//     TfLiteTensor *output_tensor = this->streaming_interpreter_->output(i);

//     size_t bytes_to_copy = output_tensor->bytes;

//     // if (i==3) {
//       // memcpy((void *) (tflite::GetTensorData<int8_t>(input_tensor)),
//       //      (const void *) (tflite::GetTensorData<int8_t>(output_tensor)), bytes_to_copy/2);
//       ESP_LOGI(TAG_LOCAL, "external variable %d [0]=%d", i, input_tensor->data.int8[0]);
//       // continue;
//     // }

//     // size_t elements = 1;

//     // for (int j = 0; j < input_tensor->dims->size; ++j) {
//     //   elements *= input_tensor->dims->data[j];
//     // }

//     // for (int j = 0; j < elements; ++j) {
//     //   // tflite::GetTensorData<int8_t>(input_tensor)[j] = input_tensor->params.zero_point;
//     //   input_tensor->data.int8[j] = output_tensor->data.int8[j];
//     //   // input_tensor->data.f[j] = 0.0f;
//     // }


//     // memcpy((void *) (input_tensor->data.data),
//     //        (const void *) (output_tensor->data.data), bytes_to_copy);

//   }
// }

// void OnDeviceWakeWord::clear_streaming_external_variables_() {
//   const size_t external_variables_count = this->streaming_interpreter_->inputs_size();
//   for (int i = 0; i < external_variables_count; ++i) {
//     TfLiteTensor *input_tensor = this->streaming_interpreter_->input(i);

//     size_t elements = 1;

//     for (int j = 0; j < input_tensor->dims->size; ++j) {
//       elements *= input_tensor->dims->data[j];
//     }

//     for (int j = 0; j < elements; ++j) {
//       // tflite::GetTensorData<int8_t>(input_tensor)[j] = input_tensor->params.zero_point;
//       input_tensor->data.int8[j] = 0;//input_tensor->params.zero_point;
//       // input_tensor->data.f[j] = 0.0f;
//     }
//   }
// }

}  // namespace voice_assistant
}  // namespace esphome
