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
#include "model.h"
#ifdef USE_INT8_PREPROCESSOR
#include "audio_preprocessor_int8_model_data.h"
#else
#include "audio_preprocessor_float32_model_data.h"
#endif

#include <cmath>

namespace esphome {
namespace voice_assistant {

#ifndef USE_INT8_PREPROCESSOR
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
#endif

bool OnDeviceWakeWord::intialize_models() {
  ExternalRAMAllocator<uint8_t> arena_allocator(ExternalRAMAllocator<uint8_t>::ALLOW_FAILURE);
  #ifdef USE_INT8_PREPROCESSOR
  ExternalRAMAllocator<int8_t> spectrogram_allocator(ExternalRAMAllocator<int8_t>::ALLOW_FAILURE);
  #else
  ExternalRAMAllocator<float> spectrogram_allocator(ExternalRAMAllocator<float>::ALLOW_FAILURE);
  #endif
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

  #ifdef USE_INT8_PREPROCESSOR
  this->preprocessor_model_ = tflite::GetModel(g_audio_preprocessor_int8_tflite);
  #else
  this->preprocessor_model_ = tflite::GetModel(g_audio_preprocessor_float32_tflite);
  #endif
  if (this->preprocessor_model_->version() != TFLITE_SCHEMA_VERSION) {
    ESP_LOGE(TAG_LOCAL, "Wake word's audio preprocessor model's schema is not supported");
    return false;
  }

  this->streaming_model_ = tflite::GetModel(streaming_model);
  if (this->streaming_model_->version() != TFLITE_SCHEMA_VERSION) {
    ESP_LOGE(TAG_LOCAL, "Wake word's streaming model's schema is not supported");
    return false;
  }

  static tflite::MicroMutableOpResolver<18> preprocessor_op_resolver;
  static tflite::MicroMutableOpResolver<12> streaming_op_resolver;

  if (!this->register_preprocessor_ops_(preprocessor_op_resolver))
    return false;
  if (!this->register_streaming_ops_(streaming_op_resolver))
    return false;

  tflite::MicroAllocator *ma =
      tflite::MicroAllocator::Create(this->streaming_var_arena_, STREAMING_MODEL_VARIABLE_ARENA_SIZE);
  this->mrv_ = tflite::MicroResourceVariables::Create(ma, 15);

  static tflite::MicroInterpreter static_preprocessor_interpreter(
      this->preprocessor_model_, preprocessor_op_resolver, this->preprocessor_tensor_arena_, PREPROCESSOR_ARENA_SIZE);

  static tflite::MicroInterpreter static_streaming_interpreter(this->streaming_model_, streaming_op_resolver,
                                                               this->streaming_tensor_arena_,
                                                               STREAMING_MODEL_ARENA_SIZE, this->mrv_);

  this->preprocessor_interperter_ = &static_preprocessor_interpreter;
  this->streaming_interpreter_ = &static_streaming_interpreter;

  // Allocate tensors for each models.
  if (this->preprocessor_interperter_->AllocateTensors() != kTfLiteOk) {
    ESP_LOGE(TAG_LOCAL, "Failed to allocate tensors for the audio preprocessor");
    return false;
  }
  if (this->streaming_interpreter_->AllocateTensors() != kTfLiteOk) {
    ESP_LOGE(TAG_LOCAL, "Failed to allocate tensors for the streaming model");
    return false;
  }

  // Clear the spectrogram
  for (int n = 0; n < SPECTROGRAM_TOTAL_PIXELS; ++n) {
    #ifdef USE_INT8_PREPROCESSOR
    this->spectrogram_[n] = 0;
    #else
    this->spectrogram_[n] = 0.0;
    #endif
  }

  for (int n = 0; n < STREAMING_MODEL_SLIDING_WINDOW_MEAN_LENGTH; ++n) {
    this->recent_streaming_probabilities_[n] = 0.0;
  }

  return true;
}

bool OnDeviceWakeWord::update_spectrogram_(ringbuf_handle_t &ring_buffer) {
  // Note that the streaming model does not need the full spectrogram, only the latest feature
  // This code is maintained for future potential uses; e.g., speaker identification

  if (!this->slice_available_(ring_buffer)) {
    return false;
  }

  // Shift over the all spectrogram feature slices by one
  for (int dest_slice = 0; dest_slice < PREPROCESSOR_FEATURE_COUNT - 1; ++dest_slice) {
    #ifdef USE_INT8_PREPROCESSOR
    int8_t *dest_slice_data = this->spectrogram_ + (dest_slice * PREPROCESSOR_FEATURE_SIZE);
    #else
    float *dest_slice_data = this->spectrogram_ + (dest_slice * PREPROCESSOR_FEATURE_SIZE);
    #endif

    const int src_slice = dest_slice + 1; // Next slice
    #ifdef USE_INT8_PREPROCESSOR
    const int8_t *src_slice_data = this->spectrogram_ + (src_slice * PREPROCESSOR_FEATURE_SIZE);
    #else
    const float *src_slice_data = this->spectrogram_ + (src_slice * PREPROCESSOR_FEATURE_SIZE);
    #endif

    #ifdef USE_INT8_PREPROCESSOR
    memcpy((void *) (dest_slice_data),
        (const void *) (src_slice_data), PREPROCESSOR_FEATURE_SIZE*sizeof(int8_t));
    #else
    memcpy((void *) (dest_slice_data),
        (const void *) (src_slice_data), PREPROCESSOR_FEATURE_SIZE*sizeof(float));
    #endif
  }

  // Retrieve strided audio samples
  int16_t *audio_samples = nullptr;
  if (!this->stride_audio_samples_(&audio_samples, ring_buffer)) {
    return false;
  }

  // Pointer to the last feature slice in the spectrogram
  #ifdef USE_INT8_PREPROCESSOR
  int8_t *new_slice_data = this->spectrogram_ + ((PREPROCESSOR_FEATURE_COUNT-1) * PREPROCESSOR_FEATURE_SIZE);
  #else
  float *new_slice_data = this->spectrogram_ + ((PREPROCESSOR_FEATURE_COUNT-1) * PREPROCESSOR_FEATURE_SIZE);
  #endif

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

  #ifdef USE_INT8_PREPROCESSOR
  int8_t *new_slice_data = this->spectrogram_ + ((PREPROCESSOR_FEATURE_COUNT-1) * PREPROCESSOR_FEATURE_SIZE);
  #else
  float *new_slice_data = this->spectrogram_ + ((PREPROCESSOR_FEATURE_COUNT-1) * PREPROCESSOR_FEATURE_SIZE);
  #endif


  #ifdef USE_INT8_PREPROCESSOR
  size_t bytes_to_copy = input->bytes;

  memcpy((void *) (tflite::GetTensorData<int8_t>(input)),
        (const void *) (new_slice_data), bytes_to_copy);
  #else
  // Copy the newest slice's features as input into the streaming model after quantizing them
  for (int i = 0; i < PREPROCESSOR_FEATURE_SIZE; ++i) {
    int8_t converted_value = convert_float_to_int8(new_slice_data[i], input_scale, input_zero_point);
    input->data.int8[i] = converted_value;
  }
  #endif

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

bool OnDeviceWakeWord::detect_wakeword(ringbuf_handle_t &ring_buffer) {
  if (!this->update_spectrogram_(ring_buffer)) {
    return false;
  }

  uint32_t streaming_length = micros();
  float streaming_prob = this->perform_streaming_inference_();

  // Add the most recent probability to the sliding window
  this->recent_streaming_probabilities_[this->last_n_index_] = streaming_prob;
  ++this->last_n_index_;
  if (this->last_n_index_ == STREAMING_MODEL_SLIDING_WINDOW_MEAN_LENGTH)
    this->last_n_index_ = 0;

  float sum = 0.0;
  for (int i = 0; i < STREAMING_MODEL_SLIDING_WINDOW_MEAN_LENGTH; ++i) {
    sum += this->recent_streaming_probabilities_[i];
  }

  float sliding_window_average = sum/static_cast<float>(STREAMING_MODEL_SLIDING_WINDOW_MEAN_LENGTH);

  // if (sliding_window_average > 0.4) {
  //       ESP_LOGD(TAG_LOCAL, "streaming rolling wake word average=%.3f", sliding_window_average);
  // }

  this->ignore_windows_ = std::min(this->ignore_windows_+1, 0);
  if (this->ignore_windows_ < 0) {
    return false;
  }

  if (sliding_window_average > STREAMING_MODEL_PROBABILITY_CUTOFF) {
    this->ignore_windows_ = -PREPROCESSOR_FEATURE_COUNT;
    for (int n = 0; n < STREAMING_MODEL_SLIDING_WINDOW_MEAN_LENGTH; ++n) {
      this->recent_streaming_probabilities_[n] = 0.0;
    }
    return true;
  }

  return false;
}

bool OnDeviceWakeWord::slice_available_(ringbuf_handle_t &ring_buffer) {
  uint8_t slices_to_process = rb_bytes_filled(ring_buffer) / (NEW_SAMPLES_TO_GET * sizeof(int16_t));

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

#ifdef USE_INT8_PREPROCESSOR
bool OnDeviceWakeWord::generate_single_feature_(const int16_t *audio_data, const int audio_data_size,
                                                     int8_t feature_output[PREPROCESSOR_FEATURE_SIZE]) {
#else
bool OnDeviceWakeWord::generate_single_feature_(const int16_t *audio_data, const int audio_data_size,
                                                     float feature_output[PREPROCESSOR_FEATURE_SIZE]) {
#endif
  TfLiteTensor *input = this->preprocessor_interperter_->input(0);
  TfLiteTensor *output = this->preprocessor_interperter_->output(0);
  std::copy_n(audio_data, audio_data_size, tflite::GetTensorData<int16_t>(input));

  if (this->preprocessor_interperter_->Invoke() != kTfLiteOk) {
    ESP_LOGE(TAG_LOCAL, "Failed to preprocess audio for local wake word.");
    return false;
  }

  #ifdef USE_INT8_PREPROCESSOR
  std::memcpy(feature_output, tflite::GetTensorData<int8_t>(output), PREPROCESSOR_FEATURE_SIZE * sizeof(int8_t));
  #else
  std::memcpy(feature_output, tflite::GetTensorData<float>(output), PREPROCESSOR_FEATURE_SIZE * sizeof(float));
  #endif

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

bool OnDeviceWakeWord::register_streaming_ops_(tflite::MicroMutableOpResolver<12> &op_resolver) {
  if (op_resolver.AddCallOnce() != kTfLiteOk)
    return false;
  if (op_resolver.AddVarHandle() != kTfLiteOk)
    return false;
  if (op_resolver.AddReshape() != kTfLiteOk)
    return false;
  // if (op_resolver.AddPad() != kTfLiteOk)
  //   return false;
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
  // if (op_resolver.AddMul() != kTfLiteOk)
  //   return false;
  // if (op_resolver.AddAdd() != kTfLiteOk)
  //   return false;
  if (op_resolver.AddMean() != kTfLiteOk)
    return false;
  // if (op_resolver.AddDepthwiseConv2D() != kTfLiteOk)
  //   return false;
  // if (op_resolver.AddAveragePool2D() != kTfLiteOk)
  //   return false;
  if (op_resolver.AddFullyConnected() != kTfLiteOk)
    return false;
  if (op_resolver.AddSoftmax() != kTfLiteOk)
    return false;
  if (op_resolver.AddQuantize() != kTfLiteOk)
    return false;

  return true;
}

}  // namespace voice_assistant
}  // namespace esphome
