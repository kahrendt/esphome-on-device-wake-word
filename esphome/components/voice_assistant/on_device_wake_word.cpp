#include "on_device_wake_word.h"

#include "esphome/core/hal.h"
#include "esphome/core/helpers.h"
#include "esphome/core/log.h"

#include <freertos/stream_buffer.h>

#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "model.h"
#include "audio_preprocessor_int8_model_data.h"

#include <cmath>

namespace esphome {
namespace voice_assistant {

bool OnDeviceWakeWord::intialize_models() {
  ExternalRAMAllocator<uint8_t> arena_allocator(ExternalRAMAllocator<uint8_t>::ALLOW_FAILURE);
  ExternalRAMAllocator<int8_t> features_allocator(ExternalRAMAllocator<int8_t>::ALLOW_FAILURE);
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

  this->new_features_data_ = features_allocator.allocate(PREPROCESSOR_FEATURE_SIZE);
  if (this->new_features_data_ == nullptr) {
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

  this->preprocessor_model_ = tflite::GetModel(g_audio_preprocessor_int8_tflite);
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

  for (int n = 0; n < STREAMING_MODEL_SLIDING_WINDOW_MEAN_LENGTH; ++n) {
    this->recent_streaming_probabilities_[n] = 0.0;
  }

  return true;
}

bool OnDeviceWakeWord::update_features_(StreamBufferHandle_t &ring_buffer) {
  // Verify we have enough samples for a feature slice
  if (!this->slice_available_(ring_buffer)) {
    return false;
  }

  // Retrieve strided audio samples
  int16_t *audio_samples = nullptr;
  if (!this->stride_audio_samples_(&audio_samples, ring_buffer)) {
    return false;
  }

  // Compute the features for the newest audio samples
  if (!this->generate_single_feature_(audio_samples, SAMPLE_DURATION_COUNT, this->new_features_data_)) {
    return false;
  }

  return true;
}

float OnDeviceWakeWord::perform_streaming_inference_() {
  TfLiteTensor *input = this->streaming_interpreter_->input(0);

  size_t bytes_to_copy = input->bytes;

  memcpy((void *) (tflite::GetTensorData<int8_t>(input)),
        (const void *) (this->new_features_data_), bytes_to_copy);

  uint32_t prior_invoke = millis();

  TfLiteStatus invoke_status = this->streaming_interpreter_->Invoke();
  if (invoke_status != kTfLiteOk) {
    ESP_LOGW(TAG_LOCAL, "Streaming Interpreter Invoke failed");
    return false;
  }

  ESP_LOGV(TAG_LOCAL, "Streaming Inference Latency=%u ms", (millis() - prior_invoke));

  TfLiteTensor *output = this->streaming_interpreter_->output(0);

  return static_cast<float>(output->data.uint8[0])/255.0;
}

bool OnDeviceWakeWord::detect_wakeword(StreamBufferHandle_t &ring_buffer) {
  if (!this->update_features_(ring_buffer)) {
    return false;
  }

  uint32_t streaming_length = micros();
  float streaming_prob = this->perform_streaming_inference_();

  // Add the most recent probability to the sliding window
  // IMPLEMENTATION DETAILS: This sliding window buffer can be better implemented with an std::deque; the user/model should be able to set the length of the window
  this->recent_streaming_probabilities_[this->last_n_index_] = streaming_prob;
  ++this->last_n_index_;
  if (this->last_n_index_ == STREAMING_MODEL_SLIDING_WINDOW_MEAN_LENGTH)
    this->last_n_index_ = 0;

  float sum = 0.0;
  for (int i = 0; i < STREAMING_MODEL_SLIDING_WINDOW_MEAN_LENGTH; ++i) {
    sum += this->recent_streaming_probabilities_[i];
  }

  float sliding_window_average = sum/static_cast<float>(STREAMING_MODEL_SLIDING_WINDOW_MEAN_LENGTH);

  this->ignore_windows_ = std::min(this->ignore_windows_+1, 0);
  if (this->ignore_windows_ < 0) {
    return false;
  }

  if (sliding_window_average > STREAMING_MODEL_PROBABILITY_CUTOFF) {
    this->ignore_windows_ = -SPECTROGRAM_LENGTH;
    for (int n = 0; n < STREAMING_MODEL_SLIDING_WINDOW_MEAN_LENGTH; ++n) {
      this->recent_streaming_probabilities_[n] = 0.0;
    }
    return true;
  }

  return false;
}

bool OnDeviceWakeWord::slice_available_(StreamBufferHandle_t &ring_buffer) {
  uint8_t slices_to_process = xStreamBufferBytesAvailable(ring_buffer) / (NEW_SAMPLES_TO_GET * sizeof(int16_t));

  if (xStreamBufferBytesAvailable(ring_buffer) > NEW_SAMPLES_TO_GET*sizeof(int16_t)) {
    return true;
  }
  return false;
}

bool OnDeviceWakeWord::stride_audio_samples_(int16_t **audio_samples, StreamBufferHandle_t &ring_buffer) {
  // Copy 320 bytes (160 samples over 10 ms) into preprocessor_audio_buffer_ from history in
  // preprocessor_stride_buffer_
  memcpy((void *) (this->preprocessor_audio_buffer_), (void *) (this->preprocessor_stride_buffer_),
         HISTORY_SAMPLES_TO_KEEP * sizeof(int16_t));

  if (xStreamBufferBytesAvailable(ring_buffer) < NEW_SAMPLES_TO_GET * sizeof(int16_t)) {
    ESP_LOGD(TAG_LOCAL, "Audio Buffer not full enough");
    return false;
  }

  // Copy 640 bytes (320 samples over 20 ms) from the ring buffer
  // The first 320 bytes (160 samples over 10 ms) will be from history
  int bytes_read = xStreamBufferReceive(ring_buffer, ((void *) (this->preprocessor_audio_buffer_ + HISTORY_SAMPLES_TO_KEEP)),
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
                                                     int8_t feature_output[PREPROCESSOR_FEATURE_SIZE]) {
  TfLiteTensor *input = this->preprocessor_interperter_->input(0);
  TfLiteTensor *output = this->preprocessor_interperter_->output(0);
  std::copy_n(audio_data, audio_data_size, tflite::GetTensorData<int16_t>(input));

  if (this->preprocessor_interperter_->Invoke() != kTfLiteOk) {
    ESP_LOGE(TAG_LOCAL, "Failed to preprocess audio for local wake word.");
    return false;
  }
  std::memcpy(feature_output, tflite::GetTensorData<int8_t>(output), PREPROCESSOR_FEATURE_SIZE * sizeof(int8_t));

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
  if (op_resolver.AddMean() != kTfLiteOk)
    return false;
  if (op_resolver.AddFullyConnected() != kTfLiteOk)
    return false;
  if (op_resolver.AddLogistic() != kTfLiteOk)
    return false;
  if (op_resolver.AddQuantize() != kTfLiteOk)
    return false;

  return true;
}

}  // namespace voice_assistant
}  // namespace esphome
