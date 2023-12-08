#pragma once

#include <ringbuf.h>

#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

namespace esphome {
namespace voice_assistant {

static const char *const TAG_LOCAL = "local_wake_word";

// Constants used for audio preprocessor model
enum {
  PREPROCESSOR_FEATURE_SIZE = 40,   // The number of features the audio preprocessor generates per slice
  PREPROCESSOR_FEATURE_COUNT = 64,  // The number of slices in the spectrogram
  FEATURE_STRIDE_MS = 20,           // How frequently the preprocessor generates a new set of features
  FEATURE_DURATION_MS = 30,         // Duration of each slice used as input into the preprocessor
  AUDIO_SAMPLE_FREQUENCY = 16000,   // Audio sample frequency in hertz
  SPECTROGRAM_TOTAL_PIXELS = (PREPROCESSOR_FEATURE_SIZE * PREPROCESSOR_FEATURE_COUNT),
  HISTORY_SAMPLES_TO_KEEP = ((FEATURE_DURATION_MS - FEATURE_STRIDE_MS) * (AUDIO_SAMPLE_FREQUENCY / 1000)),
  NEW_SAMPLES_TO_GET = (FEATURE_STRIDE_MS * (AUDIO_SAMPLE_FREQUENCY / 1000)),
  SAMPLE_DURATION_COUNT = FEATURE_DURATION_MS * AUDIO_SAMPLE_FREQUENCY / 1000,
  MAX_AUDIO_SAMPLE_SIZE = 512,
};

// Constants used for setting up tensor arenas
// TODO: Optimize these values; they are currently much larger than needed
enum {
  STREAMING_MODEL_ARENA_SIZE = 1024 * 1000,
  STREAMING_MODEL_VARIABLE_ARENA_SIZE = 10 * 1000,
  NONSTREAMING_MODEL_ARENA_SIZE = 1024 * 1000,
  PREPROCESSOR_ARENA_SIZE = 16 * 1024,
};

static constexpr float STREAMING_MODEL_PROBABILITY_CUTOFF = 0.95;
static constexpr float NONSTREAMING_MODEL_PROBABILITY_CUTOFF = 0.97;

// The nonstreaming model seems to more accurately detect the wake word if the phrase is in the middle of the
// spectrogram; however, this causes more latency => 40*20ms = 800 ms
static constexpr int STREAMING_MODEL_SUCCESSIVE_WORDS_NEEDED = 5;

class OnDeviceWakeWord {
 public:
  bool intialize_models();

  bool run_inference(ringbuf_handle_t &ring_buffer);

 protected:
  const tflite::Model *preprocessor_model_{nullptr};
  const tflite::Model *streaming_model_{nullptr};
  const tflite::Model *nonstreaming_model_{nullptr};
  tflite::MicroInterpreter *streaming_interpreter_{nullptr};
  tflite::MicroInterpreter *nonstreaming_interpreter_{nullptr};
  tflite::MicroInterpreter *preprocessor_interperter_{nullptr};

  uint8_t *streaming_tensor_arena_{nullptr};
  uint8_t *nonstreaming_tensor_arena_{nullptr};
  uint8_t *preprocessor_tensor_arena_{nullptr};

  int8_t *spectrogram_{nullptr};
  int8_t *streaming_model_input_{nullptr};
  int8_t *nonstreaming_model_input_{nullptr};
  // float *spectrogram_{nullptr};
  // float *streaming_model_input_{nullptr};
  // float *nonstreaming_model_input_{nullptr};

  // Stores audio fed into feature generator
  int16_t *preprocessor_audio_buffer_;
  int16_t *preprocessor_stride_buffer_;

  uint8_t succesive_wake_words = 0;

  void copy_streaming_external_variables_();

  void clear_streaming_external_variables_();

  // Adapted from TFLite micro speech example
  bool populate_feature_data_(ringbuf_handle_t &ring_buffer);

  // Adapted from TFLite micro speech example
  // Return true if successful
  bool stride_audio_samples_(int16_t **audio_samples, ringbuf_handle_t &ring_buffer);

  // Return true on success
  bool register_preprocessor_ops_(tflite::MicroMutableOpResolver<18> &op_resolver);

  // Return true if successful
  bool register_streaming_ops_(tflite::MicroMutableOpResolver<8> &op_resolver);

  // Return true if successful
  bool register_nonstreaming_ops_(tflite::MicroMutableOpResolver<9> &op_resolver);

  // Adapted from TFLite micro speech example
  bool generate_single_feature(const int16_t *audio_data, const int audio_data_size,
                                     int8_t feature_output[PREPROCESSOR_FEATURE_SIZE]);
};
}  // namespace voice_assistant
}  // namespace esphome
