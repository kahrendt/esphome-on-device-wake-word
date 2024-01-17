#pragma once

#include "esphome/core/automation.h"
#include "esphome/core/component.h"
#include "esphome/core/ring_buffer.h"

#include "esphome/components/microphone/microphone.h"

#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace esphome {
namespace micro_wake_word {

// Constants used for audio preprocessor model
//
// The number of slices in the spectrogram when trained IMPLEMENTATION_DETAILS: This can depend on the model, but
// doesn't directly affect any of the inference code
static const uint8_t SPECTROGRAM_LENGTH = 74;

// The following are dictated by the preprocessor model
//
// The number of features the audio preprocessor generates per slice
static const uint8_t PREPROCESSOR_FEATURE_SIZE = 40;
// How frequently the preprocessor generates a new set of features
static const uint8_t FEATURE_STRIDE_MS = 20;
// Duration of each slice used as input into the preprocessor
static const uint8_t FEATURE_DURATION_MS = 30;
// Audio sample frequency in hertz
static const uint16_t AUDIO_SAMPLE_FREQUENCY = 16000;
static const uint16_t HISTORY_SAMPLES_TO_KEEP =
    ((FEATURE_DURATION_MS - FEATURE_STRIDE_MS) * (AUDIO_SAMPLE_FREQUENCY / 1000));
static const uint16_t NEW_SAMPLES_TO_GET = (FEATURE_STRIDE_MS * (AUDIO_SAMPLE_FREQUENCY / 1000));
static const uint16_t SAMPLE_DURATION_COUNT = FEATURE_DURATION_MS * AUDIO_SAMPLE_FREQUENCY / 1000;
static const uint16_t MAX_AUDIO_SAMPLE_SIZE = 512;

// Constants used for setting up tensor arenas
// TODO: Optimize these values; they are currently much larger than needed
static const uint32_t STREAMING_MODEL_ARENA_SIZE = 1024 * 1000;
static const uint32_t STREAMING_MODEL_VARIABLE_ARENA_SIZE = 10 * 1000;
static const uint32_t PREPROCESSOR_ARENA_SIZE = 16 * 1024;

enum State {
  IDLE,
  START_MICROPHONE,
  STARTING_MICROPHONE,
  DETECTING_WAKE_WORD,
  STOP_MICROPHONE,
  STOPPING_MICROPHONE,
};

class MicroWakeWord : public Component {
 public:
  void setup() override;
  void loop() override;
  float get_setup_priority() const override;

  void start();
  void stop();

  bool is_running() const { return this->state_ != State::IDLE; }

  bool initialize_models();

  /** Detects if wake word has been said
   *
   * If enough audio samples are available, it will generate one slice of new features.
   * If the streaming model predicts the wake word, then the nonstreaming model confirms it.
   * @param ring_Buffer Ring buffer containing raw audio samples
   * @return True if the wake word is detected, false otherwise
   */
  bool detect_wakeword();

  void set_streaming_model_probability_cutoff(float streaming_model_probability_cutoff) {
    this->streaming_model_probability_cutoff_ = streaming_model_probability_cutoff;
  }
  void set_streaming_model_sliding_window_mean_length(size_t length);

  void set_microphone(microphone::Microphone *microphone) { this->microphone_ = microphone; }

  Trigger<std::string> *get_wake_word_detected_trigger() const { return this->wake_word_detected_trigger_; }

 protected:
  void set_state_(State state);
  int read_microphone_();

  microphone::Microphone *microphone_{nullptr};
  Trigger<std::string> *wake_word_detected_trigger_ = new Trigger<std::string>();
  State state_{State::IDLE};
  HighFrequencyLoopRequester high_freq_;

  std::unique_ptr<RingBuffer> ring_buffer_;

  int16_t *input_buffer_;

  const tflite::Model *preprocessor_model_{nullptr};
  const tflite::Model *streaming_model_{nullptr};
  tflite::MicroInterpreter *streaming_interpreter_{nullptr};
  tflite::MicroInterpreter *preprocessor_interperter_{nullptr};

  std::vector<float> recent_streaming_probabilities_;
  size_t last_n_index_{0};

  // Increasing either of these will reduce the rate of false acceptances while increasing the false rejection rate
  // IMPLEMENTATION DETAILS: Allow setting defaults specific to each model
  float streaming_model_probability_cutoff_{0.5};
  size_t streaming_model_sliding_window_mean_length_{10};

  // When the wake word detection first starts or after the word has been detected once, we ignore the first trained
  // spectrogram's length of probabilities
  // IMPLEMENTATION DETAILS: A model should be able to set this value, as it may have been trained on a shorter or
  // longer spectrogram
  int16_t ignore_windows_{-SPECTROGRAM_LENGTH};

  uint8_t *streaming_var_arena_{nullptr};
  uint8_t *streaming_tensor_arena_{nullptr};
  uint8_t *preprocessor_tensor_arena_{nullptr};
  int8_t *new_features_data_{nullptr};

  tflite::MicroResourceVariables *mrv_{nullptr};

  // Stores audio fed into feature generator preprocessor
  int16_t *preprocessor_audio_buffer_;
  int16_t *preprocessor_stride_buffer_;

  bool detected_{false};

  /// @brief Returns true if there are enough audio samples in the buffer to generate another slice of features
  bool slice_available_();

  /** Shifts previous feature slices over by one and generates a new slice of features
   *
   * @param ring_buffer ring buffer containing raw audio samples
   * @return True if a new slice of features was generated, false otherwise
   */
  bool update_features_();

  /** Generates features from audio samples
   *
   * Adapted from TFLite micro speech example
   * @param audio_data Pointer to array with the audio samples
   * @param audio_data_size The number of samples to use as input to the preprocessor model
   * @param feature_output Array that will store the features
   * @return True if successful, false otherwise.
   */
  bool generate_single_feature_(const int16_t *audio_data, const int audio_data_size,
                                int8_t feature_output[PREPROCESSOR_FEATURE_SIZE]);

  /** Performs inference over the most recent feature slice with the streaming model
   *
   * @return Probability of the wake word between 0.0 and 1.0
   */
  float perform_streaming_inference_();

  /** Strides the audio samples by keeping the last 10 ms of the previous slice
   *
   * Adapted from the TFLite micro speech example
   * @param ring_buffer Ring buffer containing raw audio samples
   * @param audio_samples Pointer to an array that will store the strided audio samples
   * @return True if successful, false otherwise
   */
  bool stride_audio_samples_(int16_t **audio_samples);

  /// @brief Returns true if successfully registered the preprocessor's TensorFlow operations
  bool register_preprocessor_ops_(tflite::MicroMutableOpResolver<18> &op_resolver);

  /// @brief Returns true if successfully registered the streaming model's TensorFlow operations
  bool register_streaming_ops_(tflite::MicroMutableOpResolver<12> &op_resolver);
};

template<typename... Ts> class StartAction : public Action<Ts...>, public Parented<MicroWakeWord> {
 public:
  void play(Ts... x) override { this->parent_->start(); }
};

template<typename... Ts> class StopAction : public Action<Ts...>, public Parented<MicroWakeWord> {
 public:
  void play(Ts... x) override { this->parent_->stop(); }
};

template<typename... Ts> class IsRunningCondition : public Condition<Ts...>, public Parented<MicroWakeWord> {
 public:
  bool check(Ts... x) override { return this->parent_->is_running(); }
};

}  // namespace micro_wake_word
}  // namespace esphome
