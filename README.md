# On Device Wake Word Detection for ESPHome's Voice Assistant Component

This component implements wake word detection on the ESPHome device itself. It currently implements "Hey Jarvis" as the wake word, but any custom word/phrase is possible. The component starts the assist pipeline immediately after detecting the wake word without using Wyoming-openWakeWord.

It works well. The performance metrics are comparable to openWakeWord's and will be improved further with more sources for negative training data. Using the [wake-word-benchmark framework from Picovoice](https://github.com/Picovoice/wake-word-benchmark), the model currently has a false accept per hour rate of 0.204 and a false reject rate of 0.06. The detection latency is extremely low, often faster than an ESP32 device using the openWakeWord pipeline.

Wake word detection is done entirely with [TensorFlow Lite Micro](https://github.com/espressif/esp-tflite-micro/). Wake word models are trained without using [Espressif's proprietary Skainet](https://github.com/espressif/esp-skainet) and can be customized without samples from different speakers. Sample preparation, generation, and augmentation heavily use code from [openWakeWord](https://github.com/dscripka/openWakeWord).

I have tested it only on an ESP32-S3 Box Lite, though any S3 device supporting ESP-ADF should work. It requires external PSRAM and the [ESP-ADF component](https://github.com/esphome/esphome/pull/5230).

## YAML Configuration

The ``example_esphome_yaml`` folder has a full example for an S3 Box Lite which only requires configuring the WiFi settings.

Here is a bare-bones example that implements the ``voice_assistant`` component as well as a switch to toggle the on-device wake word detection. It also requires an appropriate ``esp_adf`` component configuration.

```
external_components:
  - source: github://pr#5230
    components: esp_adf
    refresh: 0s
  - source:
      type: git
      url: https://github.com/kahrendt/esphome-on-device-wake-word
      ref: dev
    refresh: 0s
    components: [ voice_assistant ]  

esp_adf:
  board: esp32s3boxlite

voice_assistant:
  id: va
  microphone: box_mic
  speaker: box_speaker
  use_wake_word: false
  use_local_wake_word: true
  noise_suppression_level: 2
  auto_gain: 31dBFS
  volume_multiplier: 2.0
  vad_threshold: 3
  on_client_connected:
    - if:
        condition:
          switch.is_on: use_local_wake_word
        then:
          - voice_assistant.start_continuous:
          - lambda: id(init_in_progress) = false;

switch:
  - platform: template
    name: Use on device wake word
    id: use_local_wake_word
    optimistic: true
    restore_mode: RESTORE_DEFAULT_ON
    entity_category: config
    on_turn_on:
      - lambda: id(va).set_use_local_wake_word(true);
    on_turn_off:
      - voice_assistant.stop
      - lambda: id(va).set_use_local_wake_word(false);
```


## Detection Process

The component detects the wake word in two stages. Raw audio data is processed into 40 features every 20 ms. Several of these features construct a spectrogram. A streaming inference model only uses the newest slice of feature data as input to detect the wake word. If the model consistently predicts the wake word over multiple windows, then the component starts the assist pipeline.

The first stage processes the raw monochannel audio data at a sample rate of 16 kHz via the [micro_speech preprocessor](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/examples/micro_speech). The preprocessor generates 40 features over 30 ms (the window duration) of audio data. The preprocessor generates these features every 20 ms (the stride duration), so the first 10 ms of audio data is part of the previous window. This process is similar to calculating a Mel spectrogram for the audio data, but it is lightweight for devices with limited processing power. See the linked TFLite Micro example for full details on how the audio is processed.

The streaming model performs inferences every 20 ms on the newest audio stride. The model is an [inceptional neural network](https://towardsdatascience.com/a-simple-guide-to-the-versions-of-the-inception-network-7fc52b863202?gi=6bc760f44aef) converted for streaming. It executes an inference in under 10 ms on an ESP32 S3, much faster than the 20 ms stride length. Streaming and training the model uses modified open-sourced code from [Google Research](https://github.com/google-research/google-research/tree/master/kws_streaming) found in the paper [Streaming Keyword Spotting on Mobile Devices](https://arxiv.org/pdf/2005.06720.pdf) by Rykabov, Kononenko, Subrahmanya, Visontai, and Laurenzo.

## Next Steps and Improvement Plans

The model's latency is already extremely low, and the performance metrics are good. However, we can further improve the false rejection and false acceptance rates with better training.

  - Use a larger negative dataset based on more sources.
  - Add more custom negative phrases to the dataset close to the wake word, e.g., "Hey Jacky."
  - The model is prone to overfitting. Add test methods to stop training early once particular metrics are met.

There are several improvements needed for the ESPHome component.

  - Switch from using ESP-ADF's ring buffer to a generic FreeRTOS approach.
  - The current implementation allocates a large amount of memory to enable experimentation with various models. We can drastically reduce the memory allocated with no performance penalty.
  - Make it easier to switch between different wake word models/phrases.

There are several remaining general improvements.

  - Write an end-to-end Jupyter/Google Colab notebook for sample generation, augmentation, and training that generates new wake word models.
  - Release scripts that show the test for false-reject and false-accept rates for easy comparison to other wake word detection engines.
  - Add common wake words like "Alexa" and "Okay Nabu."
  - The model and training process allows multiple distinct wake words simultaneously. Experiment to see if this results in a noticeable reduction in accuracy.  
  - Test the code on an ESP32, though it may run too slow since the [ESP-NN optimizations](https://github.com/espressif/esp-nn) perform better on an S3.

## Model Training Process

We generate positive and negative samples using [openWakeWord](https://github.com/dscripka/openWakeWord), which relies on [Piper sample generator](https://github.com/rhasspy/piper-sample-generator). We also use openWakeWord's data tools to augment the positive and negative samples. Then, we train the two models using code from [Google Research](https://github.com/google-research/google-research/tree/master/kws_streaming). The streaming model is an inception neural network converted for streaming.

The models' parameters are set via

```
TRAIN_DIR = 'trained_models/inception'
SAMPLE_RATE = 16000
CLIP_DURATION_MS = 1490
FEATURE_BIN_COUNT = 40
BACKGROUND_FREQUENCY = 0
BACKGROUND_VOLUME_RANGE = 0
TIME_SHIFT_MS = 0.0
WINDOW_STRIDE = 20
WINDOW_SIZE_MS = 30
PREPROCESS = 'none'
WANTED_WORDS = "wakeword,unknown"
DATASET_DIR =  'data_training'
```

We train the streaming model using the follow (note it requires several modification's to the default code that are included in the ``kws_streaming`` folder.)

```
!python -m kws_streaming.train.model_train_eval \
--data_url='' \
--data_dir={DATASET_DIR} \
--train_dir={TRAIN_DIR} \
--split_data 0 \
--mel_upper_edge_hertz 7500.0 \
--mel_lower_edge_hertz 125.0 \
--how_many_training_steps 1000 \
--learning_rate 0.001 \
--window_size_ms={WINDOW_SIZE_MS} \
--window_stride_ms={WINDOW_STRIDE} \
--clip_duration_ms={CLIP_DURATION_MS} \
--eval_step_interval=500 \
--mel_num_bins={FEATURE_BIN_COUNT} \
--dct_num_features={FEATURE_BIN_COUNT} \
--preprocess={PREPROCESS} \
--alsologtostderr \
--train 1 \
--wanted_words={WANTED_WORDS} \
--pick_deterministically 0 \
--return_softmax 1 \
--restore_checkpoint 0 \
inception \
--cnn1_filters '32' \
--cnn1_kernel_sizes '5' \
--cnn1_strides '1' \
--cnn2_filters1 '16,16,16' \
--cnn2_filters2 '32,64,70' \
--cnn2_kernel_sizes '3,5,5' \
--cnn2_strides '1,1,1' \
--dropout 0.0 \
--bn_scale 0
```