# On Device Wake Word Detection for ESPHome's Voice Assistant Component

This component implements wake word detection on the ESPHome device itself. The component uses "Hey Jarvis" as the wake word. The component starts the assist pipeline after detecting the wake word. It does not use the Wyoming-openwakeword pipeline at all.

**The component is in a very early stage!** It works well and is usable in my tests, but the latency needs improvement before I submit this to the ESPHome project. I outline the plans to reduce latency further in this document.

Wake word detection is done entirely with [TensorFlow Lite Micro](https://github.com/espressif/esp-tflite-micro/). Wake word models are trained without using [Espressif's proprietary Skainet](https://github.com/espressif/esp-skainet). I have it tested only on an ESP32-S3 Box Lite. It currently requires external PSRAM and the [ESP-ADF component](https://github.com/esphome/esphome/pull/5230).

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
      - if:
          condition:
            not:
              - voice_assistant.is_running
          then:
            - voice_assistant.start_continuous
      - script.execute: reset_display
    on_turn_off:
      - voice_assistant.stop
      - lambda: id(va).set_use_local_wake_word(false);
      - script.execute: reset_display
```


## Detection Process

The component detects the wake word in three stages. Raw audio data is processed into 40 features every 20 ms to generate a spectrogram. Then, a streaming inference model uses each new slice of feature data to detect the wake word. This model is very lightweight, so false positives are common. A slower second model uses the full spectrogram to confirm detection to reduce false positives. The component starts the assistant pipeline if the second model detects the wake word.

The first stage processes the raw audio data at a sample rate of 16 kHz via the [micro_speech preprocessor](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/examples/micro_speech). The preprocessor generates 40 features over 30 ms (the window duration) of audio data. The preprocessor generates these features every 20 ms (the stride duration), so the first 10 ms of audio data is part of the previous window. This process is similar to calculating a Mel spectrogram for the audio data, but it is lightweight for devices with limited processing power. See the TFLite Micro example for full details on how the audio is processed.

The lightweight streaming model performs inferences every 20 ms on the newest audio stride. The model is a variation of a Depthwise Separable Convolutional Neural Network (DS-CNN) found in [Hello Edge: Keyword Spotting on Microcontrollers](https://arxiv.org/pdf/1711.07128.pdf) by Zhang, Suda, Lai, and Chandra. It is a very lightweight model, with only 2322 trainable parameters, and requires 0.069 M floating point operations and executes faster than the 20 ms stride length. Streaming and training the model uses open-sourced code from [Google Research](https://github.com/google-research/google-research/tree/master/kws_streaming) found in the paper [Streaming Keyword Spotting on Mobile Devices](https://arxiv.org/pdf/2005.06720.pdf) by Rykabov, Kononenko, Subrahmanya, Visontai, and Laurenzo.

When the streaming model predicts the wake word, the entire spectrogram is processed and confirmed using a modified [Broadcast Residual Learning model (BC-Resnet)](https://arxiv.org/pdf/2106.04140.pdf) by Kim, Chang, Lee, and Sung. This model has 7308 trainable parameters and requires 6.022 M floating point ops. The BC-Resnet inference is slow and is the biggest drawback to the current implementation. On an ESP32-S3, it takes approximately 700 ms to complete.

## Next Steps and Improvement Plans

There are minimal false positives in real-world testing when using this two-pass approach. However, if the streaming model predicts the wake word, the component uses the slow BC-Resnet model to perform an inference. This delay causes a gap in processed audio and blocks ESPHome's main loop. There are several planned improvements to reduce the latency.

  - I trained the current models to predict the wake word over 2 seconds. That is longer than most wake words. Reducing the sample clip durations will speed up this second inference with a minimal loss in accuracy.
  - Currently, the models use floating point operations. Quantization will significantly improve both models' latency, as it can use Espressif's [ESP-NN optimizations](https://github.com/espressif/esp-nn). We could then add more trainable parameters to the first-stage DS-CNN model to further reduce the frequency of the second-stage BC-Resnet model inferences.
  - We could convert the second stage BC-Resnet model to use Espressif's native [deep learning](https://github.com/espressif/esp-dl) architecture, which would also reduce the latency.
  - We can improve the samples used for model training. The negative samples used in the second stage BC-Resnet model should only use phrases very similar to the actual wake word, as it only executes if the first stage DS-CNN predicts the wake word. Also, the second stage model currently is more accurate with the wake word when centered in the spectrogram, which increases latency. We could reduce this latency simply by shifting the samples in the data augmentation step.
  - An alternative approach would use the current streaming model as a first-pass approach for wake word detection. It could then send buffered data to [Wyoming-openwwakeword](https://github.com/rhasspy/wyoming-openwakeword) for confirmation. This approach would drastically reduce the network traffic used for the current ESPHome/Home Assistant approach for wake word detection. This approach should be practical even for slower ESP32-based devices like the [M5 Atom Echo](https://shop.m5stack.com/products/atom-echo-smart-speaker-dev-kit).

There are several improvements needed for the ESPHome component.

  - Users should be able to set the current thresholds for detection (currently defined in ``on_device_wake_word.h``). Default values can be better fine-tuned.
  - Switch from using ESP-ADF's ring buffer to a generic FreeRTOS approach.
  - The current implementation allocates a large amount of memory to enable experimentation with various models. We can drastically reduce the memory allocated with no performance penalty.

There are several remaining general improvements.

  - Write an end-to-end Jupyter/Google Colab notebook for sample generation, augmentation, and training that generates new wake word models.
  - Both models should undergo hyper-parameterization optimization to increase accuracy and reduce latency.
  - Perform tests for false-reject and false-accept rates to more directly compare to openWakeWord's capabilities.



## Model Training Process

We generate positive and negative samples using [openWakeWord](https://github.com/dscripka/openWakeWord), which relies on [Piper sample generator](https://github.com/rhasspy/piper-sample-generator). We also use openWakeWord to augment the positive and negative samples. Then, we train the two models using code from [Google Research](https://github.com/google-research/google-research/tree/master/kws_streaming). We use the external streaming version for the first stage DS-CNN model. We use the non-streaming version of the second stage BC-Resnet model.

Both models' parameters are set via

```
TRAIN_DIR = 'models/bc_resnet'
SAMPLE_RATE = 16000
CLIP_DURATION_MS = 2000
FEATURE_BIN_COUNT = 40
BACKGROUND_FREQUENCY = 0
BACKGROUND_VOLUME_RANGE = 0
TIME_SHIFT_MS = 100.0
WINDOW_STRIDE = 20
WINDOW_SIZE_MS = 30
PREPROCESS = 'micro'
WANTED_WORDS = "wakeword,unknown"
DATA_URL = ''
SILENT_PERCENTAGE = 0
UNKNOWN_PERCENTAGE = 0
DATASET_DIR =  'data_training/'
```

We train the non-streaming BC-Resnet model using

```
!python -m kws_streaming.train.model_train_eval \
--data_url={DATA_URL} \
--data_dir={DATASET_DIR} \
--train_dir={TRAIN_DIR} \
--split_data 0 \
--mel_upper_edge_hertz 7500.0 \
--mel_lower_edge_hertz 125.0 \
--silence_percentage={SILENT_PERCENTAGE} \
--unknown_percentage={UNKNOWN_PERCENTAGE} \
--background_frequency={BACKGROUND_FREQUENCY} \
--background_volume={BACKGROUND_VOLUME_RANGE} \
--how_many_training_steps 20000,20000,20000,20000 \
--learning_rate 0.001,0.0005,0.0001,0.00002 \
--window_size_ms={WINDOW_SIZE_MS} \
--window_stride_ms={WINDOW_STRIDE} \
--clip_duration_ms={CLIP_DURATION_MS} \
--mel_num_bins 40 \
--dct_num_features 40 \
--preprocess={PREPROCESS} \
--feature_type='raw' \
--micro_enable_pcan 1 \
--micro_min_signal_remaining 0.05 \
--micro_out_scale 1 \
--micro_features_scale 0.0390625 \
--resample 0.1 \
--alsologtostderr \
--train 1 \
--wanted_words={WANTED_WORDS} \
--use_spec_augment 0 \
--time_masks_number 2 \
--time_mask_max_size 20 \
--frequency_masks_number 2 \
--frequency_mask_max_size 3 \
--pick_deterministically 1 \
--return_softmax 1 \
bc_resnet \
--sub_groups 5 \
--last_filters 32 \
--first_filters 16 \
--paddings 'causal' \
--dilations '(1,1),(2,1),(4,1),(8,1)' \
--strides '(1,1),(1,2),(1,2),(1,1)' \
--blocks_n '2, 2, 4, 4' \
--filters '6, 9, 12, 15' \
--dropouts '0.1, 0.1, 0.1, 0.1' \
--pools '1, 1, 1, 1' \
--max_pool 0
```

We train the streaming DS-CNN model using:

```
!python -m kws_streaming.train.model_train_eval \
--data_url={DATA_URL} \
--data_dir={DATASET_DIR} \
--train_dir={TRAIN_DIR} \
--split_data 0 \
--mel_upper_edge_hertz 7500.0 \
--mel_lower_edge_hertz 125.0 \
--silence_percentage={SILENT_PERCENTAGE} \
--unknown_percentage={UNKNOWN_PERCENTAGE} \
--background_frequency={BACKGROUND_FREQUENCY} \
--background_volume={BACKGROUND_VOLUME_RANGE} \
--how_many_training_steps 20000,20000,20000,20000 \
--learning_rate 0.001,0.0005,0.0001,0.00002 \
--window_size_ms={WINDOW_SIZE_MS} \
--window_stride_ms={WINDOW_STRIDE} \
--clip_duration_ms={CLIP_DURATION_MS} \
--mel_num_bins 40 \
--dct_num_features 40 \
--preprocess={PREPROCESS} \
--feature_type='raw' \
--micro_enable_pcan 1 \
--micro_min_signal_remaining 0.05 \
--micro_out_scale 1 \
--micro_features_scale 0.0390625 \
--resample 0.1 \
--alsologtostderr \
--train 1 \
--wanted_words={WANTED_WORDS} \
--use_spec_augment 0 \
--time_masks_number 2 \
--time_mask_max_size 20 \
--frequency_masks_number 2 \
--frequency_mask_max_size 3 \
--pick_deterministically 1 \
--return_softmax 1 \
ds_cnn \
--cnn1_kernel_size "(10,4)" \
--cnn1_dilation_rate "(1,1)" \
--cnn1_strides "(1,2)" \
--cnn1_padding "valid" \
--cnn1_filters 16 \
--cnn1_act 'relu' \
--bn_momentum 0.98 \
--bn_center 1 \
--bn_scale 0 \
--bn_renorm 0 \
--dw2_kernel_size '(3,3),(3,3),(3,3)' \
--dw2_dilation_rate '(1,1),(2,2),(2,2)' \
--dw2_strides '(1,1),(1,1),(1,1)' \
--dw2_padding "valid" \
--dw2_act "'relu','relu','relu'" \
--cnn2_filters '16,16,16' \
--cnn2_act "'relu','relu','relu'" \
--dropout1 0.2 
```
