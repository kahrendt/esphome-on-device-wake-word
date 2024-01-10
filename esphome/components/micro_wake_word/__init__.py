import esphome.config_validation as cv
import esphome.codegen as cg

from esphome.components import esp32, microphone
from esphome import automation
from esphome.automation import register_action, register_condition


from esphome.const import CONF_ID, CONF_MICROPHONE

CODEOWNERS = ["@kahrendt", "@jesserockz"]
DEPENDENCIES = ["microphone"]

CONF_STREAMING_MODEL_PROBABILITY_CUTOFF = "streaming_model_probability_cutoff"
CONF_STREAMING_MODEL_SLIDING_WINDOW_MEAN_LENGTH = (
    "streaming_model_sliding_window_mean_length"
)
CONF_ON_WAKE_WORD_DETECTED = "on_wake_word_detected"


micro_wake_word_ns = cg.esphome_ns.namespace("micro_wake_word")

MicroWakeWord = micro_wake_word_ns.class_("MicroWakeWord", cg.Component)

StartAction = micro_wake_word_ns.class_("StartAction", automation.Action)
StopAction = micro_wake_word_ns.class_("StopAction", automation.Action)

IsRunningCondition = micro_wake_word_ns.class_(
    "IsRunningCondition", automation.Condition
)

CONFIG_SCHEMA = cv.Schema(
    {
        cv.GenerateID(): cv.declare_id(MicroWakeWord),
        cv.GenerateID(CONF_MICROPHONE): cv.use_id(microphone.Microphone),
        cv.Optional(CONF_STREAMING_MODEL_PROBABILITY_CUTOFF, default=0.5): cv.float_,
        cv.Optional(
            CONF_STREAMING_MODEL_SLIDING_WINDOW_MEAN_LENGTH, default=10
        ): cv.positive_int,
        cv.Optional(CONF_ON_WAKE_WORD_DETECTED): automation.validate_automation(
            single=True
        ),
    }
).extend(cv.COMPONENT_SCHEMA)


async def to_code(config):
    var = cg.new_Pvariable(config[CONF_ID])
    await cg.register_component(var, config)

    mic = await cg.get_variable(config[CONF_MICROPHONE])
    cg.add(var.set_microphone(mic))

    cg.add(
        var.set_streaming_model_probability_cutoff(
            config[CONF_STREAMING_MODEL_PROBABILITY_CUTOFF]
        )
    )
    cg.add(
        var.set_streaming_model_sliding_window_mean_length(
            config[CONF_STREAMING_MODEL_SLIDING_WINDOW_MEAN_LENGTH]
        )
    )

    if on_wake_word_detection_config := config.get(CONF_ON_WAKE_WORD_DETECTED):
        await automation.build_automation(
            var.get_wake_word_detected_trigger(),
            [(cg.std_string, "wake_word")],
            on_wake_word_detection_config,
        )

    esp32.add_idf_component(
        name="esp-tflite-micro",
        repo="https://github.com/espressif/esp-tflite-micro",
        # path="components",
        # components=["esp-radar"],
    )
    # esp32.add_idf_component(
    #     name="esp-nn",
    #     repo="https://github.com/espressif/esp-nn",
    #     # path="components",
    #     # components=["esp-radar"],
    # )

    cg.add_build_flag("-DTF_LITE_STATIC_MEMORY")
    cg.add_build_flag("-DTF_LITE_DISABLE_X86_NEON")
    cg.add_build_flag("-DESP_NN")


MICRO_WAKE_WORD_ACTION_SCHEMA = cv.Schema({cv.GenerateID(): cv.use_id(MicroWakeWord)})


@register_action("micro_wake_word.start", StartAction, MICRO_WAKE_WORD_ACTION_SCHEMA)
@register_action("micro_wake_word.stop", StopAction, MICRO_WAKE_WORD_ACTION_SCHEMA)
@register_condition(
    "micro_wake_word.is_running", IsRunningCondition, MICRO_WAKE_WORD_ACTION_SCHEMA
)
async def micro_wake_word_action_to_code(config, action_id, template_arg, args):
    var = cg.new_Pvariable(action_id, template_arg)
    await cg.register_parented(var, config[CONF_ID])
    return var
