import inspect
import logging
import textwrap
from typing import Callable

import orjson

import dspy
from dspy.adapters.utils import get_field_description_string
from dspy.signatures import InputField, OutputField

logger = logging.getLogger(__name__)

def prepare_models_for_resampling(program: dspy.Module, n: int, teacher_settings: dict | None = None):
    lm = program.get_lm() or dspy.settings.lm

    start_rollout_id = lm.kwargs.get("rollout_id", 0)
    rollout_ids = [start_rollout_id + i for i in range(n)]


    start_rollout_idx, models = 0, []
    # If we have a teacher model, use this as the first model
    if teacher_settings:
        teacher_lm = teacher_settings.get("lm") or lm
        teacher_lm.kwargs["rollout_id"] = rollout_ids[start_rollout_idx]
        models.append(teacher_lm)
        start_rollout_idx += 1

    # The rest of the models are just copies of the base model
    models.extend([lm.copy(rollout_id=r, temperature=1.0) for r in rollout_ids[start_rollout_idx:]])

    return models

def wrap_program(program: dspy.Module, metric: Callable):
    def wrapped_program(example):
        with dspy.context(trace=[]):
            prediction, trace, score = None, None, 0.0
            try:
                prediction = program(**example.inputs())
            except Exception as e:
                logger.warning(e)
            trace = dspy.settings.trace.copy()

        output = None
        score = 0.0
        output_metadata = {}

        try:
            output = metric(example, prediction)
            if isinstance(output, (int, float)):
                score = output
            elif isinstance(output, dspy.Prediction):
                if not hasattr(output, "score"):
                    raise ValueError("When `metric` returns a `dspy.Prediction`, it must contain a `score` field.")
                score = output.score
                # Extract fields from the output dspy.Prediction, excluding `score``
                output_metadata = {
                    k: v for k, v in output.items() if k != "score"
                }
        except Exception as e:
            logger.warning(e)

        return {
            "prediction": prediction,
            "trace": trace,
            "score": score,
            "example": example,
            "output_metadata": output_metadata
        }

    return wrapped_program

def append_a_demo(demo_input_field_maxlen):
    def append_a_demo_(bucket, system, **kwargs):
        predictor2name, name2predictor = kwargs["predictor2name"], kwargs["name2predictor"]
        batch_10p_score = kwargs["batch_10p_score"]

        good = bucket[0]
        trace = good["trace"]
        name2demo = {}

        if good["score"] <= batch_10p_score:
            logger.info(f"Skipping appending a demo as good score {good['score']} is at or below the 10th percentile.")
            return False

        for step in trace:
            predictor, _inputs, _outputs = step

            for k, v in _inputs.items():
                if demo_input_field_maxlen and len(str(v)) > demo_input_field_maxlen:
                    _inputs[k] = f"{str(v)[:demo_input_field_maxlen]}\n\t\t... <TRUNCATED FOR BREVITY>"

            demo = dspy.Example(augmented=True, **_inputs, **_outputs)
            name = predictor2name[id(predictor)]
            name2demo[name] = demo  # keep the last demo for each predictor
        for name, demo in name2demo.items():
            predictor = name2predictor[name]
            predictor.demos.append(demo)

        logger.info(f"Added {len(name2demo)} demos (one each) across all predictors.")
        return True

    return append_a_demo_


def edit_rules(bucket, system, **kwargs):
    """Generates search/replace edits to existing instructions instead of
    always appending. Better for continuous optimization."""
    return _edit_rules_impl(bucket, system, **kwargs)


def append_a_rule(bucket, system, **kwargs):
    predictor2name = kwargs["predictor2name"]
    batch_10p_score, batch_90p_score = kwargs["batch_10p_score"], kwargs["batch_90p_score"]
    prompt_model = kwargs["prompt_model"] or dspy.settings.lm

    module_names = [name for name, _ in system.named_predictors()]
    good, bad = dict(bucket[0]), dict(bucket[-1])  # Copy to avoid mutation
    example = good["example"]

    if good["score"] <= batch_10p_score or bad["score"] >= batch_90p_score:
        logger.info(f"Skipping rule generation as good score {good['score']} is at or below the 10th percentile "
                    f"*or* bad score {bad['score']} is at or above the 90th percentile.")
        return False

    if good["score"] <= bad["score"]:
        if good["score"] > batch_90p_score:
            bad["trace"] = []
            bad["score"] = "N/A"
            bad["prediction"] = {"N/A": "Prediction not available"}
        else:
            good["trace"] = []
            good["score"] = "N/A"
            good["prediction"] = {"N/A": "Prediction not available"}

    better_trajectory = [
        {"module_name": predictor2name[id(p)], "inputs": i, "outputs": dict(o)}
        for p, i, o in good["trace"]
    ]
    worse_trajectory = [
        {"module_name": predictor2name[id(p)], "inputs": i, "outputs": dict(o)}
        for p, i, o in bad["trace"]
    ]

    kwargs = {
        "program_code": inspect.getsource(system.__class__),
        "modules_defn": inspect_modules(system),
        "program_inputs": {**example.inputs()},
        "oracle_metadata": {**example.labels()},
        "better_program_trajectory": better_trajectory,
        "better_program_outputs": dict(good["prediction"]),
        "worse_program_trajectory": worse_trajectory,
        "worse_program_outputs": dict(bad["prediction"] or {}),
        "worse_reward_value": bad["score"],
        "better_reward_value": good["score"],
        "worse_reward_info": bad["output_metadata"],
        "better_reward_info": good["output_metadata"],
        "module_names": module_names,
    }

    kwargs = {k: v if isinstance(v, str) else orjson.dumps(recursive_mask(v), option=orjson.OPT_INDENT_2).decode()
              for k, v in kwargs.items()}

    with dspy.context(trace=[], lm=prompt_model):
        advice_program = dspy.Predict(OfferFeedback)
        advice = advice_program(**kwargs).module_advice

    for name, predictor in system.named_predictors():
        if name in advice:
            logger.info(f"Advice for {name}: {advice[name]}")
            instructions = predictor.signature.instructions + "\n\n" + advice[name]
            predictor.signature = predictor.signature.with_instructions(instructions)

    return True

class OfferFeedback(dspy.Signature):
    """
    You will be given two trajectories of an LLM-driven program's execution. Your goal is to help the program's modules
    build up experience on how to maximize the reward value assigned to the program's outputs if it were to receive
    similar inputs in the future.

    The module won't see its own history. It will rely on your advice balancing being concrete and being generalizable.

    In your advice:
    - Avoid boilerplate. Offer advice that would change the module's behavior for the better in the future.
    - Ensure that advice offered to a module M is specific to that M's specific sub-task, not the overall program.
    - Rely on contrasting the behavior of the worse trajectory against the better trajectory in making recommendations.
    - Ensure each unique module name appears exactly once as a key in the advice dictionary.
    """

    program_code: str = InputField(desc="The code of the program that we are analyzing")
    modules_defn: str = InputField(desc="The definition of each module in the program, including its I/O")
    program_inputs: str = InputField(desc="The inputs to the program that we are analyzing")
    oracle_metadata: str = InputField(desc="Any (hidden) metadata about the training set instance we're analyzing")
    worse_program_trajectory: str = InputField(
        desc="The trajectory of the program's execution, showing each module's I/O"
    )
    worse_program_outputs: str = InputField(desc="The outputs of the program that we are analyzing")
    worse_reward_value: float = InputField(desc="The reward value assigned to the program's outputs")
    worse_reward_info: str = InputField(desc="Additional information that might be helpful to understanding the assigned reward value.")
    better_program_trajectory: str = InputField(
        desc="The trajectory of the program's execution, showing each module's I/O"
    )
    better_program_outputs: str = InputField(desc="The outputs of the program that we are analyzing")
    better_reward_value: float = InputField(desc="The reward value assigned to the program's outputs")
    better_reward_info: str = InputField(desc="Additional information that might be helpful to understanding the assigned reward value.")
    module_names: list[str] = InputField(desc="The names of the modules in the program, for which we seek advice")
    discussion: str = OutputField(desc="Discussing blame of where each module went wrong, if it did")
    module_advice: dict[str, str] = OutputField(
        desc="For each module, describe very concretely: If the module receives ${description of input or patterns "
        "therein}, then it should ${description of content, behavior, or strategies to adopt and/or others to avoid}. "
        "Basically, your advice be such that if the module has access to your tip, it would be much more likely to act "
        "like the successful trajectory rather than the lower-scoring trajectory."
    )

def inspect_modules(program):
    separator = "-" * 80
    output = [separator]

    for name, predictor in program.named_predictors():
        signature = predictor.signature
        instructions = textwrap.dedent(signature.instructions)
        instructions = ("\n" + "\t" * 2).join([""] + instructions.splitlines())

        output.append(f"Module {name}")
        output.append("\n\tInput Fields:")
        output.append(("\n" + "\t" * 2).join([""] + get_field_description_string(signature.input_fields).splitlines()))
        output.append("\tOutput Fields:")
        output.append(("\n" + "\t" * 2).join([""] + get_field_description_string(signature.output_fields).splitlines()))
        output.append(f"\tOriginal Instructions: {instructions}")
        output.append(separator)

    return "\n".join([o.strip("\n") for o in output])


def recursive_mask(o):
    # If the object is already serializable, return it.
    try:
        orjson.dumps(o)
        return o
    except (TypeError, orjson.JSONEncodeError):
        pass

    # If it's a dictionary, apply recursively to its values.
    if isinstance(o, dict):
        return {k: recursive_mask(v) for k, v in o.items()}
    # If it's a list, apply recursively.
    elif isinstance(o, list):
        return [recursive_mask(v) for v in o]
    # If it's a tuple, apply recursively.
    elif isinstance(o, tuple):
        return tuple(recursive_mask(v) for v in o)
    # Otherwise, replace it with a placeholder string (or use repr(o)).
    else:
        return f"<non-serializable: {type(o).__name__}>"


class EditRules(dspy.Signature):
    """You will be given two trajectories and the current instructions
    for each module. Generate search/replace edits to improve them.

    Each edit is {"search": "...", "replace": "..."}.
    Empty search = append. Replace with "" = delete.
    Keep instructions concise. Remove redundant rules."""

    program_code: str = InputField()
    modules_defn: str = InputField()
    current_instructions: str = InputField(
        desc="Current instructions per module as JSON dict")
    program_inputs: str = InputField()
    oracle_metadata: str = InputField()
    worse_program_trajectory: str = InputField()
    worse_program_outputs: str = InputField()
    worse_reward_value: float = InputField()
    worse_reward_info: str = InputField()
    better_program_trajectory: str = InputField()
    better_program_outputs: str = InputField()
    better_reward_value: float = InputField()
    better_reward_info: str = InputField()
    module_names: list[str] = InputField()
    discussion: str = OutputField(
        desc="Analyze what went wrong and which rules to change")
    module_edits: dict[str, list[dict[str, str]]] = OutputField(
        desc='Per module: [{"search": "...", "replace": "..."}] edits'
    )


def _edit_rules_impl(bucket, system, **kwargs):
    predictor2name = kwargs["predictor2name"]
    batch_10p_score = kwargs["batch_10p_score"]
    batch_90p_score = kwargs["batch_90p_score"]
    prompt_model = kwargs["prompt_model"] or dspy.settings.lm

    module_names = [name for name, _ in system.named_predictors()]
    good, bad = dict(bucket[0]), dict(bucket[-1])
    example = good["example"]

    if good["score"] <= batch_10p_score or bad["score"] >= batch_90p_score:
        logger.info("Skipping rule editing: insufficient score contrast.")
        return False

    if good["score"] <= bad["score"]:
        if good["score"] > batch_90p_score:
            bad["trace"], bad["score"] = [], "N/A"
            bad["prediction"] = {"N/A": "N/A"}
        else:
            good["trace"], good["score"] = [], "N/A"
            good["prediction"] = {"N/A": "N/A"}

    better_trajectory = [
        {"module_name": predictor2name[id(p)], "inputs": i,
         "outputs": dict(o)} for p, i, o in good["trace"]
    ]
    worse_trajectory = [
        {"module_name": predictor2name[id(p)], "inputs": i,
         "outputs": dict(o)} for p, i, o in bad["trace"]
    ]

    current_instructions = {
        name: pred.signature.instructions
        for name, pred in system.named_predictors()
    }

    ek = _build_edit_kwargs(
        system, example, better_trajectory, worse_trajectory,
        good, bad, module_names, current_instructions)
    with dspy.settings.context(trace=[], lm=prompt_model):
        edits = dspy.Predict(EditRules)(**ek).module_edits
    return _apply_edits(system, edits)


def _build_edit_kwargs(system, example, better_traj, worse_traj,
                       good, bad, module_names, cur_instr):
    d = {
        "program_code": inspect.getsource(system.__class__),
        "modules_defn": inspect_modules(system),
        "current_instructions": cur_instr,
        "program_inputs": {**example.inputs()},
        "oracle_metadata": {**example.labels()},
        "better_program_trajectory": better_traj,
        "better_program_outputs": dict(good["prediction"]),
        "worse_program_trajectory": worse_traj,
        "worse_program_outputs": dict(bad["prediction"] or {}),
        "worse_reward_value": bad["score"],
        "better_reward_value": good["score"],
        "worse_reward_info": bad["output_metadata"],
        "better_reward_info": good["output_metadata"],
        "module_names": module_names,
    }
    return {k: v if isinstance(v, str) else orjson.dumps(
        recursive_mask(v), option=orjson.OPT_INDENT_2).decode()
        for k, v in d.items()}


def _apply_edits(system, edits):
    applied = False
    for name, predictor in system.named_predictors():
        if name not in edits:
            continue
        edit_list = edits[name]
        instr = predictor.signature.instructions
        if isinstance(edit_list, str):
            logger.info(f"Edit {name} (append): {edit_list[:80]}")
            instr += "\n\n" + edit_list
            applied = True
        elif isinstance(edit_list, list):
            instr, applied = _apply_edit_list(
                name, instr, edit_list, applied)
        predictor.signature = predictor.signature.with_instructions(instr)
    return applied


def _apply_edit_list(name, instr, edit_list, applied):
    for edit in edit_list:
        if not isinstance(edit, dict):
            continue
        search = edit.get("search", "")
        replace = edit.get("replace", "")
        if search and search in instr:
            logger.info(f"Edit {name}: [{search[:50]}]→[{replace[:50]}]")
            instr = instr.replace(search, replace, 1)
            applied = True
        elif replace:
            if search:
                logger.info(f"Edit {name}: miss, appending: {replace[:60]}")
            else:
                logger.info(f"Edit {name} (append): {replace[:60]}")
            instr += "\n\n" + replace
            applied = True
    return instr, applied
