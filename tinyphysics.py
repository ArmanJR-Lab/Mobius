"""
Tiny physics simulator for lateral steering control.
"""

import os
import urllib.request
import zipfile
from collections import namedtuple
from hashlib import md5
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import onnxruntime as ort
import pandas as pd

ACC_G = 9.81
FPS = 10
CONTROL_START_IDX = 100
COST_END_IDX = 500
CONTEXT_LENGTH = 20
VOCAB_SIZE = 1024
LATACCEL_RANGE = [-5, 5]
STEER_RANGE = [-2, 2]
MAX_ACC_DELTA = 0.5
DEL_T = 0.1
LAT_ACCEL_COST_MULTIPLIER = 50.0

FUTURE_PLAN_STEPS = FPS * 5  # 5 secs

State = namedtuple("State", ["roll_lataccel", "v_ego", "a_ego"])
FuturePlan = namedtuple("FuturePlan", ["lataccel", "roll_lataccel", "v_ego", "a_ego"])

DATASET_URL = "https://huggingface.co/datasets/commaai/commaSteeringControl/resolve/main/data/SYNTHETIC_V0.zip"
DATASET_PATH = Path(__file__).resolve().parent / "data"


class LataccelTokenizer:
    def __init__(self):
        self.vocab_size = VOCAB_SIZE
        self.bins = np.linspace(LATACCEL_RANGE[0], LATACCEL_RANGE[1], self.vocab_size)

    def encode(
        self, value: Union[float, np.ndarray, List[float]]
    ) -> Union[int, np.ndarray]:
        value = self.clip(value)
        return np.digitize(value, self.bins, right=True)

    def decode(self, token: Union[int, np.ndarray]) -> Union[float, np.ndarray]:
        return self.bins[token]

    def clip(
        self, value: Union[float, np.ndarray, List[float]]
    ) -> Union[float, np.ndarray]:
        return np.clip(value, LATACCEL_RANGE[0], LATACCEL_RANGE[1])


class TinyPhysicsModel:
    def __init__(self, model_path: str, debug: bool = False) -> None:
        self.tokenizer = LataccelTokenizer()
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.log_severity_level = 3
        provider = "CPUExecutionProvider"

        with open(model_path, "rb") as f:
            self.ort_session = ort.InferenceSession(f.read(), options, [provider])

    def softmax(self, x, axis=-1):
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / np.sum(e_x, axis=axis, keepdims=True)

    def predict(self, input_data: dict, temperature=1.0) -> int:
        res = self.ort_session.run(None, input_data)[0]
        probs = self.softmax(res / temperature, axis=-1)
        # we only care about the last timestep (batch size is just 1)
        assert probs.shape[0] == 1
        assert probs.shape[2] == VOCAB_SIZE
        sample = np.random.choice(probs.shape[2], p=probs[0, -1])
        return sample

    def get_current_lataccel(
        self, sim_states: List[State], actions: List[float], past_preds: List[float]
    ) -> float:
        tokenized_actions = self.tokenizer.encode(past_preds)
        raw_states = [list(x) for x in sim_states]
        states = np.column_stack([actions, raw_states])
        input_data = {
            "states": np.expand_dims(states, axis=0).astype(np.float32),
            "tokens": np.expand_dims(tokenized_actions, axis=0).astype(np.int64),
        }
        return self.tokenizer.decode(self.predict(input_data, temperature=0.8))


class TinyPhysicsSimulator:
    def __init__(
        self, model: TinyPhysicsModel, data_path: str, controller, debug: bool = False
    ) -> None:
        self.data_path = data_path
        self.sim_model = model
        self.data = self.get_data(data_path)
        self.controller = controller
        self.debug = debug
        self.reset()

    def reset(self) -> None:
        self.step_idx = CONTEXT_LENGTH
        state_target_futureplans = [
            self.get_state_target_futureplan(i) for i in range(self.step_idx)
        ]
        self.state_history = [x[0] for x in state_target_futureplans]
        self.action_history = (
            self.data["steer_command"].values[: self.step_idx].tolist()
        )
        self.current_lataccel_history = [x[1] for x in state_target_futureplans]
        self.target_lataccel_history = [x[1] for x in state_target_futureplans]
        self.target_future = None
        self.current_lataccel = self.current_lataccel_history[-1]
        seed = int(md5(self.data_path.encode()).hexdigest(), 16) % 10**4
        np.random.seed(seed)

    def get_data(self, data_path: str) -> pd.DataFrame:
        df = pd.read_csv(data_path)
        processed_df = pd.DataFrame(
            {
                "roll_lataccel": np.sin(df["roll"].values) * ACC_G,
                "v_ego": df["vEgo"].values,
                "a_ego": df["aEgo"].values,
                "target_lataccel": df["targetLateralAcceleration"].values,
                "steer_command": -df[
                    "steerCommand"
                ].values,  # steer commands are logged with left-positive convention but this simulator uses right-positive
            }
        )
        return processed_df

    def sim_step(self, step_idx: int) -> None:
        pred = self.sim_model.get_current_lataccel(
            sim_states=self.state_history[-CONTEXT_LENGTH:],
            actions=self.action_history[-CONTEXT_LENGTH:],
            past_preds=self.current_lataccel_history[-CONTEXT_LENGTH:],
        )
        pred = np.clip(
            pred,
            self.current_lataccel - MAX_ACC_DELTA,
            self.current_lataccel + MAX_ACC_DELTA,
        )
        if step_idx >= CONTROL_START_IDX:
            self.current_lataccel = pred
        else:
            self.current_lataccel = self.get_state_target_futureplan(step_idx)[1]

        self.current_lataccel_history.append(self.current_lataccel)

    def control_step(self, step_idx: int) -> None:
        action = self.controller.update(
            self.target_lataccel_history[step_idx],
            self.current_lataccel,
            self.state_history[step_idx],
            future_plan=self.futureplan,
        )
        if step_idx < CONTROL_START_IDX:
            action = self.data["steer_command"].values[step_idx]
        action = np.clip(action, STEER_RANGE[0], STEER_RANGE[1])
        self.action_history.append(action)

    def get_state_target_futureplan(
        self, step_idx: int
    ) -> Tuple[State, float, FuturePlan]:
        state = self.data.iloc[step_idx]
        return (
            State(
                roll_lataccel=state["roll_lataccel"],
                v_ego=state["v_ego"],
                a_ego=state["a_ego"],
            ),
            state["target_lataccel"],
            FuturePlan(
                lataccel=self.data["target_lataccel"]
                .values[step_idx + 1 : step_idx + FUTURE_PLAN_STEPS]
                .tolist(),
                roll_lataccel=self.data["roll_lataccel"]
                .values[step_idx + 1 : step_idx + FUTURE_PLAN_STEPS]
                .tolist(),
                v_ego=self.data["v_ego"]
                .values[step_idx + 1 : step_idx + FUTURE_PLAN_STEPS]
                .tolist(),
                a_ego=self.data["a_ego"]
                .values[step_idx + 1 : step_idx + FUTURE_PLAN_STEPS]
                .tolist(),
            ),
        )

    def step(self) -> None:
        state, target, futureplan = self.get_state_target_futureplan(self.step_idx)
        self.state_history.append(state)
        self.target_lataccel_history.append(target)
        self.futureplan = futureplan
        self.control_step(self.step_idx)
        self.sim_step(self.step_idx)
        self.step_idx += 1

    def compute_cost(self) -> Dict[str, float]:
        target = np.array(self.target_lataccel_history)[CONTROL_START_IDX:COST_END_IDX]
        pred = np.array(self.current_lataccel_history)[CONTROL_START_IDX:COST_END_IDX]

        lat_accel_cost = np.mean((target - pred) ** 2) * 100
        jerk_cost = np.mean((np.diff(pred) / DEL_T) ** 2) * 100
        total_cost = (lat_accel_cost * LAT_ACCEL_COST_MULTIPLIER) + jerk_cost
        return {
            "lataccel_cost": lat_accel_cost,
            "jerk_cost": jerk_cost,
            "total_cost": total_cost,
        }

    def rollout(self) -> Dict[str, float]:
        for _ in range(CONTEXT_LENGTH, len(self.data)):
            self.step()
        return self.compute_cost()


def download_dataset():
    print("Downloading dataset (0.6G)...")
    DATASET_PATH.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(DATASET_URL) as resp:
        with zipfile.ZipFile(BytesIO(resp.read())) as z:
            for member in z.namelist():
                if not member.endswith("/"):
                    with (
                        z.open(member) as src,
                        open(DATASET_PATH / os.path.basename(member), "wb") as dest,
                    ):
                        dest.write(src.read())
