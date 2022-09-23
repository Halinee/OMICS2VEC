import os
import os.path as osp
import time
from typing import Any, Dict, List, Optional, Union


class Task:
    def __init__(self, task_name: str, config: dict, **kwargs):
        super(Task, self).__init__()
        self.task_name = task_name
        self.config = config
        self.valid_file_num = len(self.config["data_type"])
        self.params = self.set_configuration()
        self.save_file_path = osp.join(
            self.params["save_path"], self.config["experiment"]
        )
        if not osp.exists(self.save_file_path):
            os.mkdir(self.save_file_path)

    def set_configuration(self) -> Dict[str, Any]:
        print("Set", self.task_name, "task parameters...")
        if self.task_name == "Preprocess":
            # Consider the label file
            self.valid_file_num += 1
            return self.config["preprocess_params"]
        elif self.task_name == "Train":
            return self.config["training_params"]
        elif self.task_name == "Embed":
            return self.config["embedding_params"]
        elif self.task_name == "Analysis":
            return self.config["analysis_params"]
        else:
            ValueError("Invalid process type.")

    def update_configuration(
        self,
        config: Dict[str, Any],
        key: Union[str, List[str]],
        value: Optional[Any] = None,
    ) -> Dict[str, Any]:
        update_config = config
        if not isinstance(key, List):
            key = [key]
            value = [value]
        for key, value in zip(key, value):
            update_config[key] = value
            print(
                self.task_name,
                "task parameter value has changed.\n",
                key,
                ":",
                value,
            )

        return update_config

    def start_task(self) -> float:
        print(self.task_name, "step start...")
        start = time.time()
        return start

    def check_task(self) -> bool:
        print("Check the existence of the file...")
        if self.task_name == "Analysis":
            return False
        if len(os.listdir(self.save_file_path)) == self.valid_file_num:
            print(self.task_name, "file already exist:", self.save_file_path)
            print("Proceed to the next step...")
            if self.task_name == "Preprocess":
                return False
            return True
        else:
            non_exist_num = self.valid_file_num - len(os.listdir(self.save_file_path))
            print(self.task_name, f"{non_exist_num} file does not exist!")
            return False

    def run_task(self) -> Any:
        return NotImplementedError

    def end_task(self, start: float) -> None:
        print("Saved in", self.save_file_path)
        print(self.task_name, "step end...")
        print("Running time :", time.time() - start, "sec\n")

    def processing(self) -> None:
        start = self.start_task()
        if not self.check_task():
            self.run_task()
        self.end_task(start)
