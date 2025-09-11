import os
import subprocess
import threading
import time


class Worker(threading.Thread):
    def __init__(self, sid, gpu, task_provider, log_path):
        threading.Thread.__init__(self)
        self.gpu = gpu
        self.tp = task_provider
        self.id = sid
        self.log_path = log_path
        self.task_nr = 0

    def run(self):
        while self.tp.more_tasks():
            self.task_nr += 1
            cmd = self.tp.get_task()
            if cmd is not None:
                t = time.time()
                print(f"Runner-{self.id}: {' '.join(cmd)}")
                try:
                    output = subprocess.check_output(cmd, env=dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID",
                                                                   CUDA_VISIBLE_DEVICES=f'{self.gpu}'), text=True,
                                                     stderr=subprocess.STDOUT)
                except subprocess.CalledProcessError as e:
                    output = e.output
                f = open(f'{self.log_path}/{self.id}_{self.task_nr}.log', "w")
                f.write(f"Runner-{self.id}: {' '.join(cmd)}")
                f.write(str(output))
                t2 = time.time()
                f.write(f"Runner-{self.id} Execution time: {int((t2 - t) / 36) / 100} hours")
                print(f"Runner-{self.id} Execution time: {int((t2 - t) / 36) / 100} hours")
                f.close()


class TaskManager:
    def __init__(self, tasks):
        self.lock = threading.Lock()

        print("No. of tasks: {}".format(len(tasks)))
        self.tasks = tasks
        self.idx = 0

    def more_tasks(self):
        return self.idx < len(self.tasks)

    def get_task(self):
        # lock
        with self.lock:
            if self.idx >= len(self.tasks):
                return None
            else:
                t = self.tasks[self.idx]
                self.idx += 1
                print(f"Dispatched task {self.idx} / {len(self.tasks)}")
                return t
