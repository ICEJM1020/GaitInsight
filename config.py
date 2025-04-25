import os
import yaml


class Config:
    def __init__(self):
        # Default values for parameters

        ## Basic
        self.work_dir = os.path.dirname(os.path.abspath(__file__))
        self.phase = 'train'  # must be train or test
        self.device = [0]
        self.action_type = "gait"

        ## Log
        self.save_score = False  # if true, the classification score will be stored
        self.seed = 1   # random seed for pytorch
        self.log_interval = 1000  # the interval for printing messages (#iteration)
        self.save_interval = 10  # the interval for storing models (#iteration)
        self.eval_interval = 1  # the interval for evaluating models (#iteration)
        self.print_log = True  # print logging or not
        self.show_topk = [1, 5]  # which Top K accuracy will be shown

        ## DataLoader
        self.feeder = 'st_gcn.feeder.Feeder'
        self.num_worker = 8
        self.train_feeder_args = {
        }
        self.test_feeder_args = {
        }
        self.shuffle = True
        self.batch_size = 256
        self.test_batch_size = 256

        ## Model
        self.model = None  # the model will be used
        self.model_args = {

        }
        self.weights = None  # the weights for network initialization
        self.ignore_weights = []  # the name of weights which will be ignored in the initialization
        self.type = None

        ## Training
        self.base_lr = 0.01  # initial learning rate
        self.step = [20, 40, 60]  # the epoch where optimizer reduce the learning rate
        self.optimizer = 'SGD'  # 'type of optimizer'
        self.nesterov = False  # use nesterov or not
        self.start_epoch = 0  # start training from which epoch
        self.num_epoch = 100  # stop training in which epoch
        self.global_step = 0  # current global step for plotting
        self.weight_decay = 0.0005  # weight decay for optimizer
        self.save_mode = "best"  #  'best' will record best model, 'regular' will record per 'save_interval'
        self.base_acc = 0.85

    def load_config(self, config_path):
        """
        Update the class attributes based on a YAML configuration file.
        :param config_path: Path to the YAML configuration file.
        """
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)

            # Update class attributes with values from the config file
            for key, value in config_data.items():
                if hasattr(self, key):
                    setattr(self, key, self.str2bool(value))
                else:
                    print(f"Warning: Config key '{key}' is not recognized and will be ignored.")
        except FileNotFoundError:
            print(f"Error: Configuration file '{config_path}' not found.")
        except yaml.YAMLError as e:
            print(f"Error: Failed to parse YAML configuration file. {e}")
    

    @staticmethod
    def str2bool(v):
        # Helper function to handle boolean argument parsing
        if isinstance(v, str):
            if v.lower() in ('yes', 'true', 't', 'y'):
                return True
            elif v.lower() in ('no', 'false', 'f', 'n'):
                return False
            else:
                return v
        else:
            return v

CONFIG = Config()

# # Example usage
# if __name__ == "__main__":
#     config = Config()
#     config.load_config("Graph/st-gcn-master_2/config/st_gcn/kinetics-skeleton/train.yaml")
