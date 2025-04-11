from __future__ import absolute_import
from __future__ import print_function

import os
import datetime
from shutil import copyfile

from training_simulation import Simulation
from generator import TrafficGenerator
from memory import Memory
from model import TrainModel
from visualization import Visualization
from utils import import_train_configuration, set_sumo, set_train_path


if __name__ == "__main__":

    print(os.getcwd())  # Print the current working directory
    project_root = os.path.dirname(os.path.dirname(__file__))  # go up from /src to root
    config_path = os.path.join(project_root, 'configs', 'training_settings.ini')
    config = import_train_configuration(config_file=config_path)
    sumo_cmd = set_sumo(config['gui'], config['sumocfg_file_name'], config['max_steps'])
    path = set_train_path(config['models_path_name'])

    Model = TrainModel(
        config['num_layers'], 
        config['width_layers'], 
        config['batch_size'], 
        config['learning_rate'], 
        input_dim=config['num_states'], 
        output_dim=config['num_actions']
    )

    Memory = Memory(
        config['memory_size_max'], 
        config['memory_size_min']
    )

    TrafficGen = TrafficGenerator(
        config['max_steps'], 
        config['n_cars_generated']
    )

    Visualization = Visualization(
        path, 
        dpi=96
    )
    
    print("🔁 Target Update Freq from config:", config['target_update_freq'])
       
    Simulation = Simulation(
        Model,
        Memory,
        TrafficGen,
        sumo_cmd,
        config['gamma'],
        config['max_steps'],
        config['green_duration'],
        config['yellow_duration'],
        config['num_states'],
        config['num_actions'],
        config['training_epochs'],
        config['target_update_freq'],
        config['num_layers'],              # 🆕
        config['width_layers'],            # 🆕
        config['batch_size'],              # 🆕
        config['learning_rate']            # 🆕
    )
    
    episode = 0
    timestamp_start = datetime.datetime.now()
    
    while episode < config['total_episodes']:
        print(f'\n----- Episode {episode+1} of {config["total_episodes"]} -----')
        # epsilon = 1.0 - (episode / config['total_episodes'])
        epsilon = max(0.05, 1.0 - episode / config['total_episodes'])

        
        simulation_time, training_time = Simulation.run(episode, epsilon)
        
        print(f'Episode {episode+1} Summary:')
        print(f'Simulation Time: {simulation_time} s')
        print(f'Training Time: {training_time} s')
        print(f'Total Episode Time: {round(simulation_time+training_time, 1)} s')
        print(f'Epsilon: {epsilon:.2f}')
        
        episode += 1

    print("\n----- Start time:", timestamp_start)
    print("----- End time:", datetime.datetime.now())
    print("----- Session info saved at:", path)

    Model.save_model(path)

    copyfile(
    src=os.path.join("configs", "training_settings.ini"),
    dst=os.path.join(path, "training_settings.ini")
    )

    Visualization.save_data_and_plot(data=Simulation._target_sync_steps,filename='target_syncs',xlabel='Sync Event Index',ylabel='Episode')
    Visualization.save_data_and_plot(data=Simulation.reward_store, filename='reward', xlabel='Episode', ylabel='Cumulative negative reward')
    Visualization.save_data_and_plot(data=Simulation.cumulative_wait_store, filename='delay', xlabel='Episode', ylabel='Cumulative delay (s)')
    Visualization.save_data_and_plot(data=Simulation.avg_queue_length_store, filename='queue', xlabel='Episode', ylabel='Average queue length (vehicles)')