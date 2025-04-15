# Traffic Light Control System

A reinforcement learning framework using Deep Q-Learning to optimize traffic signal timing at intersections. This system uses SUMO (Simulation of Urban MObility) to simulate traffic flow and a neural network to learn optimal light phase selections that minimize vehicle wait times.

![image](https://github.com/user-attachments/assets/3049664b-4acc-4a12-bcd0-4a32abd8617b)


## Key Features

- **Deep Q-Learning Implementation**: Neural network-based reinforcement learning for traffic light control
- **SUMO Integration**: Realistic traffic simulation with customizable parameters
- **Comprehensive Metrics**: Queue length, waiting time, and reward tracking
- **Visualization Tools**: Performance analysis graphs and simulation visualization
- **Flexible Configuration**: Easily adjustable training and testing parameters

![image](https://github.com/user-attachments/assets/dfca3ffa-44db-44c0-ab6c-a2cb08205795)


## System Requirements

- Python 3.10 (recommended for TensorFlow compatibility)
- SUMO 1.10.0 or higher
- TensorFlow 2.10.0
- 4GB RAM minimum (8GB+ recommended for larger simulations)
- GPU acceleration recommended for faster training

## Project Structure

- **configs/**: Configuration files for training and testing
- **intersection/**: SUMO network and configuration files
- **models/**: Saved models and training results
- **src/**: Source code
  - **generator.py**: Traffic generation for simulation
  - **model.py**: Neural network architecture
  - **training_main.py**: Main script for training
  - **testing_main.py**: Main script for testing
  - **visualization.py**: Performance visualization tools

## Configuration Options

Key parameters in `training_settings.ini` and `testing_settings.ini`:

### Simulation Parameters
- `gui`: Enable/disable SUMO GUI (true/false)
- `total_episodes`: Number of training episodes
- `max_steps`: Maximum simulation steps per episode
- `n_cars_generated`: Number of vehicles in simulation
- `green_duration`/`yellow_duration`: Traffic light timing in seconds

![image](https://github.com/user-attachments/assets/e17ceada-efab-4df2-8028-b0e6c77889f7)


### Model Parameters
- `num_layers`: Neural network depth
- `width_layers`: Neurons per hidden layer
- `learning_rate`: Training rate for optimization
- `batch_size`: Training batch size
- `gamma`: Discount factor for future rewards
![image](https://github.com/user-attachments/assets/d93b2fec-ca38-40cf-ba11-213a5f9c0d9e)


## Interpreting Results

After training/testing, the system generates:

1. **Reward Plots**: Shows cumulative rewards per episode - higher (less negative) values indicate better performance
2. **Queue Length Plots**: Shows average number of waiting vehicles - downward trends indicate improved traffic flow
3. **Delay Plots**: Shows cumulative waiting time - lower values mean reduced delays

![image](https://github.com/user-attachments/assets/7260eecf-151b-4c10-9a4e-b3c413280a8b)


## Advanced Usage

### Custom Traffic Patterns
Modify `generator.py` to create different traffic distribution patterns:
- Adjust the Weibull distribution parameters
- Change the proportion of turning vs. straight-moving vehicles
- Create rush-hour scenarios by clustering vehicle generation
![image](https://github.com/user-attachments/assets/abce08de-ecc0-4c8c-b40c-53bb8280575d)

![image](https://github.com/user-attachments/assets/bdd7d8ff-6ca2-498b-88be-daf8504157e5)

![image](https://github.com/user-attachments/assets/f7f247b7-5b5d-4391-b7bb-6c1f15894ec3)


### Neural Network Tuning
Modify `model.py` to experiment with different architectures:
- Change activation functions
- Add dropout layers for regularization
- Implement different optimization algorithms

![image](https://github.com/user-attachments/assets/8b006914-d35c-43da-9526-ea7a119427bc)


# Traffic Light Control System: Setup Guide

This guide walks you through setting up and running the Traffic Light Control reinforcement learning project. The system uses Deep Q-Learning to optimize traffic signal timing at intersections through SUMO (Simulation of Urban MObility) and TensorFlow.

## Prerequisites

- Python 3.10 (recommended, compatible with TensorFlow)
- SUMO traffic simulator
- Git

## Python Installation

- Download python 3.10 from [here](https://www.python.org/downloads/release/python-3100/)
- If you have windows accordingly select which verison
  
![image](https://github.com/user-attachments/assets/8c0d35da-503a-4343-a5f1-f020f62149c8)

- Launch the installer

  -  Remember to check the option to "Add python.exe to PATH" and install
  
![image](https://github.com/user-attachments/assets/ebee444a-563c-4721-9e57-2e40d6065744)

- Now check if Python 3.10 is installed in your system
  ```bash
  py -0
  ```
OR
  ```bash
  python --version
  ```
  ![image](https://github.com/user-attachments/assets/5ae8132e-fe2e-428a-b40e-2b60f01eb377)

NOTE: Make sure you are using Python 3.10 


## SUMO installation

## Step 1: Install SUMO

SUMO is essential for traffic simulation. Install it before proceeding:

### Windows
1. Download the latest SUMO version from [SUMO's official website](https://sumo.dlr.de/releases/1.2.0/)
2. Select the zip option:

![image](https://github.com/user-attachments/assets/c26ecafe-8443-497e-8e44-6f060af2c12e)

3. Add SUMO to your PATH environment variable:
   - Search for "Environment Variables" in Windows search
  
   - In you User Variables - set `SUMO_HOME` environment variable to: `C:\path to where you extracted your SUMO directory \sumo-win64-1.2.0\sumo-1.2.0`

     ![image](https://github.com/user-attachments/assets/a1a04b05-9056-447f-a55b-7f5ac685ec45)

       
   - Now again in User variables edit the "Path" and add: `C:\path to where you extracted your SUMO directory \sumo-1.2.0\bin`
     
    ![image](https://github.com/user-attachments/assets/51c1cbfc-06d9-4236-91db-78798c61530b)

   - Now edit the System Variables - "Path" and add: `C:\path to where you extracted your SUMO directory \sumo-1.2.0\bin`
    
   ![image](https://github.com/user-attachments/assets/238c62d2-0e8f-4180-8662-d55b8de02e30)
   


  
   - Now restart your system to take effect of those variables

3. Now check if SUMO is installed properly

   ```bash
    sumo --version
   ```
### macOS
```bash
brew install --cask sumo
export SUMO_HOME="/usr/local/opt/sumo/share/sumo"
```

### Linux (Ubuntu/Debian)
```bash
sudo add-apt-repository ppa:sumo/stable
sudo apt-get update
sudo apt-get install sumo sumo-tools sumo-doc
export SUMO_HOME="/usr/share/sumo"
```

## Step 2: Clone the Repository

NOTE: Make sure you clone the repository and SUMO extracted at same root directory

```bash
git clone https://github.com/dheeraj3choudhary/Traffic_Light_Control_System.git
cd Traffic_Light_Control_System
```

## Step 3: Create a Virtual Environment

```bash
# Create virtual environment
py -3.10 -m venv venv
```
OR 

if there is only one python version in your system then (which we installed earlier that is Python 3.10)

```bash
python -m venv venv
```

# Activate virtual environment

# On Windows:
```bash
venv\Scripts\activate
```

# On macOS/Linux:
```bash
source venv/bin/activate
```

## Step 4: Install Dependencies

```bash
pip install tensorflow==2.10.0
pip install -r updated_requirements.txt
```

Only reason to install these dependencies separate to mitigate the conflicts 

## Step 5: Project Structure Verification

Ensure your project structure looks like this:
```
Traffic_Light_Control_System/
├── configs/
│   ├── testing_settings.ini
│   └── training_settings.ini
├── intersection/
│   ├── environment.net.xml
│   └── sumo_config.sumocfg
├── models/
│   └── .gitkeep
├── src/
│   ├── generator.py
│   ├── memory.py
│   ├── model.py
│   ├── testing_main.py
│   ├── testing_simulation.py
│   ├── training_main.py
│   ├── training_simulation.py
│   ├── utils.py
│   └── visualization.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Step 6: Training a Model(Optional since we already trained the model with best performance)

Training a model generates the traffic signal control policy:

```bash
# Navigate to the src directory
cd src

# Run the training script
python training_main.py
```

This will execute training with parameters specified in `configs/training_settings.ini`. The training process may take several hours depending on your computer's specifications.

During training, you'll see:
- Episodes being processed
- Rewards for each episode
- Training times

Models will be saved in the `models/` directory.

## Step 7: Testing a Model(Actually simulates the traffic intersection with the optimal model setting already applied)
NOTE: if you just want to see the simulation just skip to step 2. below by running the testing_main.py 

Once training is complete, test the model to see how well it performs:

1. Modify `configs/testing_settings.ini` to specify which model to test:
   - Set `model_to_test` to match the model number you want to evaluate (usually the highest number in the `models/` directory)
   - Set `gui = True` to visualize the simulation

2. Run the testing script(which is inside src/ ):
   
```bash
python .src/testing_main.py
```

This will open SUMO-GUI with your trained model controlling the traffic lights. You can observe how the system performs with the trained reinforcement learning model.

-  Click on the run button to start the traffic simulation

![image](https://github.com/user-attachments/assets/34e69beb-eb97-4113-8666-8caf452027cd)


https://github.com/user-attachments/assets/50262a4b-c592-420a-a6e5-b1607383be22


## Step 8: Understanding the Visualization

During the SUMO-GUI visualization:
- Red/yellow/green lights show the current traffic signal states
- Vehicles will spawn according to the simulation settings
- The trained model decides when to change traffic light phases

After testing completes, performance metrics will be saved in the `models/model_X/test/` directory, including:
- Queue length plots
- Reward plots
- Detailed data in text files

## Troubleshooting

### SUMO_HOME not found
If you encounter a "please declare environment variable 'SUMO_HOME'" error:
1. Make sure SUMO is properly installed
2. Ensure the SUMO_HOME environment variable is set correctly for your operating system

### TensorFlow compatibility issues
If you encounter TensorFlow errors:
1. Try using Python 3.8-3.10 which are better supported by TensorFlow
2. Install the specific TensorFlow version mentioned: `pip install tensorflow==2.10.0`

### ImportError when running scripts
Make sure you're running scripts from the correct directory. Always run the scripts from the `src/` directory.

## Next Steps

- Try modifying the `configs/training_settings.ini` parameters to see their effect on training
- Experiment with different neural network architectures in `model.py`
- Generate different traffic patterns by modifying `generator.py`

## Acknowledgements

This project implements concepts from recent research in reinforcement learning for traffic control systems, particularly building on work in deep Q-learning applications for transportation systems.

## License

MIT License - Feel free to use and modify this code for academic, research, or commercial purposes with appropriate attribution.

For additional information, refer to the SUMO documentation at [https://sumo.dlr.de/docs/](https://sumo.dlr.de/docs/) and the TensorFlow documentation at [https://www.tensorflow.org/guide](https://www.tensorflow.org/guide).

## Note this implementation guide is only implemented on windows personally for development and testing.
