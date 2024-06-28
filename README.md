# Eye Tracking Game 👀


I was looking through my old projects and found a stereo vision project, so I got inspired to start a new Computer Vision project.  
I wanted something that didn't need special hardware (I don't have any stereo cameras lying around hehe..) and after a bit of brainstorming I thught about gaze tracking using a webcam.

The program relies on Roboflow's inference library, I've started from this tutorial:  
[James Gallagher. (Sep 22, 2023). Gaze Detection and Eye Tracking: A How-To Guide. Roboflow](https://blog.roboflow.com/gaze-direction-position/).

The goal is to track the user's gaze on the screen.

## 👩‍💻 Local Setup

To set up this project locally, follow these steps:

1. Clone the repository:
    ```
    git clone https://github.com/saraelhark/eye-tracking-game.git
    ```

2. Navigate to the project directory:
    ```
    cd eye-tracking-game
    ```

3. Install the required dependencies:
    ```
    pip install -r requirements.txt
    ```

4. Download Roboflow's inference container (choose gpu or cpu) and run it:
    ```
    docker pull roboflow/roboflow-inference-server-gpu
    docker run -p 9001:9001 roboflow/roboflow-inference-server-gpu
    ```

5. Run the application:
    ```
    python main.py
    ```


## 🎢 Deployment Setup (TBD)

At some point I want to deploy this on my website, but still need to figure out how.


## 🎯 Accuracy
I've set the resolution to 640x480 to not add too much overhead with larger frames.

To improve accuracy:  
1. I added a step to get the face position in the center of the frame, to get more uniform results.

2. There is a 5 point calibration: each point is captured 4 times and the averages are used to calculate the transformation matrix.


<img src="https://github.com/saraelhark/eye-tracking-game/assets/41480355/60fa3efe-e308-4c78-a608-14093ec1ad74" width="427" height="320">

3. I tested different filtering techniques.  

With the class `CheckGazeAccuracyForTarget` it's possible to calculate accuracy, adding different filtering techniques.  
The user needs to look at target points for a specified amount of time and then the normalised distance (considering the frame size) from the acquired gaze points and the target point is calculated.  


<img src="https://github.com/saraelhark/eye-tracking-game/assets/41480355/5a90e105-3103-4a17-ac96-0f55fadcf49a" width="427" height="320">

Here are some preliminary results I got:

|          | Accuracy |
|----------|----------|
| Baseline (no filter) |   86%    |
| Moving Average |   92%    |
| Median Filter |   90%    |
| Adaptive Moving Average |   96%    |
| Kalman Filter |   92%    |

The Kalman Filter tuning can be optimized.


## ✨ Demo

Here is a demo of the game, (using the Kalman Filter for filtering):


<img src="https://github.com/saraelhark/eye-tracking-game/assets/41480355/154dc427-bf2f-4e85-9dd1-1d3a1c3ce3ce" width="427" height="320">

The video has low fps (due to the slow processing on my machine), which adds a delay on the gaze dot movements.
The accuracy is not great: the head position is important for gaze prediction but looking at a small frame 640x480 doesn't require moving your head.


## 🤝 Contributions
Contributions to this project are welcome. If you have any suggestions or feedback, please feel free to open an issue or submit a pull request.
