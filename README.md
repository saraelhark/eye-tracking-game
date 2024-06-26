# Eye Tracking Game 👀

## 📚 Summary 
I was looking through my old projects and found a Stereo Vision project, so I go inspired to start a new Computer Vision project. 
I wanted something that didn't need special hardware and after a bit of brainstorming I thught about gaze-tracking using a laptop's webcam.

The program relies on Roboflow's inference library, I've started from [this tutorial](https://blog.roboflow.com/gaze-direction-position/).

## 👩‍💻 Local Setup (TBD)


## 🎢 Deployment Setup (TBD)


## 🎯 Accuracy
To get more uniform results I added a step to get the face position in the center of the frame.

Then there is a 5 point calibration: each point is captured 4 times and the averages are used to calculate the transformation matrix.

(TBD add GIF)

With the '''CheckGazeAccuracyForTarget''' the accuracy is calculated after calibration, adding different filtering techniques.

|          | Accuracy |
|----------|----------|
| Baseline |   86%    |
| Moving Average |   92%    |
| Median Filter |   90%    |
| Adaptive Moving Average |   96%    |
| Kalman Filter |   92%    |

(TBD add GIF)

But calculating accuracy with a fixed point is misleading if the goal is to get a smooth the gaze track.

## ✨ Demo
Here is a small demo of the program in action:
(TBD add GIF)

## 📆 Future Improvements 

- [ ] use mouse click instead of spacebar for calibration, maybe other things too
- [ ] make program work with any webcam resolution
- [ ] add a check for light conditions
- [ ] add face position check in the background of all steps
- [ ] improve accuracy
- [ ] add deployment instructions
