# Hand-Gestures-Recognition

A Pattern Recognition Machine Learning based on vector projection and euclidian distance, with the maximal hand processed of: 1

This project uses the following library each for its own purposes:

* MediaPipe (HandLandmark Detection Model) : Detect and track hands
* OpenCV : Opening the camera and process each frame live, feeding it to MediaPipe
* Numpy : Process matrices and vector calculation
* Pathlib : Generate/open savefiles

# System

The system first loads and initiates every single library needed for processing.

`cam = cv.VideoCapture(0)`

This initializes OpenCV system telling it to process and set up the camera to open

MediaPipe is a huge library specialized to "see" and "hear" in real-time, such as tracking or recognizing facial features. Google offers a single library to process these data's, but the thing on what we want the library to process depends on us. Google offers various AI models, to keep it simple, the core MediaPipe library is like a "PC", and google offers some kind of "disks" that we can insert for the "PC" to process.

In this case, the disk we wnat to insert is the Hand Landmarker model. The file of this model can be donwloaded in the official MediaPipe website, named "hand_landmarker.task". First we would need to store the name file into s variable so that we can load it into the MediaPipe library later

`model = 'hand_landmarker.task'`

MediaPipe offers flexibility, thats why we google give us the ability to easily modify each components in MediaPipe's library from the classes declared as:

* `BaseOption`: This class contains various core settings, the only thing we need to set for the library's settings is the model. Think it like a system, it has all perfectly set for us, the only thing it doesnt have is the disk it wants to process. The disk is the model previously mentioned, we simply need to insert it. The insert socket for this model is this class. To easily modify and acces it later, we declare a variable and load this class into our own local variable: `baseOpt = mp.tasks.BaseOptions`

* `HandLandmarker`: This acts as our main CPU. Because we want to load the model for tracking hands, we would need to load the CPU specialized in hand tracking. We need to load this into our local variable because it is the program that we want to use around our own program: `handLandmarker = mp.tasks.vision.HandLandmarker`

* `HandLandmarkerOptions`: This is almost the same as the BaseOption, but instead of being the settings for the whole PC, this specific settings behaves as the settings for the HandLandmarker CPU. Various things we can set here such as how many hands we want to specifically detect, how sure the AI needs to be before he says "I found the hand!", etc. We would need to load this into our local variabel too since we want to modify some of its settings later: `handLandmarkerOpt = mp.tasks.vision.HandLandmarkerOptions`

* `HandLandmarkResult`: This is the class that would contain the result data's after MediaPipe is done processing the frame, In this case because we need the hand coordinate tracked, we ned to load it into our own local variable: `handLandmarkerRes = mp.tasks.vision.HandLandmarkerResult`

* `RunningMode`: This is its own settings, it isnt a complex one, it simply a variable that conatins what mode we would like to be in. In Mediapipe, we can ask the program to process certain type of files, we say "I want you to track the hand in this video" or "I want you to track the hand in this photo". Since we are using a live camera, we would like to say "I want you to track the hand in this Live Stream", in this sense, live stream is the camera. Although for now we would just like to set thins up first, so we would need to simply load it into a variable first: `runningMode = mp.tasks.vision.RunningMode`

MediaPipe works by running a function made by us, we call it the "Callback Function". The callback function works as a drop-off point by the google's library for our program.

We work with mediapipe by first feeding it a single image frame, we feed the frame into mediapipe's program, and when it has done its thing, it holds the result (hand coordinates) in some variable. Here, the library didnt know where to give it to, what location it wanst to give it to. Thus we give it a location, a function, this is what is called a "Callback Function", it calls back the results processed by the library.

In mediapipe, to give the information of this "drop-off" point, we need to set it in its settings first, the landmark settings, which is the function `HandLandmarkOptions()`. Here because we already loaded it into our local variable HandLandmarkerOpt, we can assign it as:

`options = handLandmarkerOpt(base_options=baseOpt(model), running_mode = runningMode.LIVE_STREAM, result_callback=process_result, num_hands=1)`

This function have various arguments, but we would only need to set 4 of them by specifying the name and assigning values

- `base_options`: Remember our variable that contains teh base options, yeah, we feed it into this variable. By writing `base_options=baseOpt(model)`, we specified that we would like to use the Base Default options, with the model hand landmarker (remember: `model = 'hand_landmarker.task'`).

- `running_mode`: As said, here we choose what mode we would like to use, in this case we'll use LIVE_STREAM by writing `running_mode = runningMode.LIVE_STREAM`

- `result_callback`: This is the field where we specify our "drop-off point", for this, we must first create a function, a place, for the library to put the results in. Say we create the function `process_result()`. after creating, we must also declare some arguments, these arguments serves as eahc pint of the drop off location. `result`, the variable that will contain the result variables, such as the finger's coordinate, `outIMG`, which is the literal frmae we are working on, the same one that we fed previously, and finally the `timestamp_ms`, which is the duration since our prgram was ran in miliseconds.

- `num_hands`: simply the amount of hands we want to process.

Here, we save the configuration into a local variable that we can feed later into the program, so its:
change the settings -> save it into a variable -> feed the customized settings into the program later.
