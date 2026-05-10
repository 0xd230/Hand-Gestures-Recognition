# Hand-Gestures-Recognition

A Pattern Recognition Machine Learning based on vector projection and euclidian distance, with the maximal hand processed of: 1

This project uses the following library each for its own purposes:

* MediaPipe (HandLandmark Detection Model) : Detect and track hands
* OpenCV : Opening the camera and process each frame live, feeding it to MediaPipe
* Numpy : Process matrices and vector calculation
* Pathlib : Generate/open savefiles

# Algorithm

To detect hand gestures, the simplest appraoch that can be done is calculating the "similairty" between the live gesture the user is doing in real time, and the "golden sample" stored in a certain savefile.

To put it in simple words, first we would need a sample which can be done by making a "recording" system. In this part of the program, the user would make a gesture and record it, the recorded data would be saved in a savefile. Later on, when the user want to actually use the program to detetc recorded gestures, the user simply holds a certain gesture. The program would then see the shape of the hand and take its mathematical form and compare it one byb one by the recorded samples provided earlier. For each gestures the program would apply some kind of similarity score spanning between 0 and 1, and return the gesture with the highest score

To achieve this, this program utilizes two main components of hand gesture:
- Hand orrientation
- Hand shape

These two componnents is the base of what makes a hand gesture

* Normalization

To start off, when we feed the MediaPipe library with a frame or an image, the library would process the image, and define the location of each indices of the hand based on the landmark:

<img width="1073" height="372" alt="hand-landmarks" src="https://github.com/user-attachments/assets/8bd3c6a8-12f4-4590-9a68-9abfcbfad4c4" />


Inside each of those indices contains their current location on the single fram/image we game MediaPipe. Although not in a way of regular coordinate like (2, 3). MediaPipe instead provides us with a coordinate that only spans from 0 to 1, (0, 0) being the top left, (1, 1) being the bottom right. Because this is technically in a "Percentage" form, we must later multiply the coordinate given by MediaPipe with the screen resoltion (width and height)

Normalization involves making something consistant and efficient. Because we are tracking the hand gesture, we need to make the coordinates of each indices consistent, why?

for simplicity, lets say the screen resolution is 1280 by 720, width by height respectively

A user makes a hand shape of a peace sign, hovering it at the middle left side of the camera, take a single node, say the 8th node (the tip of the index), the coordinate of that node would be (190, 220). but once the user moves the hadn to the right side of the camera, that coordinate would change (890, 220). The hand shape didnt change, but with that drastic change of the coordinate, the program will see it as a change of shape.

To do this, we can introduce a new point on our hand that acts as a parent. So whenever we move out hand around the camera, the coordinate of each processed coordinates wont change and stays consistent.

The most ocnsistent point on the hand is the wirst (node 0). During hand gesture its the only point that doesnt transform, change, or fold. this way we can normalize the other nodes by subtracting each nodes coordinate (wrist node excluded) by the wrist node. This way even though every single node have different coordinates around the screen (say on the screen, the index node is at (190, 220), it will stay consistent on the wrist frame of reference (always (200, 100) even though we move the hand around)

This is our "normalization", we normalize each coordinates to consistently change in the wrist frame of reference

* Corodinate as a vector

From here, since we already have our consistent coordinates, we can treat them as vectors. Although in this case we only care about the unit vectors

See, a vector consists of a magnitude and the dirction of the vector. The magnitude represents the distance between a certain node and the point of reference (teh wrist). this way, if an index finger is open, the vector of the tip of of index finger would have a bigger magnitude, compared to when the finger is curled, teh distance would be closer to teh wrist, this the magnitude would be lower.

Although theres a flaw in this magnitude system. If the hand is closer to teh camera, by the camera's pint of view, the distance from the wrist and the index finger at any shape would be much more larger than when the hand is far away. Thus emerged inconsistency. In conclusion, we wouldnt need the magnitude of the vector, and instead we would simply need its direction. Since we would like to neglect the vector's magnitude, a simple way is to turn it into a unit vector, a vector which has the magnitude of 1, described by the following formula:

$$
\hat{v} = \frac{\overrightarrow v}{||v||}
$$

* Hand Orientation

The direction of this vector relies on the hand orientation, as the following:

<img width="1408" height="768" alt="Gemini_Generated_Image_1d3o741d3o741d3o" src="https://github.com/user-attachments/assets/0f0b1cef-eb5e-4c48-b5da-824e9efdc2e9" />
Where $$\overrightarrow{a}$$ is the vector for the hand tilted 90º, and $$\overrightarrow{v}$$ is the vector for the up right hand.

This shows that our vector's direction is proportional to our hand orientation, or to be specific, the position relative to the wrist.

Suppose we have a sample of a peace sign, and the user shows an open palm in the live stream camera. The program would compare each indices with each of their own matching sample indices. The program will compare the vector for the tip of the index in camera, to the vector of the tip of the index in the provided sample. such as show below

INSERT IMAGE HERE LATER

this way, we can match the hand's and finger's orientation by the vector projectoon Dot Product. By rearranging the dot product by uts cosine law, we can have whats called the "Cosine Similarity" denoted by:

$$
\cos{ \theta } = \frac{\overrightarrow{A} \dot \overrightarrow{B}}{||A|| \times ||B||}
$$

* Hand Shape

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

- `result_callback`: This is the field where we specify our "drop-off point", for this, we must first create a function where we can take and process the data given, a place, for the library to put the results in. Say we create the function `process_result()`. after creating, we must also declare some arguments, these arguments serves as eahc pint of the drop off location. `result`, the variable that will contain the result variables, such as the finger's coordinate, `outIMG`, which is the literal frmae we are working on, the same one that we fed previously, and finally the `timestamp_ms`, which is the duration since our prgram was ran in miliseconds.

- `num_hands`: simply the amount of hands we want to process.

Here, we save the configuration into a local variable that we can feed later into the program, so its:
change the settings -> save it into a variable -> feed the customized settings into the program later.


