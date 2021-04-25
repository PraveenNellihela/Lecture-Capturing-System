# Lecture-Capturing-System

This project can identify a whiteboard and the lecturer from a live video feed, exctract them seperately and combine the frames together with desktop screen recording to produce a comprehensive lecture video.

Can do the following in parallel:
<ul>
<li>record video and audio</li>
<li>detecting the whiteboard with TF classifier</li>
<li>combining sliced whiteboard from video frames with desktop recording and lecturers interactions to produce final video</li>
</ul>

Follows the lecturer if he moves out of frame by rotating the camera. If the whiteboard goes out of frame during this action, the last frame that contained the whiteboard will be kept in the video until the whole whiteboard comes into view.

External resources that helped with developing the TF model were from https://github.com/EdjeElectronics
 
