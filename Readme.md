# Lip-Sync AI using  Wave2Lip 

### Objective- 
The objective of this project is to create an AI model that is proficient in lip-syncing i.e. synchronizing an audio file with a video file. The model is accurately matching the lip movements of the characters in the given video file with the corresponding audio file.

### Results-

Orignal Video - https://www.youtube.com/watch?v=YMuuEv37s0o&sub_confirmation=1
 
Orignal Audio - https://drive.google.com/file/d/1jhUOAeGw8lPjNf7Q1cIcBOvzE3CJ3gVz/view?pli=1

Generated Lip Synced Video -  https://github.com/stokome/LipSync-wave2lip-ai/assets/87638990/75dde50e-ca58-4d1b-9153-0bfd3045f981





### Further Improvements- 
Used Face Restoration AI(GFPGAN) to get ultra high quality videos-

Improved Lip Synced Video - https://drive.google.com/file/d/1eXF9nijXPRCIDUHA8x1V8YJeoH03WR-S/view?usp=drive_link

### Challenges Faced and Solved- 
1. Wave2Lib model dosent support video frames that dosent have face detected. So I had to make changes int the code base to ensure all frames are processed and frames that dosent had face got ignored by the model.

inference.py:
```
for rect, image in zip(predictions, images):
		if rect is None:
			#Add the original frame without face to results
			results.append([image, []])

```
```
	if len(coords) == 0: # Check if there are no face coordinates
		face = np.random.rand(args.img_size, args.img_size, 3) * 255
		coords_batch.append([])
		else:
		face = cv2.resize(face, (args.img_size, args.img_size))
		coords_batch.append(coords)
```

2.  Hyperparameter tunning- A fix resolution could not be passed as parameter in the model again modified the inference.py in wave2lip. After various analysis, I found 720x720 resolution frames with wave2lip_gan weights gave best result with `--nosmooth` as false.


### Use of Image Restoration AI GFPGAN_ 
Video quality was improved when I replaced wave2lip.pth weigths with wave2lip_gan weights. Further improvement was done using GFPGAN. GFPGAN is an image restoration AI. To use it on our inference we first divided the output images into frames, improved quality of each frame independently and then combined the frames in 25fps and audio.

### How to run:
Run the ipython notebooks in the google colab

### References

[Wave2Lip](https://github.com/Rudrabha/Wav2Lip)

[GFPGAN](https://github.com/TencentARC/GFPGAN)

[Wave2Lip-GFPGAN](https://github.com/ajay-sainy/Wav2Lip-GFPGAN)
