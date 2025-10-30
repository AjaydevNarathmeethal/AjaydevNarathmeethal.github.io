---
layout: page
title: OpenCV - ALPR System
description: Automatic License Plate Recognition (ALPR) System using OpenCV
img: assets/img/proj_ALPR/004.jpg
importance: 1
category: work
related_publications: false
---

📂 [GitHub Repo Link](https://github.com/AjaydevNarathmeethal/ALPR_system)

## Project Overview

Automatic License Plate Recognition (ALPR) is a key component in smart surveillance systems, toll collection, parking management, and traffic monitoring. This project uses OpenCV and Tesseract OCR to detect license plates from video frames and extract their alphanumeric content.

The goal is to process a video input, detect vehicle license plates, extract and recognize the plate text using OCR, and display the results in real-time.

---

## How It Works

This ALPR system is built around two main components:

1. License Plate Detection
2. Optical Character Recognition (OCR) on the detected plate

The system processes each frame of a video feed and attempts to find contours that match the geometric characteristics of a license plate. Once a potential plate is found, it extracts the region of interest (ROI), preprocesses it, and uses Tesseract OCR to read the license number.

---

## Key Features

- Real-time license plate detection and recognition from video.
- Contour and morphological operations to isolate potential license plate regions.
- Aspect ratio filtering to accurately identify rectangular license plate candidates.
- Integration with Tesseract OCR for character recognition.
- Debug mode with visualizations to inspect processing stages.

---

## Workflow Summary

### 1. Video Frame Processing

- Frames are read from a video file (a.mp4) using OpenCV's VideoCapture.
- Each frame is passed through the ALPR pipeline to detect and recognize license plates.


<div class="row">
    <div class="col-sm-4 mx-auto mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/proj_ALPR/orig_frame.png" title="original frame" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Image of a frame before processing
</div>



### 2. License Plate Detection

- A blackhat morphological operation is applied to highlight darker regions (text) on a light background (license plate).
- Gradient filtering and thresholding are used to enhance edge-like structures.
- The resulting binary mask is cleaned using erosion/dilation to remove noise.
- Contours are detected and filtered by area and aspect ratio to find likely license plate regions.

<div class="row">
    <div class="col-sm-4 mx-auto mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/proj_ALPR/blackhat.png" title="final mask" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm-4 mx-auto mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/proj_ALPR/final.png" title="final mask" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    (a) Image after blackhat operation. (b) Image with final mask for detecting the Region of Interest (ROI)
</div>


### 3. Region of Interest (ROI) Extraction

- From the filtered contour list, the most probable license plate region is selected based on aspect ratio.
- The region is cropped, thresholded, and optionally cleaned using clear_border() to remove edge noise.

<div class="row">
    <div class="col-sm-3 mx-auto mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/proj_ALPR/license_plate.png" title="extracted license plate" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    The area of License plate extracted
</div>


### 4. Optical Character Recognition (OCR)

- The ROI is passed through Tesseract OCR with customized parameters (e.g., PSM mode and whitelist of alphanumeric characters).
- The recognized license plate text is cleaned and displayed on the original frame.

<div class="row">
    <div class="col-sm-4 mx-auto mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/proj_ALPR/detected_license_plate.png" title="extracted license plate" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    The detected license number is displayed in the live video feed. 
</div>

---

## Technologies Used

- **Python**
- **OpenCV** – for image preprocessing and video frame handling
- **Tesseract OCR** – for text extraction from processed images
- **scikit-image** – for optional cleaning (clear_border)
- **NumPy** – for array and matrix operations

## Potential Improvements

- Add support for multiple plates per frame.
- Enhance performance using deep learning models (e.g., YOLO, EAST detector).
- Expand to multi-language or non-standard plates.
- Optimize for mobile deployment or edge devices.



