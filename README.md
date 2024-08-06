##Conversational AI Meets Object Detection

## Overview

This project integrates state-of-the-art models and frameworks to create a system that combines Conversational AI with advanced Object Detection. The pipeline processes both text and image queries, using YOLOv8 for object detection and a multimodal RAG approach with the Gemini Pro Vision model. The results are then integrated into the RASA framework to facilitate interactive conversations.

## Features

- **Object Detection**: Utilizes YOLOv8 from Ultralytics (pretrained on the COCO dataset) for accurate object detection in images.
- **Multimodal Processing**: Employs the Gemini Pro Vision model for multimodal retrieval-augmented generation (RAG), handling both text and image queries.
- **Integration with Google Cloud Vertex AI**: Leverages Google Cloud's Vertex AI for scalable and efficient model deployment.
- **Conversational AI**: Integrates with the RASA framework for interactive and dynamic conversations based on detected objects and textual queries.
- **Real-time Processing**: Handles queries in real-time or near real-time, providing timely responses based on detected data and user input.

## Installation

To set up the project locally, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/SahilBarbade1203/Conversational_AI_meets_Object_Detection.git
   cd Conversational_AI_meets_Object_Detection

