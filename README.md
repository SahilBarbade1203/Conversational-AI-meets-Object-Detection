Conversational AI Meets Object Detection

Overview

This project integrates state-of-the-art models and frameworks to create a system that combines Conversational AI with advanced Object Detection. The pipeline processes both text and image queries, using YOLOv8 for object detection and a multimodal RAG approach with the Gemini Pro Vision model. The results are then integrated into the RASA framework to facilitate interactive conversations.

Features

Object Detection: Uses YOLOv8 from Ultralytics (pretrained on the COCO dataset) for accurate object detection in images.
Multimodal Processing: Employs the Gemini Pro Vision model for multimodal retrieval-augmented generation (RAG), handling both text and image queries.
Integration with Google Cloud Vertex AI: Leverages Google Cloud's Vertex AI for scalable and efficient model deployment.
Conversational AI: Integrates with the RASA framework for interactive and dynamic conversations based on detected objects and textual queries.
Real-time Processing: Handles queries in real-time or near real-time, providing timely responses based on detected data and user input.
Installation

To set up the project locally, follow these steps:

Clone the Repository:
bash
Copy code
git clone https://github.com/SahilBarbade1203/Conversational_AI_meets_Object_Detection.git
cd Conversational_AI_meets_Object_Detection
Create a Virtual Environment (optional but recommended):
bash
Copy code
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
Install Dependencies:
bash
Copy code
pip install -r requirements.txt
Setup Google Cloud Vertex AI: Ensure you have your Google Cloud credentials configured and Vertex AI set up according to the project requirements.
Usage

Run the Object Detection Module:
bash
Copy code
python object_detection.py --input <path-to-image>
This will use YOLOv8 to detect objects in the image and output results such as object labels, bounding boxes, and confidence intervals.
Run the Multimodal RAG Pipeline:
bash
Copy code
python multimodal_pipeline.py --input <text-or-image-query>
This processes the input through the Gemini Pro Vision model, integrating the object detection results (if the input is an image) with the text query.
Run the Conversational AI Module:
bash
Copy code
python conversational_ai.py
Integrates the multimodal RAG results with the RASA framework for interactive conversations.
Configuration

Object Detection: Configure YOLOv8 parameters in config/yolo_config.yaml.
Multimodal RAG: Adjust settings for the Gemini Pro Vision model and Google Cloud Vertex AI in config/multimodal_config.yaml.
RASA Integration: Update RASA configurations and conversational models in the config/rasa_config.yaml.
Contributing

Contributions are welcome! To contribute to this project:

Fork the Repository.
Create a New Branch:
bash
Copy code
git checkout -b feature/your-feature
Make Your Changes and Commit:
bash
Copy code
git add .
git commit -m "Add your message here"
Push to Your Fork:
bash
Copy code
git push origin feature/your-feature
Create a Pull Request on GitHub.
License

This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgments

Object Detection: YOLOv8 from Ultralytics, pretrained on the COCO dataset.
Multimodal RAG: Gemini Pro Vision model with Google Cloud Vertex AI.
Conversational AI: RASA framework for conversational capabilities.
Contact

For any questions or further information, please contact Sahil Barbade.

Feel free to modify or add more details to better fit your project specifics!
