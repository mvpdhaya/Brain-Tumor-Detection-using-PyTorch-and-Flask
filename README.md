# Brain Tumor Detection using CNN

This project is a web application designed for brain tumor detection using a trained Convolutional Neural Network (CNN). The application allows users to upload MRI images and predicts the presence and type of brain tumor in real time.

## Features
- Upload MRI images via a user-friendly HTML frontend.
- Backend powered by Flask to handle requests and process images.
- Utilizes a PyTorch-trained CNN model to classify images into one of the following categories:
  - Glioma
  - Meningioma
  - No Tumor
  - Pituitary
- Displays prediction results in a popup with real-time feedback.

## Dataset
The model is trained using a dataset containing MRI images categorized into four classes:
- **Glioma**
- **Meningioma**
- **No Tumor**
- **Pituitary**

The dataset includes separate training and testing folders, each containing subfolders for these categories.

## Technologies Used
- **Frontend:** HTML, CSS, JavaScript
- **Backend:** Flask
- **Model Development:** PyTorch
- **Deployment:** Flask server

## Setup and Installation

### Prerequisites
- Python 3.7 or later
- Pip
- Virtual environment (recommended)
- PyTorch library

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/brain-tumor-detection.git
   cd brain-tumor-detection
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # For Linux/Mac
   venv\Scripts\activate   # For Windows
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Place the trained PyTorch model (`model.pth`) in the `models` folder.

5. Start the Flask server:
   ```bash
   python app.py
   ```

6. Open your browser and navigate to `http://127.0.0.1:5000` to use the application.

## Project Structure
```
brain-tumor-detection/
├── static/
│   ├── css/
│   ├── js/
│   └── uploads/
├── templates/
│   └── index.html
├── models/
│   └── model.pth
├── app.py
├── requirements.txt
└── README.md
```

## Usage
1. Upload an MRI image through the web interface.
2. Click the "Predict" button.
3. View the prediction result displayed in a popup.

## Model Training
The CNN model is trained using PyTorch. The dataset is split into training and testing sets, and the model achieves classification accuracy by learning features from MRI scans. Further training details can be found in the `model_training.ipynb` file (if included).

## Future Enhancements
- Integration of more advanced neural network architectures.
- Adding support for multiple image uploads.
- Deployment on cloud platforms like AWS or Heroku.

## Contributing
Feel free to fork this repository and submit pull requests. Contributions are welcome!

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgements
- The dataset used for training and testing.
- PyTorch documentation for model development.
- Flask documentation for backend support.
