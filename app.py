from flask import Flask, request, jsonify, render_template
import torch
from torchvision import transforms
from PIL import Image

# Define the same model architecture used during training
class BrainTumorCNN(torch.nn.Module):
    def __init__(self):
        super(BrainTumorCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = torch.nn.Linear(64 * 56 * 56, 128)  # Assuming input size is 224x224
        self.fc2 = torch.nn.Linear(128, 4)  # 4 classes

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 56 * 56)  # Flatten for fully connected layer
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize the model
model = BrainTumorCNN()

# Load the state dictionary
model.load_state_dict(torch.load('custom_brain_tumor_model.pth', map_location=torch.device('cpu')))
model.eval()

# Class labels
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Flask app setup
app = Flask(__name__)

# Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Process image
    img = Image.open(file).convert('RGB')
    img = transform(img).unsqueeze(0)

    # Predict
    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)
        result = class_names[predicted.item()]

    return jsonify({'prediction': result})

# Home route
@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
