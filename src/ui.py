import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import torch
from torchvision import transforms
from model import DigitCNN

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Handwritten Digit Recognizer")
        self.canvas = tk.Canvas(self, width=280, height=280, bg='white')
        self.canvas.pack()
        self.button_predict = tk.Button(self, text="Predict", command=self.predict)
        self.button_predict.pack()
        self.button_clear = tk.Button(self, text="Clear", command=self.clear)
        self.button_clear.pack()
        self.label_result = tk.Label(self, text="Draw a digit and click Predict")
        self.label_result.pack()
        self.canvas.bind("<B1-Motion>", self.paint)
        self.image1 = Image.new("L", (280, 280), 'white')
        self.draw = ImageDraw.Draw(self.image1)
        self.model = self.load_model()

    def paint(self, event):
        x1, y1 = (event.x - 8), (event.y - 8)
        x2, y2 = (event.x + 8), (event.y + 8)
        self.canvas.create_oval(x1, y1, x2, y2, fill='black', width=8)
        self.draw.ellipse([x1, y1, x2, y2], fill='black')

    def clear(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, 280, 280], fill='white')
        self.label_result.config(text="Draw a digit and click Predict")

    def load_model(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = DigitCNN().to(device)
        model.load_state_dict(torch.load("digit_cnn.pth", map_location=device))
        model.eval()
        self.device = device
        return model

    def predict(self):
        # Resize and invert image to match MNIST format
        img = self.image1.resize((28, 28))
        img = ImageOps.invert(img)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        img = transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(img)
            pred = output.argmax(dim=1, keepdim=True).item()
        self.label_result.config(text=f"Prediction: {pred}")

if __name__ == "__main__":
    app = App()
    app.mainloop()