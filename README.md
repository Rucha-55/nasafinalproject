# 🛰️ Satellite Image Segmentation - Urban Planning Intelligence

A beautiful web application for satellite image segmentation using deep learning. This system can classify satellite images into different land use categories including buildings, vegetation, water bodies, roads, and more.

## ✨ Features

- **🎯 Advanced Segmentation**: Uses U-Net deep learning model for precise land use classification
- **🎨 Beautiful UI**: Modern, responsive web interface with drag-and-drop file upload
- **📊 Detailed Analytics**: Statistical analysis with interactive charts and visualizations
- **🎨 Color-coded Maps**: Easy-to-understand segmentation maps with legend
- **📥 Export Options**: Download segmented maps, charts, and analysis reports
- **⚡ Real-time Processing**: Fast image processing with progress indicators

## 🏷️ Supported Classes

The model can identify 6 different land use types:

- 🌊 **Water Bodies** (Blue) - Rivers, lakes, ponds
- 🏠 **Buildings** (Purple) - Residential, commercial structures  
- 🛣️ **Roads** (Light Blue) - Streets, highways, pathways
- 🌱 **Vegetation** (Yellow) - Parks, forests, green areas
- 🏔️ **Land (Unpaved)** (Dark Purple) - Open land, soil
- ❓ **Unlabeled** (Gray) - Unclassified areas

## 🚀 Quick Start

### Option 1: Easy Start (Windows)
1. **Double-click `start.bat`** - This will automatically:
   - Create a Python virtual environment
   - Install all required dependencies
   - Start the application
   - Open your browser to http://localhost:5000

### Option 2: Manual Installation

1. **Clone or download this repository**

2. **Install Python 3.8+** (if not already installed)

3. **Create virtual environment**:
   ```bash
   python -m venv venv
   ```

4. **Activate virtual environment**:
   - Windows: `venv\Scripts\activate`
   - macOS/Linux: `source venv/bin/activate`

5. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

6. **Run the application**:
   ```bash
   python app.py
   ```

7. **Open your browser** to http://localhost:5000

## 📁 Project Structure

```
satellite-segmentation/
├── app.py                          # Main Flask application
├── requirements.txt                # Python dependencies
├── start.bat                      # Windows startup script
├── README.md                      # This file
├── models/
│   └── satellite_segmentation_full.h5  # Trained model (required)
├── templates/
│   └── index.html                 # Web interface
├── static/
│   ├── results/                   # Generated segmentation results
│   └── charts/                    # Generated analysis charts
└── uploads/                       # Temporary uploaded files
```

## 🎯 How to Use

1. **Start the application** using `start.bat` or manually
2. **Upload a satellite image** by dragging and dropping or clicking "Choose File"
3. **Wait for processing** - The AI model will analyze your image
4. **View results**:
   - Original vs segmented image comparison
   - Area coverage statistics for each land use type
   - Interactive charts and visualizations
   - Color-coded legend
5. **Download results** - Get segmented maps, charts, and reports

## 📊 Sample Outputs

### Segmentation Map
- Color-coded overlay showing different land use types
- High-resolution segmentation masks
- Easy-to-interpret visual results

### Statistical Analysis
```
Residential: 42.5%
Vegetation: 28.3%
Industrial: 12.1%
Water Bodies: 8.7%
Roads: 6.2%
Informal Settlements: 2.2%
```

### Visual Charts
- Pie charts showing land use distribution
- Bar charts for area coverage comparison
- Professional-quality visualizations

## 🔧 Technical Details

- **Framework**: Flask (Python web framework)
- **Model**: U-Net architecture with custom loss functions
- **Deep Learning**: TensorFlow/Keras
- **Image Processing**: OpenCV, PIL
- **Visualization**: Matplotlib, Seaborn
- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)

## 📋 Requirements

- **Python**: 3.8 or higher
- **RAM**: 8GB+ recommended for large images
- **Storage**: 2GB+ for model and dependencies
- **Browser**: Modern web browser (Chrome, Firefox, Safari, Edge)

## 🎨 Model Training Data

The model was trained on the **Dubai Satellite Imagery Dataset** featuring:
- 72 high-resolution satellite images
- 6 land use classes with pixel-wise annotations
- Images from Mohammed Bin Rashid Space Center (MBRSC)
- Professional annotations by Roia Foundation trainees

## 🚨 Troubleshooting

### Model Not Loading
- Ensure `satellite_segmentation_full.h5` is in the `models/` folder
- Check that the file is not corrupted
- Verify you have enough RAM (8GB+ recommended)

### Upload Issues
- Maximum file size: 16MB
- Supported formats: PNG, JPG, JPEG
- Try resizing large images before upload

### Performance Issues
- Large images may take longer to process
- Consider using smaller images (1024x1024 or less)
- Ensure adequate RAM and CPU resources

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Dubai Municipality and MBRSC for the satellite imagery dataset
- Roia Foundation for image annotations
- TensorFlow and OpenCV communities for excellent tools

---

**Made with ❤️ for Urban Planning and Smart City Development**
