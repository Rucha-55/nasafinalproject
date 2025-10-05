from flask import Flask, render_template, request, jsonify, send_file, url_for, session
import os
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import base64
import io
import json
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from groq import Groq
import requests
from math import radians, cos, sin, asin, sqrt
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'static/results'
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
os.makedirs('static/charts', exist_ok=True)

# Configure Groq API
GROQ_API_KEY = os.environ.get('GROQ_API_KEY', '')

def initialize_groq_client():
    try:
        client = Groq(api_key=GROQ_API_KEY)
        # Test with current supported models
        supported_models = [
            "llama-3.1-8b-instant",
            "llama3-8b-8192", 
            "mixtral-8x7b-32768",
            "gemma2-9b-it",
            "llama3-groq-8b-8192-tool-use-preview"
        ]
        
        for model in supported_models:
            try:
                response = client.chat.completions.create(
                    messages=[{"role": "user", "content": "Hello"}],
                    model=model,
                    max_tokens=10
                )
                print(f"✅ Successfully initialized Groq client with model: {model}")
                return client
            except Exception as model_error:
                print(f"❌ Model {model} failed: {str(model_error)}")
                continue
                
        print(f"❌ Failed to initialize Groq client - no working models found")
        return None
    except Exception as e:
        print(f"❌ Failed to initialize Groq client: {str(e)}")
        return None

groq_client = initialize_groq_client()

# Global variables
model = None
minmaxscaler = MinMaxScaler()

# Class definitions from training code
CLASS_COLORS = {
    'Water': [226, 169, 41],      # #E2A929
    'Land': [132, 41, 246],       # #8429F6  
    'Road': [110, 193, 228],      # #6EC1E4
    'Building': [60, 16, 152],    # #3C1098
    'Vegetation': [254, 221, 58], # #FEDD3A
    'Unlabeled': [155, 155, 155]  # #9B9B9B
}

CLASS_NAMES = ['Water', 'Land', 'Road', 'Building', 'Vegetation', 'Unlabeled']

def load_model():
    """Load the trained model"""
    global model
    try:
        model_path = os.path.join('models', 'satellite_segmentation_full.h5')
        if os.path.exists(model_path):
            # Custom objects for the model
            def jaccard_coef(y_true, y_pred):
                from tensorflow.keras import backend as K
                yt, yp = K.flatten(y_true), K.flatten(y_pred)
                inter = K.sum(yt * yp)
                return (inter+1.0) / (K.sum(yt)+K.sum(yp)-inter+1.0)
            
            def dice_loss(y_true, y_pred, smooth=1):
                from tensorflow.keras import backend as K
                yt, yp = K.flatten(y_true), K.flatten(y_pred)
                inter = K.sum(yt*yp)
                return 1 - (2.*inter+smooth)/(K.sum(yt)+K.sum(yp)+smooth)
            
            def categorical_focal_loss(gamma=2., alpha=.25):
                def loss(y_true, y_pred):
                    from tensorflow.keras import backend as K
                    y_pred = K.clip(y_pred, K.epsilon(), 1-K.epsilon())
                    ce = -y_true*K.log(y_pred)
                    weight = alpha*K.pow(1-y_pred,gamma)
                    return K.sum(weight*ce,axis=-1)
                return loss
            
            def total_loss(y_true, y_pred):
                return dice_loss(y_true, y_pred) + categorical_focal_loss()(y_true, y_pred)
            
            custom_objects = {
                'jaccard_coef': jaccard_coef,
                'dice_loss': dice_loss,
                'categorical_focal_loss': categorical_focal_loss,
                'total_loss': total_loss
            }
            
            model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
            print("✅ Model loaded successfully!")
            return True
        else:
            print(f"❌ Model file not found at {model_path}")
            return False
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return False

def preprocess_image(image_path, patch_size=256):
    """Preprocess image for prediction"""
    image = cv2.imread(image_path, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Store original for display
    original_image = image.copy()
    
    # Resize to exact model input size (256x256)
    image_resized = cv2.resize(image, (patch_size, patch_size))
    
    # Normalize using the same method as training
    image_normalized = image_resized.astype(np.float32)
    
    # Apply MinMaxScaler normalization (same as training)
    image_flat = image_normalized.reshape(-1, image_normalized.shape[-1])
    image_normalized = minmaxscaler.fit_transform(image_flat).reshape(image_normalized.shape)
    
    return image_normalized, original_image

def create_segmentation_map(prediction, original_image):
    """Create colored segmentation map"""
    pred_mask = np.argmax(prediction, axis=-1)
    h, w = pred_mask.shape
    
    # Create colored mask
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_idx, class_name in enumerate(CLASS_NAMES):
        mask = pred_mask == class_idx
        colored_mask[mask] = CLASS_COLORS[class_name]
    
    # Resize colored mask and prediction to match original image size
    original_h, original_w = original_image.shape[:2]
    colored_mask_resized = cv2.resize(colored_mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
    pred_mask_resized = cv2.resize(pred_mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
    
    # Blend with original image
    alpha = 0.6
    blended = cv2.addWeighted(original_image, alpha, colored_mask_resized, 1-alpha, 0)
    
    return blended, colored_mask_resized, pred_mask_resized

def calculate_statistics(pred_mask):
    """Calculate area statistics for each class"""
    total_pixels = pred_mask.size
    stats = {}
    
    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_pixels = np.sum(pred_mask == class_idx)
        percentage = (class_pixels / total_pixels) * 100
        stats[class_name] = {
            'pixels': int(class_pixels),
            'percentage': round(percentage, 2)
        }
    
    return stats

def create_charts(stats, timestamp):
    """Create visualization charts"""
    # Set style
    plt.style.use('seaborn-v0_8')
    colors = [np.array(CLASS_COLORS[name])/255.0 for name in CLASS_NAMES]
    
    # Pie chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Filter out classes with 0%
    filtered_data = {k: v for k, v in stats.items() if v['percentage'] > 0}
    labels = list(filtered_data.keys())
    sizes = [filtered_data[label]['percentage'] for label in labels]
    filtered_colors = [np.array(CLASS_COLORS[label])/255.0 for label in labels]
    
    wedges, texts, autotexts = ax1.pie(sizes, labels=labels, colors=filtered_colors, 
                                       autopct='%1.1f%%', startangle=90)
    ax1.set_title('Land Use Distribution', fontsize=14, fontweight='bold')
    
    # Bar chart
    ax2.bar(labels, sizes, color=filtered_colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax2.set_title('Area Coverage by Class', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Percentage (%)', fontweight='bold')
    ax2.set_xlabel('Land Use Classes', fontweight='bold')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Add value labels on bars
    for i, v in enumerate(sizes):
        ax2.text(i, v + 0.5, f'{v:.1f}%', ha='center', fontweight='bold')
    
    plt.tight_layout()
    
    # Save chart
    chart_path = f'static/charts/analysis_{timestamp}.png'
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return chart_path

@app.route('/process-map-patches', methods=['POST'])
def process_map_patches():
    """Process multiple patches from map selection and return average statistics"""
    if not model:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        patches = data.get('patches', [])
        
        if not patches:
            return jsonify({'error': 'No patches provided'}), 400
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        all_stats = []
        processed_patches = []
        total_pixels = 0
        combined_class_counts = {name: 0 for name in CLASS_NAMES}
        
        # Process each patch
        for i, patch_data in enumerate(patches):
            try:
                # Decode base64 image
                image_data = base64.b64decode(patch_data.split(',')[1])
                image = Image.open(io.BytesIO(image_data))
                image_np = np.array(image.convert('RGB'))
                
                # Resize to model input size
                img_resized = cv2.resize(image_np, (256, 256))
                img_preprocessed = img_resized.astype('float32') / 255.0
                img_preprocessed = np.expand_dims(img_preprocessed, axis=0)
                
                # Predict
                pred = model.predict(img_preprocessed, verbose=0)[0]
                pred_mask = np.argmax(pred, axis=-1)
                
                # Calculate statistics for this patch
                unique_classes, counts = np.unique(pred_mask, return_counts=True)
                total_patch_pixels = pred_mask.size
                total_pixels += total_patch_pixels
                
                patch_stats = {}
                for class_idx, count in zip(unique_classes, counts):
                    if class_idx < len(CLASS_NAMES):
                        class_name = CLASS_NAMES[class_idx]
                        percentage = (count / total_patch_pixels) * 100
                        patch_stats[class_name] = {
                            'pixels': int(count),
                            'percentage': round(percentage, 2)
                        }
                        combined_class_counts[class_name] += count
                
                all_stats.append(patch_stats)
                
                # Save processed patch (optional - for visualization)
                colored_mask = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.uint8)
                for class_idx, color in enumerate([CLASS_COLORS[name] for name in CLASS_NAMES]):
                    colored_mask[pred_mask == class_idx] = color
                
                patch_info = {
                    'index': i,
                    'statistics': patch_stats,
                    'processed': True
                }
                processed_patches.append(patch_info)
                
            except Exception as e:
                patch_info = {
                    'index': i,
                    'error': str(e),
                    'processed': False
                }
                processed_patches.append(patch_info)
                continue
        
        # Calculate average statistics across all patches
        avg_stats = {}
        for class_name in CLASS_NAMES:
            if total_pixels > 0:
                avg_percentage = (combined_class_counts[class_name] / total_pixels) * 100
                avg_stats[class_name] = {
                    'total_pixels': int(combined_class_counts[class_name]),
                    'percentage': round(avg_percentage, 2)
                }
            else:
                avg_stats[class_name] = {'total_pixels': 0, 'percentage': 0.0}
        
        # Create average statistics chart
        chart_path = create_charts(avg_stats, timestamp)
        
        response = {
            'success': True,
            'timestamp': timestamp,
            'total_patches': len(patches),
            'processed_patches': len([p for p in processed_patches if p.get('processed', False)]),
            'failed_patches': len([p for p in processed_patches if not p.get('processed', False)]),
            'average_statistics': avg_stats,
            'individual_patch_stats': all_stats,
            'chart_image': chart_path,
            'total_pixels_analyzed': total_pixels
        }
        
        # Store analysis results in session for prediction use
        session['latest_analysis'] = {
            'statistics': avg_stats,
            'timestamp': timestamp,
            'image_name': f'Map Analysis - {len(patches)} patches',
            'chart_image': chart_path,
            'is_map_analysis': True
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': f'Batch processing failed: {str(e)}'}), 500

# ========================================
# Urban Planning Analysis Functions
# ========================================

def haversine(lon1, lat1, lon2, lat2):
    """Calculate distance between two points in kilometers"""
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    return 6371 * c

def fetch_osm_data(south, west, north, east, query_items):
    """Fetch data from OpenStreetMap"""
    overpass_url = "https://overpass-api.de/api/interpreter"
    
    query_parts = []
    for item in query_items:
        query_parts.append(f'node[{item}]({south},{west},{north},{east});')
        query_parts.append(f'way[{item}]({south},{west},{north},{east});')
    
    overpass_query = f"""
    [out:json][timeout:30];
    ({''.join(query_parts)});
    out center;
    """
    
    try:
        response = requests.post(overpass_url, data=overpass_query, timeout=35)
        return response.json()
    except Exception as e:
        print(f"OSM Error: {str(e)}")
        return {'elements': []}

def extract_locations(osm_data):
    """Extract coordinates and detailed information from OSM data"""
    locations = []
    for element in osm_data.get('elements', []):
        loc = {}
        if 'lat' in element and 'lon' in element:
            loc = {'lat': element['lat'], 'lon': element['lon']}
        elif 'center' in element:
            loc = {'lat': element['center']['lat'], 'lon': element['center']['lon']}
        else:
            continue
        
        tags = element.get('tags', {})
        name = (tags.get('name') or tags.get('name:en') or tags.get('official_name') or 
                tags.get('alt_name') or tags.get('operator') or tags.get('brand') or 'Unnamed Facility')
        
        loc['name'] = name
        loc['type'] = tags.get('amenity') or tags.get('shop') or tags.get('building') or tags.get('landuse') or 'facility'
        loc['address'] = tags.get('addr:street', '') + ' ' + tags.get('addr:housenumber', '')
        loc['tags'] = tags
        
        locations.append(loc)
    return locations

def find_optimal_locations(bounds, existing_locations, needed_count, area_type="Urban", housing_locations=None, transport_locations=None, industrial_locations=None):
    """
    Find optimal locations for new facilities with REQUIREMENTS-BASED PRECISION
    
    Requirements:
    1. Maximize coverage of underserved areas (gaps in service)
    2. Prioritize proximity to population centers (housing clusters)
    3. Ensure accessibility via transportation infrastructure
    4. Avoid industrial/polluted zones
    5. Maintain proper spacing between facilities
    """
    if needed_count <= 0:
        return []
    
    south, west, north, east = bounds['south'], bounds['west'], bounds['north'], bounds['east']
    
    # REQUIREMENTS-BASED GRID: High precision for accurate placement
    if area_type == "Urban":
        grid_size = 35  # Ultra-fine grid for urban areas (1,225 candidates)
    elif area_type == "Semi-Urban":
        grid_size = 28  # Fine grid for semi-urban (784 candidates)
    else:  # Rural
        grid_size = 22  # Medium grid for rural areas (484 candidates)
    
    lat_step = (north - south) / grid_size
    lon_step = (east - west) / grid_size
    
    print(f"🔍 Evaluating {grid_size}x{grid_size} = {grid_size*grid_size} candidate locations...")
    
    candidates = []
    for i in range(grid_size):
        for j in range(grid_size):
            # PRECISE POSITIONING: Center each candidate in grid cell
            lat = south + (i + 0.5) * lat_step
            lon = west + (j + 0.5) * lon_step
            
            # REQUIREMENT 1: SERVICE GAP ANALYSIS (70% weight)
            # Find how far this location is from existing facilities
            if existing_locations:
                distances = [haversine(lon, lat, loc['lon'], loc['lat']) for loc in existing_locations]
                min_dist = min(distances)
                avg_dist = sum(distances) / len(distances)
                
                # Service gap score: Higher distance = better coverage
                service_gap_score = min_dist * 10.0  # Scale up for scoring
                
                # Penalty for clustering: reduce score if too many facilities nearby
                nearby_count = sum(1 for d in distances if d < 0.5)  # Within 500m
                clustering_penalty = nearby_count * 5.0
            else:
                # No existing facilities: distribute evenly
                service_gap_score = 50.0
                min_dist = 999
                avg_dist = 0
                clustering_penalty = 0
            
            # REQUIREMENT 2: POPULATION PROXIMITY (20% weight)
            # Prefer locations near residential areas (housing clusters)
            population_score = 0
            if housing_locations:
                housing_distances = [haversine(lon, lat, h['lon'], h['lat']) for h in housing_locations[:50]]  # Check nearest 50
                if housing_distances:
                    nearest_housing = min(housing_distances)
                    # Score: closer to housing = higher score (inverse relationship)
                    if nearest_housing < 0.5:  # Within 500m
                        population_score = 20.0
                    elif nearest_housing < 1.0:  # Within 1km
                        population_score = 15.0
                    elif nearest_housing < 2.0:  # Within 2km
                        population_score = 10.0
                    else:
                        population_score = 5.0
            
            # REQUIREMENT 3: TRANSPORTATION ACCESSIBILITY (15% weight)
            # Prefer locations near transportation infrastructure
            accessibility_score = 0
            if transport_locations:
                transport_distances = [haversine(lon, lat, t['lon'], t['lat']) for t in transport_locations[:30]]
                if transport_distances:
                    nearest_transport = min(transport_distances)
                    # Score: closer to transport = higher score
                    if nearest_transport < 0.3:  # Within 300m
                        accessibility_score = 15.0
                    elif nearest_transport < 0.6:  # Within 600m
                        accessibility_score = 10.0
                    elif nearest_transport < 1.0:  # Within 1km
                        accessibility_score = 5.0
            
            # REQUIREMENT 4: AVOID INDUSTRIAL ZONES (penalty)
            # Penalize locations near industrial/polluted areas
            industrial_penalty = 0
            if industrial_locations:
                industrial_distances = [haversine(lon, lat, ind['lon'], ind['lat']) for ind in industrial_locations[:20]]
                if industrial_distances:
                    nearest_industrial = min(industrial_distances)
                    # Penalty: closer to industrial = higher penalty
                    if nearest_industrial < 0.5:  # Within 500m
                        industrial_penalty = 20.0
                    elif nearest_industrial < 1.0:  # Within 1km
                        industrial_penalty = 10.0
                    elif nearest_industrial < 2.0:  # Within 2km
                        industrial_penalty = 5.0
            
            # REQUIREMENT 5: IDEAL SPACING based on area type
            if area_type == "Urban":
                ideal_distance = 1.2  # 1.2km ideal spacing (dense coverage)
                min_spacing = 0.25  # 250m minimum between suggestions
            elif area_type == "Semi-Urban":
                ideal_distance = 2.5  # 2.5km ideal spacing
                min_spacing = 0.4  # 400m minimum between suggestions
            else:  # Rural
                ideal_distance = 5.0  # 5km ideal spacing
                min_spacing = 0.8  # 800m minimum between suggestions
            
            # Ideal spacing bonus
            spacing_bonus = 0
            if existing_locations:
                distance_from_ideal = abs(min_dist - ideal_distance)
                # Bonus if close to ideal distance (within 50% tolerance)
                if distance_from_ideal < (ideal_distance * 0.5):
                    spacing_bonus = 10.0 - (distance_from_ideal / ideal_distance * 10)
            
            # COMBINED SCORE: All requirements integrated
            final_score = (
                service_gap_score +           # 70% - Service gap coverage
                population_score +            # 20% - Near population
                accessibility_score +         # 15% - Near transport
                spacing_bonus -               # Bonus - Ideal spacing
                clustering_penalty -          # Penalty - Too close to others
                industrial_penalty            # Penalty - Near pollution
            )
            
            candidates.append({
                'lat': round(lat, 6),
                'lon': round(lon, 6),
                'distance': round(min_dist, 3) if existing_locations else 999,
                'score': round(final_score, 2),
                'avg_distance': round(avg_dist, 3) if existing_locations else 0,
                'population_score': round(population_score, 1),
                'accessibility_score': round(accessibility_score, 1),
                'grid_pos': f"{i},{j}"
            })
    
    # INTELLIGENT SELECTION: Sort by combined score
    candidates.sort(key=lambda x: x['score'], reverse=True)
    
    print(f"📊 Top candidate scores: {[c['score'] for c in candidates[:5]]}")
    
    # Select best locations with proper spacing
    selected = []
    
    for candidate in candidates:
        if len(selected) >= needed_count:
            break
        
        # Check minimum separation from selected locations
        too_close = False
        for sel in selected:
            dist = haversine(candidate['lon'], candidate['lat'], sel['lon'], sel['lat'])
            if dist < min_spacing:
                too_close = True
                break
        
        # Also check distance from existing facilities (avoid too close placement)
        if existing_locations and not too_close:
            for exist in existing_locations:
                dist = haversine(candidate['lon'], candidate['lat'], exist['lon'], exist['lat'])
                if dist < (min_spacing * 0.4):  # 40% of min_spacing
                    too_close = True
                    break
        
        if not too_close:
            selected.append(candidate)
    
    # Fallback: relax constraints if needed
    if len(selected) < needed_count and len(candidates) > len(selected):
        print(f"⚠️  Relaxing spacing to meet requirement ({len(selected)}/{needed_count})")
        relaxed_spacing = min_spacing * 0.6
        for candidate in candidates:
            if len(selected) >= needed_count:
                break
            if candidate not in selected:
                too_close = False
                for sel in selected:
                    if haversine(candidate['lon'], candidate['lat'], sel['lon'], sel['lat']) < relaxed_spacing:
                        too_close = True
                        break
                if not too_close:
                    selected.append(candidate)
    
    print(f"✅ Selected {len(selected)}/{needed_count} REQUIREMENT-BASED locations (score range: {selected[0]['score']:.1f} to {selected[-1]['score']:.1f})")
    return selected

# ========================================
# Main Routes
# ========================================

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/analysis')
def analysis():
    return render_template('analysis.html')

@app.route('/map')
def map_view():
    return render_template('map3.html')

# Prediction routes
@app.route('/predictions/urban-growth')
def urban_growth():
    return render_template('predictions/urban_growth.html')

@app.route('/predictions/environmental')
def environmental():
    return render_template('predictions/environmental.html')

@app.route('/predictions/infrastructure')
def infrastructure():
    return render_template('predictions/infrastructure.html')

@app.route('/predictions/health')
def health():
    return render_template('predictions/health.html')

@app.route('/predictions/disaster')
def disaster():
    return render_template('predictions/disaster.html')

@app.route('/predictions/agriculture')
def agriculture():
    return render_template('predictions/agriculture.html')

# Utility pages
@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/docs')
def docs():
    return render_template('docs.html')

@app.route('/api')
def api_docs():
    return render_template('api_docs.html')

# Generate predictions based on land use data
@app.route('/generate-prediction', methods=['POST'])
def generate_prediction():
    try:
        data = request.get_json()
        prediction_type = data.get('type')
        land_use_stats = data.get('land_use_stats')
        
        if not land_use_stats or not prediction_type:
            return jsonify({'error': 'Missing required data'}), 400
        
        # Generate prediction using Groq API
        prediction = generate_groq_prediction(prediction_type, land_use_stats)
        
        return jsonify({
            'success': True,
            'prediction': prediction,
            'type': prediction_type
        })
        
    except Exception as e:
        return jsonify({'error': f'Prediction generation failed: {str(e)}'}), 500

def generate_groq_prediction(prediction_type, land_use_stats):
    """Generate predictions using Groq AI based on land use statistics"""
    
    # Format the land use data for the prompt
    land_use_text = ""
    for land_type, stats in land_use_stats.items():
        land_use_text += f"- {land_type}: {stats['percentage']}% ({stats['total_pixels']} pixels)\n"
    
    prompts = {
        'urban-growth': f"""
        Based on the following satellite imagery land use analysis:
        {land_use_text}
        
        Provide insights on urban growth and expansion patterns. Analyze:
        1. Current urbanization level (buildings + roads percentage)
        2. Urban sprawl potential and expansion patterns
        3. Growth trajectory predictions
        4. Infrastructure development needs
        5. Recommended planning strategies
        
        Provide specific percentages and actionable recommendations.
        """,
        
        'environmental': f"""
        Based on the following land use distribution:
        {land_use_text}
        
        Analyze environmental impact and sustainability:
        1. Green cover assessment (vegetation percentage)
        2. Water resource availability and status
        3. Environmental stress indicators
        4. Urban heat island potential
        5. Biodiversity impact assessment
        6. Recommendations for environmental protection
        
        Include specific metrics and conservation strategies.
        """,
        
        'infrastructure': f"""
        Based on this land use analysis:
        {land_use_text}
        
        Provide infrastructure planning insights:
        1. Transportation network adequacy
        2. Utility infrastructure requirements
        3. Housing demand projections
        4. Public facilities planning
        5. Resource allocation priorities
        6. Future infrastructure investment areas
        
        Include specific recommendations and priority rankings.
        """,
        
        'health': f"""
        Analyze public health implications from this land use data:
        {land_use_text}
        
        Assess:
        1. Air quality impact (built-up vs green areas)
        2. Urban heat island health risks
        3. Access to green spaces for mental health
        4. Disease outbreak risks
        5. Healthcare facility planning needs
        6. Wellness infrastructure recommendations
        
        Provide health risk assessments and mitigation strategies.
        """,
        
        'disaster': f"""
        Based on this land use distribution:
        {land_use_text}
        
        Analyze disaster vulnerability and management:
        1. Flood risk assessment (water bodies + built-up areas)
        2. Evacuation route planning
        3. Emergency facility requirements
        4. Vulnerable area identification
        5. Disaster preparedness recommendations
        6. Climate resilience strategies
        
        Include risk levels and specific action plans.
        """,
        
        'agriculture': f"""
        Analyze agricultural and food security implications:
        {land_use_text}
        
        Assess:
        1. Agricultural land availability
        2. Food production capacity
        3. Urban agriculture potential
        4. Land conversion trends
        5. Food security projections
        6. Sustainable agriculture recommendations
        
        Provide specific metrics and sustainability strategies.
        """
    }
    
    prompt = prompts.get(prediction_type, prompts['urban-growth'])
    
    try:
        if not groq_client:
            return "AI prediction service is currently unavailable. Please try again later or contact support."
        
        # Try multiple models in order of preference
        models_to_try = [
            "llama3-groq-70b-8192-tool-use-preview",  # Larger model for detailed analysis
            "mixtral-8x7b-32768",  # Alternative large model
            "llama3-8b-8192",  # Smaller but reliable model
            "llama-3.1-8b-instant",  # Fast model
            "gemma2-9b-it"  # Fallback option
        ]
        
        for model_name in models_to_try:
            try:
                response = groq_client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": "You are an expert urban planning and satellite imagery analysis consultant. Provide detailed, professional insights based on land use data."},
                        {"role": "user", "content": prompt}
                    ],
                    model=model_name,
                    max_tokens=1500,
                    temperature=0.7
                )
                return response.choices[0].message.content
            except Exception as model_error:
                print(f"❌ Model {model_name} failed: {str(model_error)}")
                continue
        
        return "AI prediction service is temporarily unavailable due to model issues. Please try again later."
    except Exception as e:
        return f"Error generating prediction: {str(e)}. Please try again or contact support if the issue persists."

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not model:
        return jsonify({'error': 'Model not loaded. Please check model file.'}), 500
    
    try:
        # Save uploaded file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"upload_{timestamp}_{file.filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process image
        processed_image, original_image = preprocess_image(filepath)
        
        # Make prediction
        prediction = model.predict(np.expand_dims(processed_image, 0), verbose=0)[0]
        
        # Create segmentation map
        blended_image, colored_mask, pred_mask = create_segmentation_map(prediction, original_image)
        
        # Calculate statistics
        stats = calculate_statistics(pred_mask)
        
        # Create charts
        chart_path = create_charts(stats, timestamp)
        
        # Save results
        result_original = f'static/results/original_{timestamp}.png'
        result_segmented = f'static/results/segmented_{timestamp}.png'
        result_mask = f'static/results/mask_{timestamp}.png'
        result_overlay = f'static/results/overlay_{timestamp}.png'
        
        cv2.imwrite(result_original, cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR))
        cv2.imwrite(result_segmented, cv2.cvtColor(blended_image, cv2.COLOR_RGB2BGR))
        cv2.imwrite(result_mask, cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR))
        
        # Create overlay comparison image
        h, w = original_image.shape[:2]
        overlay_comparison = np.zeros((h, w * 3, 3), dtype=np.uint8)
        overlay_comparison[:, 0:w] = original_image
        overlay_comparison[:, w:2*w] = colored_mask
        overlay_comparison[:, 2*w:3*w] = blended_image
        cv2.imwrite(result_overlay, cv2.cvtColor(overlay_comparison, cv2.COLOR_RGB2BGR))
        
        # Prepare response
        response = {
            'success': True,
            'timestamp': timestamp,
            'original_image': result_original,
            'segmented_image': result_segmented,
            'mask_image': result_mask,
            'overlay_image': result_overlay,
            'chart_image': chart_path,
            'statistics': stats,
            'total_area': pred_mask.size,
            'image_shape': original_image.shape[:2]
        }
        
        # Store analysis results in session for prediction use
        session['latest_analysis'] = {
            'statistics': stats,
            'timestamp': timestamp,
            'image_name': file.filename,
            'original_image': result_original,
            'segmented_image': result_segmented,
            'mask_image': result_mask,
            'overlay_image': result_overlay,
            'chart_image': chart_path
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

@app.route('/api/model-status')
def model_status():
    return jsonify({'loaded': model is not None})

@app.route('/api/latest-analysis')
def get_latest_analysis():
    """Get the latest analysis results for predictions"""
    if 'latest_analysis' in session:
        return jsonify({
            'success': True,
            'data': session['latest_analysis']
        })
    else:
        return jsonify({
            'success': False,
            'message': 'No analysis data available. Please upload and analyze an image first.'
        })

@app.route('/create-sample', methods=['POST'])
def create_sample():
    """Create and process a sample satellite image"""
    if not model:
        return jsonify({'error': 'Model not loaded. Please check model file.'}), 500
    
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create a sample response indicating the feature is available
        response = {
            'success': True,
            'message': 'Sample creation feature is available. Use the map interface to analyze real satellite data.',
            'timestamp': timestamp,
            'redirect_to': '/map'
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': f'Sample creation failed: {str(e)}'}), 500


# ========================================
# Urban Planning Analyzer Routes
# ========================================

@app.route('/urban-planning')
def urban_planning_index():
    """Main urban planning analyzer page"""
    return render_template('urban_planning/index.html')

@app.route('/urban-planning/dashboard')
def urban_planning_dashboard():
    """Urban planning dashboard"""
    return render_template('urban_planning/dashboard.html')

@app.route('/urban-planning/analysis')
def urban_planning_analysis():
    """Urban planning analysis page"""
    return render_template('urban_planning/analysis.html')

@app.route('/urban-planning/recommendations')
def urban_planning_recommendations():
    """Urban planning recommendations page"""
    return render_template('urban_planning/recommendations.html')

@app.route('/urban-planning/reports')
def urban_planning_reports():
    """Urban planning reports page"""
    return render_template('urban_planning/reports.html')

@app.route('/api/urban-planning/analysis', methods=['POST'])
def urban_planning_api_analysis():
    """Complete analysis of all 10 objectives"""
    try:
        data = request.get_json()
        bounds = data.get('bounds', {})
        
        south, west, north, east = bounds['south'], bounds['west'], bounds['north'], bounds['east']
        
        # Calculate area
        lat_dist = haversine(west, south, west, north)
        lon_dist = haversine(west, south, east, south)
        area_km2 = lat_dist * lon_dist
        
        print(f"Analyzing area: {area_km2:.2f} km²")
        
        # Fetch all data categories
        print("Fetching healthcare data...")
        healthcare_data = fetch_osm_data(south, west, north, east, [
            '"amenity"="hospital"', 
            '"amenity"="clinic"', 
            '"amenity"="doctors"',
            '"amenity"="health_centre"',
            '"amenity"="pharmacy"',
            '"amenity"="dentist"',
            '"healthcare"="hospital"',
            '"healthcare"="clinic"',
            '"healthcare"="centre"',
            '"healthcare"="doctor"',
            '"amenity"="nursing_home"',
            '"amenity"="veterinary"',
            '"healthcare"="laboratory"',
            '"healthcare"="physiotherapist"',
            '"amenity"="social_facility"'
        ])
        
        print("Fetching food access data...")
        food_data = fetch_osm_data(south, west, north, east, [
            '"shop"="supermarket"', 
            '"shop"="convenience"', 
            '"shop"="greengrocer"',
            '"shop"="general"',
            '"shop"="grocery"',
            '"shop"="mall"',
            '"shop"="department_store"',
            '"amenity"="marketplace"',
            '"shop"="bakery"',
            '"shop"="butcher"',
            '"shop"="farm"',
            '"shop"="seafood"',
            '"shop"="deli"',
            '"shop"="food"',
            '"shop"="organic"',
            '"amenity"="fast_food"',
            '"amenity"="restaurant"',
            '"amenity"="cafe"'
        ])
        
        print("Fetching housing data...")
        housing_data = fetch_osm_data(south, west, north, east, [
            '"building"="residential"', 
            '"building"="apartments"', 
            '"building"="house"',
            '"building"="detached"',
            '"building"="terrace"',
            '"building"="dormitory"',
            '"building"="bungalow"',
            '"landuse"="residential"',
            '"building"="semi_detached_house"',
            '"building"="villa"',
            '"building"="flat"',
            '"building"="townhouse"',
            '"building"="maisonette"',
            '"building"="farm_auxiliary"',
            '"building"="semidetached_house"'
        ])
        
        print("Fetching transportation data...")
        transport_data = fetch_osm_data(south, west, north, east, [
            '"amenity"="bus_station"', 
            '"highway"="bus_stop"', 
            '"public_transport"="station"',
            '"public_transport"="stop_position"',
            '"railway"="station"',
            '"railway"="halt"',
            '"amenity"="taxi"',
            '"public_transport"="platform"',
            '"railway"="tram_stop"',
            '"railway"="subway_entrance"',
            '"amenity"="ferry_terminal"',
            '"amenity"="parking"',
            '"amenity"="parking_entrance"',
            '"public_transport"="stop_area"',
            '"highway"="services"'
        ])
        
        print("Fetching parks data...")
        parks_data = fetch_osm_data(south, west, north, east, [
            '"leisure"="park"', 
            '"leisure"="playground"', 
            '"leisure"="garden"',
            '"leisure"="recreation_ground"',
            '"leisure"="nature_reserve"',
            '"landuse"="recreation_ground"',
            '"leisure"="sports_centre"',
            '"leisure"="pitch"',
            '"leisure"="stadium"',
            '"leisure"="track"',
            '"leisure"="dog_park"',
            '"leisure"="fitness_centre"',
            '"leisure"="swimming_pool"',
            '"natural"="beach"',
            '"tourism"="picnic_site"'
        ])
        
        print("Fetching industrial/pollution data...")
        industrial_data = fetch_osm_data(south, west, north, east, [
            '"landuse"="industrial"', '"man_made"="works"'
        ])
        
        print("Fetching water bodies...")
        water_data = fetch_osm_data(south, west, north, east, [
            '"natural"="water"', '"waterway"="river"'
        ])
        
        print("Fetching agricultural data...")
        agriculture_data = fetch_osm_data(south, west, north, east, [
            '"landuse"="farmland"', '"landuse"="orchard"'
        ])
        
        print("Fetching waste facilities...")
        waste_data = fetch_osm_data(south, west, north, east, [
            '"amenity"="recycling"', '"amenity"="waste_disposal"'
        ])
        
        print("Fetching energy infrastructure...")
        energy_data = fetch_osm_data(south, west, north, east, [
            '"power"="plant"', '"power"="substation"', '"power"="tower"'
        ])
        
        # Extract locations
        healthcare_locs = extract_locations(healthcare_data)
        food_locs = extract_locations(food_data)
        housing_locs = extract_locations(housing_data)
        transport_locs = extract_locations(transport_data)
        parks_locs = extract_locations(parks_data)
        industrial_locs = extract_locations(industrial_data)
        water_locs = extract_locations(water_data)
        agriculture_locs = extract_locations(agriculture_data)
        waste_locs = extract_locations(waste_data)
        energy_locs = extract_locations(energy_data)
        
        # Count facilities
        counts = {
            'healthcare': len(healthcare_locs),
            'food': len(food_locs),
            'housing': len(housing_locs),
            'transportation': len(transport_locs),
            'parks': len(parks_locs),
            'industrial': len(industrial_locs),
            'water': len(water_locs),
            'agriculture': len(agriculture_locs),
            'waste': len(waste_locs),
            'energy': len(energy_locs)
        }
        
        print(f"Facility counts: {counts}")
        
        # Enhanced area classification with multiple factors
        # Calculate facility density (excluding water, agriculture, energy infrastructure)
        urban_facilities = counts['healthcare'] + counts['food'] + counts['transportation'] + counts['parks'] + counts['waste']
        facility_density = urban_facilities / area_km2 if area_km2 > 0 else 0
        
        # Calculate housing density (key indicator)
        housing_density = counts['housing'] / area_km2 if area_km2 > 0 else 0
        
        # Calculate infrastructure score (0-100)
        infrastructure_score = 0
        if counts['healthcare'] > 0: infrastructure_score += 20
        if counts['food'] > 0: infrastructure_score += 15
        if counts['transportation'] > 0: infrastructure_score += 20
        if counts['parks'] > 0: infrastructure_score += 10
        if counts['waste'] > 0: infrastructure_score += 10
        if counts['energy'] > 0: infrastructure_score += 10
        if counts['industrial'] > 0: infrastructure_score += 15
        
        # Multi-factor classification with absolute counts consideration - VERY LENIENT
        classification_score = 0
        
        # Factor 1: Facility density (30% weight) - VERY LENIENT thresholds
        if facility_density > 10:  # Very dense urban (lowered from 20)
            classification_score += 30
        elif facility_density > 2:  # Moderate urban (lowered from 5)
            classification_score += 28
        elif facility_density > 0.5:  # Low density urban/suburban (lowered from 1)
            classification_score += 20
        elif facility_density > 0.1:  # Sparse but still has some (lowered from 0.3)
            classification_score += 12
        
        # Factor 2: Housing density (25% weight) - VERY LENIENT thresholds
        if housing_density > 30:  # Very dense (lowered from 50)
            classification_score += 25
        elif housing_density > 5:  # Moderate (lowered from 10)
            classification_score += 22
        elif housing_density > 1:  # Low density (lowered from 3)
            classification_score += 15
        elif housing_density > 0.2:  # Sparse (lowered from 0.5)
            classification_score += 8
        
        # Factor 3: Infrastructure completeness (25% weight)
        if infrastructure_score >= 80:
            classification_score += 25
        elif infrastructure_score >= 50:
            classification_score += 20
        elif infrastructure_score >= 30:
            classification_score += 15
        elif infrastructure_score >= 10:  # Even minimal infrastructure counts
            classification_score += 8
        
        # Factor 4: ABSOLUTE FACILITY COUNTS (20% weight) - VERY LENIENT!
        # For large areas, density is low but total count is high = URBAN
        total_key_facilities = counts['healthcare'] + counts['food'] + counts['transportation'] + counts['parks']
        if total_key_facilities > 100:  # High count = definitely urban (lowered from 150)
            classification_score += 20
        elif total_key_facilities > 50:  # Moderate count = likely urban (lowered from 80)
            classification_score += 18
        elif total_key_facilities > 20:  # Some facilities = semi-urban (lowered from 30)
            classification_score += 12
        elif total_key_facilities > 5:  # Few facilities = sparse area (lowered from 10)
            classification_score += 6
        
        # Determine area type based on classification score - VERY LENIENT thresholds
        if classification_score >= 45:  # Much lower! (was 60)
            area_type = "Urban"
        elif classification_score >= 25:  # Much lower! (was 35)
            area_type = "Semi-Urban"
        else:
            area_type = "Rural"
        
        print(f"Area classified as: {area_type} (Score: {classification_score}/100, Facility Density: {facility_density:.1f}/km², Housing Density: {housing_density:.1f}/km²)")
        
        # AI Analysis with area-type specific guidance
        print("Calling Groq AI for comprehensive analysis...")
        
        # Customize recommendations based on area type
        if area_type == "Urban":
            area_context = """This is an URBAN area with high population density. Focus on:
- Healthcare: Hospitals within 2-3km radius
- Food: Supermarkets every 1-2km
- Transportation: Bus stops every 500m-1km
- Parks: Green spaces for every 5,000 residents
- Waste: Frequent collection points"""
        elif area_type == "Semi-Urban":
            area_context = """This is a SEMI-URBAN area with moderate density. Focus on:
- Healthcare: Clinics within 5km, hospitals within 10km
- Food: Grocery stores every 3-5km
- Transportation: Bus stops every 2-3km
- Parks: Community parks accessible within 2km
- Mixed residential and commercial development"""
        else:  # Rural
            area_context = f"""This is a RURAL area with low population density. DEVELOPMENT ROADMAP TO URBAN:

CURRENT STATUS:
- Classification Score: {classification_score}/100 (Need 70+ for Urban)
- Facility Density: {facility_density:.1f}/km² (Need 50+ for Urban)
- Housing Density: {housing_density:.1f}/km² (Need 100+ for Urban)
- Infrastructure Score: {infrastructure_score}/100 (Need 80+ for Urban)

PHASE 1 - BASIC INFRASTRUCTURE (0-2 years):
Priority: Establish foundational services
✓ Primary Health Center: 1 per 10-15km radius - CRITICAL
✓ Food Market/Store: 1 per 5-10km radius - HIGH PRIORITY
✓ Bus Stop/Transit Point: Every 10km on main roads - HIGH PRIORITY
✓ Electricity Grid: Expand to 90%+ coverage - CRITICAL
✓ Water Supply: Clean water access points - CRITICAL
✓ Road Connectivity: Paved roads to main areas - HIGH PRIORITY

PHASE 2 - COMMUNITY FACILITIES (2-5 years):
Priority: Build community infrastructure
✓ Community Health Center: Upgrade from primary to secondary care
✓ Grocery Stores: 2-3 medium-sized stores - MODERATE PRIORITY
✓ Schools: Primary + Secondary education facilities
✓ Post Office/Bank: Basic financial services
✓ Community Center: Gathering space for 500+ people
✓ Parks/Playgrounds: 1-2 recreational areas
✓ Waste Management: Collection points + disposal system - HIGH PRIORITY

PHASE 3 - ECONOMIC DEVELOPMENT (5-10 years):
Priority: Create employment and commerce
✓ Market Complex: Central marketplace for commerce
✓ Small Industries: Light manufacturing/processing units
✓ Agricultural Support: Cold storage, processing facilities
✓ Skill Development Center: Vocational training facility
✓ Banking Services: Full-service bank branches
✓ Internet/Telecom: High-speed connectivity - CRITICAL FOR GROWTH

PHASE 4 - URBANIZATION (10-15 years):
Priority: Transform to semi-urban/urban standards
✓ Hospital: Multi-specialty hospital facility
✓ Supermarkets: 3-5 modern retail stores
✓ Housing Development: Planned residential complexes (Target: 100+ units/km²)
✓ Public Transport: Regular bus services every 2-3km
✓ Commercial Hub: Shopping centers, offices
✓ Parks & Recreation: Green spaces every 2-3km
✓ Educational Institutions: Colleges, technical institutes
✓ Waste Treatment: Modern waste processing facility
✓ Industrial Estate: Organized industrial zone with pollution control

CRITICAL SUCCESS FACTORS:
1. Population Growth: Need to increase population density by 3-5x
2. Infrastructure Investment: ₹50-100 crore minimum over 10 years
3. Employment Creation: 500-1000 jobs to retain population
4. Road Network: Connect to major highways/cities
5. Services Access: Reduce travel time for essentials to < 30 minutes

IMMEDIATE NEXT STEPS (0-6 months):
1. Survey exact population and current infrastructure gaps
2. Identify land for Phase 1 facilities (5-10 hectares)
3. Secure funding from government schemes (PMAY, Smart Village, etc.)
4. Start with healthcare + transportation (highest impact)
5. Plan road network connecting all settlements

For recommendations, be SPECIFIC about:
- Exact number of facilities needed for each phase
- Priority order (what to build first)
- Approximate locations for new facilities (use coverage gaps)
- Expected timeline for development
- How each addition moves towards urban classification"""
        rural_roadmap_field = ""
        if area_type == "Rural":
            rural_roadmap_field = '"development_roadmap": {"phase1_0to2yrs": "Critical facilities to build first", "phase2_2to5yrs": "Community infrastructure next", "phase3_5to10yrs": "Economic development focus", "phase4_10to15yrs": "Urbanization targets", "investment_needed": "₹X crore estimate", "jobs_to_create": number},'
        
        ai_prompt = f"""Analyze this {area_type} area ({area_km2:.2f} km²) with these facilities:
Healthcare: {counts['healthcare']}, Food: {counts['food']}, Housing: {counts['housing']}, Transport: {counts['transportation']}, 
Parks: {counts['parks']}, Industrial: {counts['industrial']}, Water: {counts['water']}, Agriculture: {counts['agriculture']}, 
Waste: {counts['waste']}, Energy: {counts['energy']}

{area_context}

Provide JSON recommendations tailored to this {area_type} area type. {"For RURAL areas, include DEVELOPMENT ROADMAP with phased recommendations to achieve urban status. Be specific about what to build, where, and in what order." if area_type == "Rural" else "For Rural/Semi-Urban areas, ensure recommendations are practical and acknowledge lower density needs."}
{{
  "food_access": {{"status": "Good/Moderate/Poor", "additional_needed": number, "recommendation": "specific action for {area_type} area with exact locations", "urgency": "Critical/High/Moderate/Low"}},
  "housing": {{"status": "Adequate/Insufficient/Critical", "new_units_needed": number, "recommendation": "development strategy for {area_type} with target density"}},
  "transportation": {{"status": "Excellent/Good/Poor", "additional_stops_needed": number, "recommendation": "transit plan for {area_type} with exact spacing"}},
  "pollution": {{"risk_level": "High/Medium/Low", "air_concern": true/false, "water_concern": true/false, "recommendation": "monitoring strategy"}},
  "healthcare": {{"facilities_needed": number, "urgency": "Critical/High/Moderate/Low", "recommendation": "healthcare plan for {area_type} - specify facility types (PHC/CHC/Hospital)"}},
  "parks": {{"additional_needed": number, "recommendation": "green space strategy with target per capita"}},
  "growth": {{"priority": "High/Medium/Low", "zones": "development zones", "recommendation": "growth strategy for {area_type} with population target"}},
  "public_health": {{"heat_risk": "High/Medium/Low", "cold_risk": "High/Medium/Low", "recommendation": "resilience plan"}},
  "environment": {{"habitat_impact": "Severe/Moderate/Minimal", "recommendation": "conservation strategy"}},
  "agriculture": {{"expansion_needed": true/false, "hectares": number, "recommendation": "agricultural development for {area_type}"}},
  "waste": {{"status": "Good/Moderate/Poor", "additional_facilities": number, "recommendation": "waste management plan with collection frequency"}},
  "energy": {{"status": "Good/Moderate/Poor", "additional_infrastructure": number, "recommendation": "energy infrastructure plan - specify grid/solar/wind"}},
  {rural_roadmap_field}
  "overall_priority": "top 3-5 priorities for this {area_type} area with timeline"
}}"""
        
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an urban planner. Respond ONLY with valid JSON."},
                {"role": "user", "content": ai_prompt}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=2000
        )
        
        ai_response = chat_completion.choices[0].message.content.strip()
        
        # Parse AI response
        try:
            if '```json' in ai_response:
                ai_response = ai_response.split('```json')[1].split('```')[0].strip()
            elif '```' in ai_response:
                ai_response = ai_response.split('```')[1].split('```')[0].strip()
            
            recommendations = json.loads(ai_response)
            print("AI recommendations parsed successfully")
        except Exception as e:
            print(f"Error parsing AI: {str(e)}")
            recommendations = {
                "food_access": {"status": "Moderate", "additional_needed": 2, "recommendation": "Add grocery stores in underserved areas", "urgency": "Moderate"},
                "housing": {"status": "Adequate", "new_units_needed": 0, "recommendation": "Monitor growth"},
                "transportation": {"status": "Good", "additional_stops_needed": 0, "recommendation": "Maintain service"},
                "pollution": {"risk_level": "Low", "air_concern": False, "water_concern": False, "recommendation": "Continue monitoring"},
                "healthcare": {"facilities_needed": 1, "urgency": "Moderate", "recommendation": "Plan future facility"},
                "parks": {"additional_needed": 1, "recommendation": "Add community park"},
                "growth": {"priority": "Medium", "zones": "Central", "recommendation": "Focus on center"},
                "public_health": {"heat_risk": "Medium", "cold_risk": "Medium", "recommendation": "Prepare resources"},
                "environment": {"habitat_impact": "Minimal", "recommendation": "Maintain conservation"},
                "agriculture": {"expansion_needed": False, "hectares": 0, "recommendation": "Current balance OK"},
                "waste": {"status": "Moderate", "additional_facilities": 1, "recommendation": "Add recycling center"},
                "energy": {"status": "Good", "additional_infrastructure": 0, "recommendation": "Maintain infrastructure"},
                "overall_priority": "Balanced"
            }
        
        # INTELLIGENT AREA TYPE OVERRIDE FOR SUGGESTIONS
        # Large urban areas may be classified as "Rural" due to low density,
        # but we should provide urban-level suggestions based on absolute facility counts
        urban_facilities = counts['healthcare'] + counts['food'] + counts['transportation'] + counts['parks'] + counts['waste']
        
        # Override logic: Use absolute facility counts to detect urban characteristics
        suggestion_area_type = area_type  # Default to classification result
        
        if urban_facilities >= 200:
            # High facility count = definitely urban, use dense grid
            suggestion_area_type = "Urban"
            if area_type != "Urban":
                print(f"⚠️  OVERRIDE: Area classified as '{area_type}' but has {urban_facilities} facilities - using Urban grid for suggestions")
        elif urban_facilities >= 100 and area_type == "Rural":
            # Moderate facility count but classified as Rural = likely misclassified suburban area
            suggestion_area_type = "Semi-Urban"
            print(f"⚠️  OVERRIDE: Area classified as 'Rural' but has {urban_facilities} facilities - using Semi-Urban grid for suggestions")
        
        print(f"Classification: {area_type} | Suggestion Grid: {suggestion_area_type} | Urban Facilities: {urban_facilities}")
        
        # Generate REQUIREMENT-BASED suggestions (RED DOTS) with contextual intelligence
        print("🎯 Generating REQUIREMENT-BASED location suggestions...")
        suggestions = {}
        
        # Pass contextual data for intelligent placement
        if recommendations['healthcare']['facilities_needed'] > 0:
            suggestions['healthcare'] = find_optimal_locations(
                bounds, healthcare_locs, recommendations['healthcare']['facilities_needed'], 
                suggestion_area_type, housing_locs, transport_locs, industrial_locs
            )
        
        if recommendations['food_access']['additional_needed'] > 0:
            suggestions['food'] = find_optimal_locations(
                bounds, food_locs, recommendations['food_access']['additional_needed'], 
                suggestion_area_type, housing_locs, transport_locs, industrial_locs
            )
        
        if recommendations['parks']['additional_needed'] > 0:
            suggestions['parks'] = find_optimal_locations(
                bounds, parks_locs, recommendations['parks']['additional_needed'], 
                suggestion_area_type, housing_locs, transport_locs, industrial_locs
            )
        
        if recommendations['transportation']['additional_stops_needed'] > 0:
            suggestions['transportation'] = find_optimal_locations(
                bounds, transport_locs, recommendations['transportation']['additional_stops_needed'], 
                suggestion_area_type, housing_locs, transport_locs, industrial_locs
            )
        
        if recommendations['waste']['additional_facilities'] > 0:
            suggestions['waste'] = find_optimal_locations(
                bounds, waste_locs, recommendations['waste']['additional_facilities'], 
                suggestion_area_type, housing_locs, transport_locs, industrial_locs
            )
        
        if recommendations['energy']['additional_infrastructure'] > 0:
            suggestions['energy'] = find_optimal_locations(
                bounds, energy_locs, recommendations['energy']['additional_infrastructure'], 
                suggestion_area_type, housing_locs, transport_locs, industrial_locs
            )
        
        if recommendations['housing']['new_units_needed'] > 0:
            suggestions['housing'] = find_optimal_locations(
                bounds, housing_locs, min(5, recommendations['housing']['new_units_needed'] // 100), 
                suggestion_area_type, housing_locs, transport_locs, industrial_locs
            )
        
        print(f"Generated suggestions for {len(suggestions)} categories")
        print(f"Suggestion details: {[(cat, len(locs)) for cat, locs in suggestions.items()]}")
        
        return jsonify({
            'success': True,
            'area_type': area_type,
            'area_km2': round(area_km2, 2),
            'facility_density': round(facility_density, 2),
            'housing_density': round(housing_density, 2),
            'classification_score': classification_score,
            'infrastructure_score': infrastructure_score,
            'counts': counts,
            'facilities': {
                'healthcare': healthcare_locs,
                'food': food_locs,
                'housing': housing_locs,
                'transportation': transport_locs,
                'parks': parks_locs,
                'industrial': industrial_locs,
                'water': water_locs,
                'agriculture': agriculture_locs,
                'waste': waste_locs,
                'energy': energy_locs
            },
            'recommendations': recommendations,
            'suggestions': suggestions
        })
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting Satellite Image Segmentation Application...")
    
    # Load model on startup
    model_loaded = load_model()
    if not model_loaded:
        print("⚠️ Warning: Model not loaded. Some features may not work.")
    
    # Get port from environment variable (Render provides this)
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_ENV', 'production') != 'production'
    
    print(f"🌐 Application ready on port {port}")
    app.run(debug=debug_mode, host='0.0.0.0', port=port)

