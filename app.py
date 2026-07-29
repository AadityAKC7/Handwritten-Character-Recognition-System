# app.py - FIXED VERSION

import tensorflow as tf
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import os
import uuid
import sqlite3
from datetime import datetime
import json
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import cv2

app = Flask(__name__)

# FIX: secret key now comes from an environment variable instead of being
# hardcoded in source. Falls back to a random key so the app still runs
# locally, but you should set FLASK_SECRET_KEY in production.
app.secret_key = os.environ.get('FLASK_SECRET_KEY', os.urandom(32).hex())

# FIX: debug/host/port and whether debug-only routes are exposed are now
# controlled by an environment variable instead of being hardcoded.
DEBUG_MODE = os.environ.get('FLASK_DEBUG', '0') == '1'

# Configuration
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static/css', exist_ok=True)
os.makedirs('static/js', exist_ok=True)
os.makedirs('templates', exist_ok=True)

# Import prediction module
# NOTE: ImagePreprocessor (src/data/preprocess.py) used to be imported here
# too, but it was never actually called - process_image() below has its own
# self-contained preprocessing pipeline (resize -> threshold -> contour crop
# -> center -> dilate). Removed the dead import to avoid confusion about
# which preprocessing path is actually live.
from src.evaluation.predict import CharacterPredictor

predictor = None  # Will be initialized when model is loaded
CURRENT_MODEL_TYPE = 'letters'  # 'digits' or 'letters'


def load_model(model_type='letters'):
    """Load the trained model - specify 'digits' or 'letters'"""
    global predictor, CURRENT_MODEL_TYPE

    try:
        # FIX: both paths now use the same model/best_hcr_model_{type}.h5
        # convention, which is what train_model.py's ModelCheckpoint
        # actually writes. The old digits path ('model/hcr_digits_model.h5')
        # was never produced by any training script - training digits would
        # silently never be found by the app.
        if model_type == 'letters':
            model_paths = [
                'model/best_hcr_model_letters.h5',
            ]
            expected_classes = 26
            print("📊 Attempting to load LETTERS model (26 classes: A-Z)...")
        else:
            model_paths = [
                'model/best_hcr_model_digits.h5',
            ]
            expected_classes = 10
            print("📊 Attempting to load DIGITS model (10 classes: 0-9)...")

        for path in model_paths:
            if os.path.exists(path):
                print(f"   Found model at: {path}")
                predictor = CharacterPredictor(model_path=path)

                if predictor.model:
                    output_shape = predictor.model.output_shape[-1]
                    print(f"✅ Model loaded from {path}")
                    print(f"📊 Model has {output_shape} output classes")

                    if output_shape != expected_classes:
                        print(f"⚠️ Warning: Model has {output_shape} classes but expected {expected_classes}")

                    CURRENT_MODEL_TYPE = model_type
                    return True
                else:
                    print(f"   Model at {path} failed to load properly")

        print(f"❌ No {model_type} model found in any location")
        print(f"   Searched in: {model_paths}")
        return False

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def get_db_connection():
    """Get database connection"""
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize database tables if they don't exist"""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            full_name TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP,
            is_active INTEGER DEFAULT 1
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS recognition_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            image_path TEXT NOT NULL,
            predicted_character TEXT NOT NULL,
            confidence_score REAL NOT NULL,
            actual_character TEXT,
            is_correct INTEGER,
            recognition_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            processing_time_ms REAL
        )
    ''')

    cursor.execute("PRAGMA table_info(recognition_history)")
    columns = [column[1] for column in cursor.fetchall()]
    if 'model_used' not in columns:
        print("📝 Adding model_used column to recognition_history table...")
        cursor.execute("ALTER TABLE recognition_history ADD COLUMN model_used TEXT")
        print("✅ model_used column added")

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS model_metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT NOT NULL,
            model_version TEXT NOT NULL,
            accuracy REAL,
            training_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_active INTEGER DEFAULT 0
        )
    ''')

    conn.commit()
    conn.close()
    print("✅ Database initialized")


# Initialize database on startup
init_db()

# ==================== PAGE ROUTES ====================

@app.route('/')
def home():
    """Home page"""
    return render_template('home.html')


@app.route('/upload')
def upload_page():
    """Upload page"""
    return render_template('upload.html')


@app.route('/draw')
def draw_page():
    """Drawing canvas page"""
    return render_template('draw.html')


@app.route('/history')
def history_page():
    """History page"""
    conn = get_db_connection()
    try:
        history = conn.execute('''
            SELECT id, image_path, predicted_character, confidence_score, recognition_timestamp, model_used
            FROM recognition_history
            ORDER BY recognition_timestamp DESC
            LIMIT 50
        ''').fetchall()
    except sqlite3.OperationalError:
        history = conn.execute('''
            SELECT id, image_path, predicted_character, confidence_score, recognition_timestamp, NULL as model_used
            FROM recognition_history
            ORDER BY recognition_timestamp DESC
            LIMIT 50
        ''').fetchall()
    conn.close()
    return render_template('history.html', history=history)


# ==================== API ROUTES ====================

@app.route('/api/upload', methods=['POST'])
def upload_image():
    """API endpoint for image upload"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400

    ext = file.filename.rsplit('.', 1)[1].lower()
    filename = f"{uuid.uuid4().hex}.{ext}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    result = process_image(filepath)

    if result:
        return jsonify({
            'success': True,
            'filename': filename,
            'prediction': result['character'],
            'confidence': result['confidence'],
            'image_url': f'/static/uploads/{filename}',
            'model_used': result['model_used']
        })
    else:
        return jsonify({'error': 'Could not process image'}), 500


@app.route('/api/draw', methods=['POST'])
def process_drawing():
    """API endpoint for canvas drawing"""
    data = request.get_json(silent=True) or {}
    image_data = data.get('image')

    if not image_data:
        return jsonify({'error': 'No image data'}), 400

    if 'base64,' in image_data:
        image_data = image_data.split('base64,')[1]

    try:
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))
        image = image.convert('L')  # Convert to grayscale only

        # DON'T resize here - let process_image handle all preprocessing
        filename = f"drawing_{uuid.uuid4().hex}.png"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(filepath)  # Save the original size image

        result = process_image(filepath)

        if result:
            return jsonify({
                'success': True,
                'filename': filename,
                'prediction': result['character'],
                'confidence': result['confidence'],
                'image_url': f'/static/uploads/{filename}',
                'model_used': result['model_used']
            })
        else:
            return jsonify({'error': 'Could not process image'}), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 500


def process_image(image_path):
    """Process image and make prediction"""
    global predictor, CURRENT_MODEL_TYPE

    if predictor is None:
        if not load_model(CURRENT_MODEL_TYPE):
            return None

    try:
        # Read image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"❌ Could not read image: {image_path}")
            return None

        print(f"\n📸 Processing: {os.path.basename(image_path)}")
        print(f"   Using model: {CURRENT_MODEL_TYPE.upper()}")

        # Step 1: Resize to a larger size for better processing
        img = cv2.resize(img, (56, 56), interpolation=cv2.INTER_AREA)

        # Step 2: Apply Gaussian blur to reduce noise
        img = cv2.GaussianBlur(img, (3, 3), 0)

        # Step 3: Apply adaptive threshold for better binarization
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 11, 2)

        # Step 4: Morphological operations to clean up
        kernel = np.ones((2, 2), np.uint8)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

        # Step 5: Find contours
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest)

            padding = 4
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(img.shape[1] - x, w + 2 * padding)
            h = min(img.shape[0] - y, h + 2 * padding)

            character_img = img[y:y + h, x:x + w]
        else:
            character_img = img

        # Step 6: Resize to 28x28 while maintaining aspect ratio
        size = 28
        final_img = np.zeros((size, size), dtype=np.uint8)

        h, w = character_img.shape

        target_size = 20
        if w > h:
            scale = target_size / w
            new_w = target_size
            new_h = max(1, int(h * scale))
        else:
            scale = target_size / h
            new_h = target_size
            new_w = max(1, int(w * scale))

        resized = cv2.resize(character_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        x_offset = (size - new_w) // 2
        y_offset = (size - new_h) // 2
        final_img[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

        # Step 7: Ensure binary (0 or 255)
        _, final_img = cv2.threshold(final_img, 127, 255, cv2.THRESH_BINARY)

        # Step 8: Thicken the strokes slightly for better recognition
        kernel = np.ones((2, 2), np.uint8)
        final_img = cv2.dilate(final_img, kernel, iterations=1)

        # Step 9: Normalize to [0, 1]
        final_img = final_img.astype('float32') / 255.0

        # Step 10: Reshape for model
        final_input = final_img.reshape(1, 28, 28, 1)

        # FIX: debug image now gets its own unique filename built from the
        # original file's stem, instead of blindly replacing '.png'. The old
        # code did image_path.replace('.png', '_FINAL_INPUT.png'), which was
        # a no-op for any non-png upload (jpg/jpeg/gif/bmp) and silently
        # OVERWROTE the original uploaded image with the processed debug
        # version. Using os.path.splitext avoids that entirely.
        debug_img = (final_img * 255).astype(np.uint8)
        base, _ext = os.path.splitext(image_path)
        debug_path = f"{base}_FINAL_INPUT.png"
        cv2.imwrite(debug_path, debug_img)

        # FIX: removed the redundant second call to predictor.model.predict()
        # for digits. We now call the raw model exactly once and derive both
        # the character label and confidence from that single set of
        # predictions, so the two code paths can no longer disagree.
        raw_predictions = predictor.model.predict(final_input, verbose=0)

        if hasattr(predictor, 'predict_top_k'):
            top_k = predictor.predict_top_k(final_input, k=5)
            print("   Top 5 predictions:")
            for pred in top_k:
                print(f"     {pred['character']}: {pred['confidence']:.4f}")

        if CURRENT_MODEL_TYPE == 'digits':
            digit_index = int(np.argmax(raw_predictions[0]))
            character = str(digit_index)
            confidence = float(raw_predictions[0][digit_index])
            print(f"   🔢 Digit prediction: {character} (confidence: {confidence:.4f})")
        else:
            character, confidence = predictor.predict(final_input)

        print(f"   ✅ Final prediction: {character} (confidence: {confidence:.4f})")

        # Save to database
        conn = get_db_connection()
        try:
            conn.execute('''
                INSERT INTO recognition_history (user_id, image_path, predicted_character, confidence_score, model_used)
                VALUES (?, ?, ?, ?, ?)
            ''', (1, image_path, character, confidence, CURRENT_MODEL_TYPE))
        except Exception as e:
            print(f"⚠️ Could not insert with model_used: {e}")
            conn.execute('''
                INSERT INTO recognition_history (user_id, image_path, predicted_character, confidence_score)
                VALUES (?, ?, ?, ?)
            ''', (1, image_path, character, confidence))
        conn.commit()
        conn.close()

        return {
            'character': character,
            'confidence': confidence,
            'model_used': CURRENT_MODEL_TYPE
        }

    except Exception as e:
        print(f"❌ Error processing image: {e}")
        import traceback
        traceback.print_exc()
        return None


@app.route('/api/switch-model', methods=['POST'])
def switch_model():
    """API endpoint to switch between digits and letters model"""
    global CURRENT_MODEL_TYPE, predictor

    data = request.get_json(silent=True) or {}
    model_type = data.get('model_type', 'letters')

    if model_type not in ['digits', 'letters']:
        return jsonify({'error': 'Invalid model type'}), 400

    if CURRENT_MODEL_TYPE == model_type:
        return jsonify({
            'success': True,
            'message': f'Already using {model_type.upper()} model'
        })

    success = load_model(model_type)

    if success:
        return jsonify({
            'success': True,
            'model_type': model_type,
            'message': f'Successfully switched to {model_type.upper()} model'
        })
    else:
        return jsonify({
            'error': f'Could not load {model_type} model. Keeping {CURRENT_MODEL_TYPE.upper()} model active.',
            'current_model': CURRENT_MODEL_TYPE
        }), 500


@app.route('/api/current-model', methods=['GET'])
def get_current_model():
    """Get the currently active model type"""
    return jsonify({
        'model_type': CURRENT_MODEL_TYPE,
        'is_loaded': predictor is not None
    })


@app.route('/api/history', methods=['GET'])
def get_history():
    """Get recognition history as JSON"""
    conn = get_db_connection()
    try:
        history = conn.execute('''
            SELECT id, image_path, predicted_character, confidence_score, recognition_timestamp, model_used
            FROM recognition_history
            ORDER BY recognition_timestamp DESC
            LIMIT 50
        ''').fetchall()
    except sqlite3.OperationalError:
        history = conn.execute('''
            SELECT id, image_path, predicted_character, confidence_score, recognition_timestamp, NULL as model_used
            FROM recognition_history
            ORDER BY recognition_timestamp DESC
            LIMIT 50
        ''').fetchall()
    conn.close()
    return jsonify([dict(row) for row in history])


@app.route('/api/history/<int:id>', methods=['DELETE'])
def delete_history_item(id):
    """Delete a specific history item"""
    conn = get_db_connection()
    conn.execute('DELETE FROM recognition_history WHERE id = ?', (id,))
    conn.commit()
    conn.close()
    return jsonify({'success': True})


@app.route('/api/clear-history', methods=['POST'])
def clear_history():
    """Clear all recognition history"""
    conn = get_db_connection()
    conn.execute('DELETE FROM recognition_history')
    conn.commit()
    conn.close()
    return jsonify({'success': True})


# ==================== STATIC FILES ====================

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files"""
    from flask import send_from_directory
    return send_from_directory('static', filename)


# ==================== ERROR HANDLERS ====================

@app.errorhandler(404)
def not_found_error(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Page not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error'}), 500


# ==================== DEBUG ROUTES ====================
# FIX: all debug/introspection routes below are now gated behind DEBUG_MODE
# (FLASK_DEBUG=1). Previously they were reachable by anyone in production,
# leaked filesystem paths and model internals, and in the case of
# test-switch-and-predict, could flip the live model on a plain GET request
# with no confirmation step at all.

if DEBUG_MODE:

    @app.route('/api/debug-image', methods=['POST'])
    def debug_image():
        """Debug endpoint to see what the model sees"""
        data = request.get_json(silent=True) or {}
        image_data = data.get('image')

        # FIX: guard against a missing 'image' key, which previously caused
        # an unhandled TypeError ('base64,' in None).
        if not image_data:
            return jsonify({'error': 'No image data'}), 400

        if 'base64,' in image_data:
            image_data = image_data.split('base64,')[1]

        try:
            image_bytes = base64.b64decode(image_data)
            image = Image.open(BytesIO(image_bytes))
            image = image.convert('L')

            original_path = f"static/uploads/debug_original_{uuid.uuid4().hex}.png"
            image.save(original_path)

            img = np.array(image)
            img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)

            if np.mean(img) > 127:
                img = 255 - img

            processed_path = f"static/uploads/debug_processed_{uuid.uuid4().hex}.png"
            cv2.imwrite(processed_path, img)

            return jsonify({
                'original': original_path,
                'processed': processed_path,
                'mean_value': float(np.mean(img))
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 400

    @app.route('/api/check-models', methods=['GET'])
    def check_models():
        """Check which model files exist"""
        models_info = {'digits': [], 'letters': []}

        # NOTE: 'model/best_hcr_model_{type}.h5' is the path load_model()
        # actually uses. The other paths below are old/legacy names kept
        # here only so this debug route can tell you if stray files from a
        # previous run are still sitting around.
        digits_paths = [
            'model/best_hcr_model_digits.h5',
            'model/hcr_digits_model.h5',
            'models/hcr_digits_model.h5'
        ]

        for path in digits_paths:
            exists = os.path.exists(path)
            size = os.path.getsize(path) if exists else 0
            models_info['digits'].append({
                'path': path,
                'exists': exists,
                'size_mb': round(size / (1024 * 1024), 2) if exists else 0
            })

        letters_paths = [
            'model/best_hcr_model_letters.h5',
            'model/continued_model.h5',
            'model/hcr_letters_model_100epochs.h5',
            'models/hcr_letters_model.h5'
        ]

        for path in letters_paths:
            exists = os.path.exists(path)
            size = os.path.getsize(path) if exists else 0
            models_info['letters'].append({
                'path': path,
                'exists': exists,
                'size_mb': round(size / (1024 * 1024), 2) if exists else 0
            })

        return jsonify({
            'current_model': CURRENT_MODEL_TYPE,
            'predictor_loaded': predictor is not None,
            'models': models_info
        })

    @app.route('/api/test-load-digits', methods=['GET'])
    def test_load_digits():
        """Test loading digits model and show detailed error"""
        import traceback

        results = []

        digits_paths = [
            'model/best_hcr_model_digits.h5',
            'model/hcr_digits_model.h5',
            'models/hcr_digits_model.h5'
        ]

        for path in digits_paths:
            result = {'path': path, 'exists': os.path.exists(path)}

            if os.path.exists(path):
                try:
                    model = tf.keras.models.load_model(path)
                    result['success'] = True
                    result['output_shape'] = str(model.output_shape)
                    result['input_shape'] = str(model.input_shape)
                    result['num_classes'] = model.output_shape[-1]

                    test_predictor = CharacterPredictor(model_path=path)
                    result['predictor_created'] = test_predictor.model is not None

                    if test_predictor.model:
                        test_input = np.random.rand(1, 28, 28, 1).astype('float32')
                        try:
                            output = test_predictor.model.predict(test_input, verbose=0)
                            result['prediction_works'] = True
                            result['prediction_shape'] = str(output.shape)
                        except Exception as e:
                            result['prediction_works'] = False
                            result['prediction_error'] = str(e)

                except Exception as e:
                    result['success'] = False
                    result['error'] = str(e)
                    result['traceback'] = traceback.format_exc()
            else:
                result['success'] = False
                result['error'] = 'File not found'

            results.append(result)

        return jsonify({
            'current_model_type': CURRENT_MODEL_TYPE,
            'predictor_loaded': predictor is not None,
            'test_results': results
        })

    # FIX: changed from GET to POST. This route mutates server state (it
    # calls load_model('digits') and leaves CURRENT_MODEL_TYPE switched), so
    # it should never be triggerable by a plain GET (link click, prefetch,
    # crawler). It's also still gated behind DEBUG_MODE like the rest of
    # this block.
    @app.route('/api/test-switch-and-predict', methods=['POST'])
    def test_switch_and_predict():
        """Test switching to digits and making a prediction"""
        global predictor, CURRENT_MODEL_TYPE

        results = {
            'initial_model': CURRENT_MODEL_TYPE,
            'steps': []
        }

        results['steps'].append({'action': 'Loading digits model'})
        if load_model('digits'):
            results['steps'][-1]['success'] = True
            results['steps'][-1]['model_loaded'] = CURRENT_MODEL_TYPE

            test_img = np.zeros((56, 56), dtype=np.uint8)
            cv2.line(test_img, (15, 10), (40, 10), 255, 3)
            cv2.line(test_img, (40, 10), (40, 30), 255, 3)
            cv2.line(test_img, (40, 30), (15, 30), 255, 3)
            cv2.line(test_img, (15, 30), (15, 45), 255, 3)
            cv2.line(test_img, (15, 45), (40, 45), 255, 3)

            os.makedirs('static/uploads', exist_ok=True)
            test_path = 'static/uploads/test_digit.png'
            cv2.imwrite(test_path, test_img)

            results['steps'].append({'action': 'Processing test image'})
            try:
                result = process_image(test_path)
                if result:
                    results['steps'][-1]['success'] = True
                    results['steps'][-1]['prediction'] = result['character']
                    results['steps'][-1]['confidence'] = result['confidence']
                    results['steps'][-1]['model_used'] = result['model_used']
                else:
                    results['steps'][-1]['success'] = False
                    results['steps'][-1]['error'] = 'process_image returned None'
            except Exception as e:
                results['steps'][-1]['success'] = False
                results['steps'][-1]['error'] = str(e)
        else:
            results['steps'][0]['success'] = False
            results['steps'][0]['error'] = 'Failed to load digits model'

        return jsonify(results)


# ==================== MAIN ====================

if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("Starting Handwritten Character Recognition System")
    print("=" * 50)

    if load_model('letters'):
        print(f"✅ System ready with {CURRENT_MODEL_TYPE.upper()} model")
    else:
        print("⚠️ Could not load letters model, trying digits model...")
        if load_model('digits'):
            print(f"✅ System ready with {CURRENT_MODEL_TYPE.upper()} model")
        else:
            print("❌ No model found. Please train a model first.")
            print("   Run: python src/train/train_model.py")

    print("\n" + "=" * 50)
    print("Available Routes:")
    print("=" * 50)
    for rule in app.url_map.iter_rules():
        if not rule.endpoint.startswith('static'):
            print(f"  {rule.endpoint}: {rule.methods} {rule}")
    print("=" * 50 + "\n")

    # FIX: debug mode and bind host are now controlled by env vars instead
    # of being hardcoded. Set FLASK_DEBUG=1 locally for the Werkzeug
    # debugger + debug-only routes; leave unset (or 0) in production so the
    # interactive debugger is never exposed on the network.
    app.run(debug=DEBUG_MODE, host=os.environ.get('FLASK_HOST', '127.0.0.1'), port=int(os.environ.get('FLASK_PORT', 5000)))