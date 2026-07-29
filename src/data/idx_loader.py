# idx_loader.py
import numpy as np
import os

def load_idx_images(file_path, fix_emnist_orientation=False):
    """
    Load IDX image file.

    fix_emnist_orientation: EMNIST's raw byte layout is stored transposed
        relative to standard MNIST (a known artifact of how NIST generated
        the dataset). Every image needs an extra transpose after reshaping
        or it comes out rotated 90 degrees and mirrored. Standard MNIST
        files do NOT need this - only pass True for EMNIST digits/letters.
    """
    with open(file_path, 'rb') as f:
        magic = int.from_bytes(f.read(4), 'big')
        num_images = int.from_bytes(f.read(4), 'big')
        num_rows = int.from_bytes(f.read(4), 'big')
        num_cols = int.from_bytes(f.read(4), 'big')
        
        # Read image data
        image_size = num_rows * num_cols
        data = np.frombuffer(f.read(num_images * image_size), dtype=np.uint8)
        data = data.reshape(num_images, num_rows, num_cols)

        if fix_emnist_orientation:
            # Correct the transpose so images end up right-side-up, matching
            # the orientation of images app.py will feed the model at
            # inference time.
            data = np.transpose(data, axes=(0, 2, 1))

        return data

def load_idx_labels(file_path):
    """Load IDX label file"""
    with open(file_path, 'rb') as f:
        magic = int.from_bytes(f.read(4), 'big')
        num_labels = int.from_bytes(f.read(4), 'big')
        
        # Read label data
        data = np.frombuffer(f.read(num_labels), dtype=np.uint8)
        
        return data

def load_mnist_from_files(data_dir='dataset/mnist'):
    """Load MNIST from IDX files"""
    print("\n" + "="*60)
    print("LOADING MNIST FROM IDX FILES")
    print("="*60)
    
    # Paths to the files
    train_images_path = os.path.join(data_dir, 'train-images-idx3-ubyte')
    train_labels_path = os.path.join(data_dir, 'train-labels-idx1-ubyte')
    test_images_path = os.path.join(data_dir, 't10k-images-idx3-ubyte')
    test_labels_path = os.path.join(data_dir, 't10k-labels-idx1-ubyte')
    
    print("Loading training data...")
    x_train = load_idx_images(train_images_path)
    y_train = load_idx_labels(train_labels_path)
    print(f"Loaded {len(x_train)} images of size {x_train.shape[1]}x{x_train.shape[2]}")
    print(f"Loaded {len(y_train)} labels")
    
    print("Loading test data...")
    x_test = load_idx_images(test_images_path)
    y_test = load_idx_labels(test_labels_path)
    print(f"Loaded {len(x_test)} images of size {x_test.shape[1]}x{x_test.shape[2]}")
    print(f"Loaded {len(y_test)} labels")
    
    return (x_train, y_train), (x_test, y_test)

def load_emnist_digits(data_dir='dataset/emnist'):
    """Load EMNIST Digits from IDX files"""
    print("\n" + "="*60)
    print("LOADING EMNIST DIGITS FROM IDX FILES")
    print("="*60)
    
    # Paths to the files
    train_images_path = os.path.join(data_dir, 'emnist-digits-train-images-idx3-ubyte')
    train_labels_path = os.path.join(data_dir, 'emnist-digits-train-labels-idx1-ubyte')
    test_images_path = os.path.join(data_dir, 'emnist-digits-test-images-idx3-ubyte')
    test_labels_path = os.path.join(data_dir, 'emnist-digits-test-labels-idx1-ubyte')
    
    print("Loading training data...")
    x_train = load_idx_images(train_images_path, fix_emnist_orientation=True)
    y_train = load_idx_labels(train_labels_path)
    print(f"Loaded {len(x_train)} images of size {x_train.shape[1]}x{x_train.shape[2]}")
    print(f"Loaded {len(y_train)} labels")
    
    print("Loading test data...")
    x_test = load_idx_images(test_images_path, fix_emnist_orientation=True)
    y_test = load_idx_labels(test_labels_path)
    print(f"Loaded {len(x_test)} images of size {x_test.shape[1]}x{x_test.shape[2]}")
    print(f"Loaded {len(y_test)} labels")
    
    return (x_train, y_train), (x_test, y_test)

def load_emnist_letters(data_dir='dataset/emnist'):
    """Load EMNIST Letters from IDX files"""
    print("\n" + "="*60)
    print("LOADING EMNIST LETTERS FROM IDX FILES")
    print("="*60)
    
    # Paths to the files
    train_images_path = os.path.join(data_dir, 'emnist-letters-train-images-idx3-ubyte')
    train_labels_path = os.path.join(data_dir, 'emnist-letters-train-labels-idx1-ubyte')
    test_images_path = os.path.join(data_dir, 'emnist-letters-test-images-idx3-ubyte')
    test_labels_path = os.path.join(data_dir, 'emnist-letters-test-labels-idx1-ubyte')
    
    print("Loading training data...")
    x_train = load_idx_images(train_images_path, fix_emnist_orientation=True)
    y_train = load_idx_labels(train_labels_path)
    print(f"Loaded {len(x_train)} images of size {x_train.shape[1]}x{x_train.shape[2]}")
    print(f"Loaded {len(y_train)} labels")
    
    print("Loading test data...")
    x_test = load_idx_images(test_images_path, fix_emnist_orientation=True)
    y_test = load_idx_labels(test_labels_path)
    print(f"Loaded {len(x_test)} images of size {x_test.shape[1]}x{x_test.shape[2]}")
    print(f"Loaded {len(y_test)} labels")
    
    return (x_train, y_train), (x_test, y_test)

def combine_mnist_emnist_digits(mnist_dir='dataset/mnist', emnist_dir='dataset/emnist'):
    """Combine MNIST and EMNIST Digits datasets for training"""
    print("\n" + "="*60)
    print("COMBINING MNIST AND EMNIST DIGITS")
    print("="*60)
    
    # Load MNIST
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = load_mnist_from_files(mnist_dir)
    
    # Load EMNIST Digits
    (x_train_emnist, y_train_emnist), (x_test_emnist, y_test_emnist) = load_emnist_digits(emnist_dir)
    
    # Combine training data
    x_train_combined = np.concatenate([x_train_mnist, x_train_emnist], axis=0)
    y_train_combined = np.concatenate([y_train_mnist, y_train_emnist], axis=0)
    
    # Combine test data
    x_test_combined = np.concatenate([x_test_mnist, x_test_emnist], axis=0)
    y_test_combined = np.concatenate([y_test_mnist, y_test_emnist], axis=0)
    
    print(f"\nCombined Training Data: {x_train_combined.shape[0]} images")
    print(f"Combined Test Data: {x_test_combined.shape[0]} images")
    
    return (x_train_combined, y_train_combined), (x_test_combined, y_test_combined)