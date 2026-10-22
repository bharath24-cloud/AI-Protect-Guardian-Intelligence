from PIL import Image
from backend.app.storage.manager import storage_manager

def test_storage_save_and_retrieve():
    sample_bytes = b"sample image bytes 12345"
    filename, path = storage_manager.save_upload_bytes(sample_bytes, "test.png")
    
    assert path.exists()
    assert storage_manager.get_upload_path(filename) == path
    
    # Read back
    with open(path, "rb") as f:
        read_bytes = f.read()
    assert read_bytes == sample_bytes
    
    # Cleanup test file
    path.unlink()

def test_storage_save_pil():
    img = Image.new("RGB", (64, 64), color="blue")
    filename, path = storage_manager.save_pil_image(img, prefix="test_out", format="PNG")
    
    assert path.exists()
    assert storage_manager.get_processed_path(filename) == path
    
    path.unlink()
