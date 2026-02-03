from src.text_preprocessing import preprocess_text

def test_preprocess():
    assert "amazing" in preprocess_text("This is AMAZING!!!")
